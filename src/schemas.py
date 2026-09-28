"""Validated domain models for extraction, QA, review, and evaluation."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared output and validates later assignment."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DocumentationStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_DOCUMENTED = "not_documented"
    CANNOT_BE_ASSIGNED = "cannot_be_assigned"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"
    UNSUPPORTED = "unsupported"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


class ReviewAction(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    FLAGGED = "flagged_for_review"


CoreVariableName = Literal[
    "histologic_diagnosis",
    "tumor_size",
    "pathological_t_category",
    "pathological_n_category",
]

CORE_VARIABLES: tuple[CoreVariableName, ...] = (
    "histologic_diagnosis",
    "tumor_size",
    "pathological_t_category",
    "pathological_n_category",
)

CancerType = Literal["LUAD", "LUSC"]
ReviewPriority = Literal["low", "medium", "high", "critical"]

# Lung TNM categories supported by this prototype. Subcategories that do not
# exist for lung cancer are deliberately excluded rather than accepted by a
# permissive generic stage regex.
LUNG_PT_SUFFIX_PATTERN = r"(?:1mi|is|X|0|1[abc]?|2[ab]?|3|4)"
LUNG_PN_SUFFIX_PATTERN = r"(?:X|[0-3])"
LUNG_PT_CATEGORIES = frozenset(
    {"pTX", "pT0", "pTis", "pT1mi", "pT1", "pT1a", "pT1b", "pT1c", "pT2", "pT2a", "pT2b", "pT3", "pT4"}
)
LUNG_PN_CATEGORIES = frozenset({"pNX", "pN0", "pN1", "pN2", "pN3"})


def _value_evidence_pattern(
    variable_name: CoreVariableName,
    value: str,
) -> str | None:
    """Return a boundary-safe regex for one canonical value."""

    normalized_value = re.sub(r"\s+", " ", value.casefold()).strip()
    if variable_name == "tumor_size":
        number = re.search(r"\d+(?:\.\d+)?", normalized_value)
        if not number:
            return None
        value_number = re.escape(number.group(0))
        # Synoptic reports commonly state the greatest dimension as the first
        # number in a multi-dimensional measurement (for example,
        # ``3.3 x 2.5 cm``). The canonical value remains ``3.3 cm``, while the
        # exact evidence must retain the original report wording.
        return (
            rf"(?<![\d.]){value_number}(?:\s*cm\b|"
            rf"(?:\s*[x×]\s*\d+(?:\.\d+)?){{1,2}}\s*cm\b)"
        )
    if variable_name in {"pathological_t_category", "pathological_n_category"}:
        return rf"\b{re.escape(normalized_value)}\b"
    meaningful_tokens = {
        token
        for token in re.findall(r"[a-z]+", normalized_value)
        if token not in {"invasive", "cell", "carcinoma"}
    }
    core_token = max(meaningful_tokens, key=len) if meaningful_tokens else normalized_value
    return rf"\b{re.escape(core_token)}\b" if core_token else None


def _evidence_text_lexically_states_value(
    variable_name: CoreVariableName,
    value: str,
    evidence_text: str,
) -> bool:
    """Check literal value presence without interpreting clinical context."""

    normalized_value = re.sub(r"\s+", " ", value.casefold()).strip()
    normalized_evidence = evidence_text.casefold()
    pattern = _value_evidence_pattern(variable_name, value)
    if pattern is None:
        return False
    if variable_name != "histologic_diagnosis":
        return bool(re.search(pattern, normalized_evidence, re.IGNORECASE))
    meaningful_tokens = {
        token
        for token in re.findall(r"[a-z]+", normalized_value)
        if token not in {"invasive", "cell", "carcinoma"}
    }
    return normalized_value in normalized_evidence or bool(
        meaningful_tokens
        and meaningful_tokens.issubset(set(re.findall(r"[a-z]+", normalized_evidence)))
    )


def _status_target_pattern(
    variable_name: CoreVariableName,
    value: str | None,
) -> str:
    """Return a regex that ties a documentation cue to its intended field."""

    if value:
        value_pattern = _value_evidence_pattern(variable_name, value)
        if value_pattern:
            return value_pattern
    if variable_name == "histologic_diagnosis":
        return (
            r"(?:adenocarcinoma|squamous\s+cell\s+carcinoma|large\s+cell\s+carcinoma|"
            r"poorly\s+differentiated\s+carcinoma|carcinoma|malignan\w*|tumou?r)"
        )
    if variable_name == "tumor_size":
        return r"(?:tumou?r(?:\s+size)?|mass|lesion|measurement|size)"
    if variable_name == "pathological_t_category":
        return rf"(?:pathological\s+T\s+category|pT{LUNG_PT_SUFFIX_PATTERN})"
    return rf"(?:pathological\s+N\s+category|pN{LUNG_PN_SUFFIX_PATTERN}|regional\s+lymph\s+nodes?)"


def evidence_text_supports_value(
    variable_name: CoreVariableName,
    value: str,
    evidence_text: str,
    context_text: str | None = None,
) -> bool:
    """Verify lexical support and reject obvious wrong-context evidence."""

    normalized_value = value.casefold()
    normalized_evidence = evidence_text.casefold()
    normalized_context = (context_text or evidence_text).casefold()
    candidate_pattern = _value_evidence_pattern(variable_name, value)
    if candidate_pattern is None or not _evidence_text_lexically_states_value(
        variable_name, value, evidence_text
    ):
        return False
    if variable_name == "tumor_size":
        number = re.search(r"\d+(?:\.\d+)?", normalized_value)
        assert number is not None
        if "margin" in normalized_context and "tumor size" not in normalized_context:
            return False
        measurement_patterns = (
            rf"\btumou?r\s+size\b[^.\n]{{0,80}}{candidate_pattern}",
            rf"\b(?:tumou?r|mass|lesion|carcinoma)\s+(?:is\s+|measur\w*\s+)?{candidate_pattern}",
            rf"\b(?:amended|revised|estimated|maximum)\s+size\b[^.\n]{{0,50}}{candidate_pattern}",
            rf"\bsize\s*(?::|is)?\s*(?:approximately\s+|about\s+)?{candidate_pattern}",
            rf"\bgreatest\s+dimension(?:\s+of\s+(?:the\s+)?tumou?r)?\s*(?::|is|=)?\s*{candidate_pattern}",
            rf"\bgreatest\s+diameter(?:\s+of\s+(?:the\s+)?tumou?r)?\s*(?::|is|=)?\s*{candidate_pattern}",
            rf"\b(?:size|dimension)\s+of\s+(?:invasive\s+)?(?:carcinoma|tumou?r)\s*(?::|is|=)?\s*{candidate_pattern}",
            rf"{candidate_pattern}[^.\n]{{0,35}}\b(?:greatest\s+dimension|tumou?r|mass|lesion)\b",
            rf"\b(?:gross|microscopic|synoptic|narrative)\s+(?:section\s+)?records\s+{candidate_pattern}",
        )
        lexical_support = any(
            re.search(pattern, normalized_context) for pattern in measurement_patterns
        )
    elif variable_name in {"pathological_t_category", "pathological_n_category"}:
        lexical_support = True
    else:
        lexical_support = True

    if not lexical_support:
        return False

    negated_patterns = (
        rf"\b(?:no\s+evidence\s+of|negative\s+for|absence\s+of|without)\b[^.\n]{{0,60}}{candidate_pattern}",
        rf"\bno\b[^.\n]{{0,25}}{candidate_pattern}",
        rf"{candidate_pattern}[^.\n]{{0,45}}\b(?:not\s+(?:identified|present|found|detected)|ruled\s+out|absent)\b",
    )
    historical_patterns = (
        rf"\b(?:prior|previous|previously|former|formerly|original|originally)\b[^.\n]{{0,45}}{candidate_pattern}",
        rf"\b(?:amended|revised|corrected|changed|updated)\s+from\s+{candidate_pattern}",
        rf"{candidate_pattern}[^.\n]{{0,45}}\b(?:was\s+)?(?:amended|revised|corrected|changed|updated|superseded)\s+(?:to|by)\b",
    )
    return not any(
        re.search(pattern, normalized_context)
        for pattern in (*negated_patterns, *historical_patterns)
    )


def local_evidence_context(report_text: str, start_offset: int, end_offset: int) -> str:
    """Return the containing sentence/line for context-sensitive evidence checks."""

    if not (0 <= start_offset < end_offset <= len(report_text)):
        return ""
    prefix = report_text[:start_offset]
    starts = [prefix.rfind("\n") + 1]
    for marker in (". ", "? ", "! "):
        found = prefix.rfind(marker)
        if found >= 0:
            starts.append(found + len(marker))
    statement_start = max(starts)
    suffix = report_text[end_offset:]
    boundary = re.search(r"(?:[.!?](?=\s|$)|\n)", suffix)
    statement_end = end_offset + (boundary.end() if boundary else len(suffix))
    return report_text[statement_start:statement_end]


def _documented_candidates(variable_name: CoreVariableName, text: str) -> set[str]:
    if variable_name == "tumor_size":
        return {match.casefold() for match in re.findall(r"\d+(?:\.\d+)?\s*cm\b", text, re.IGNORECASE)}
    if variable_name == "pathological_t_category":
        return {match.casefold() for match in re.findall(rf"\bpT{LUNG_PT_SUFFIX_PATTERN}\b", text, re.IGNORECASE)}
    if variable_name == "pathological_n_category":
        return {match.casefold() for match in re.findall(rf"\bpN{LUNG_PN_SUFFIX_PATTERN}\b", text, re.IGNORECASE)}
    return {
        match.casefold()
        for match in re.findall(
            r"\b(?:minimally\s+invasive\s+adenocarcinoma|(?:invasive\s+)?adenocarcinoma|"
            r"(?:invasive\s+)?squamous\s+cell\s+carcinoma|large\s+cell\s+carcinoma|"
            r"poorly\s+differentiated\s+carcinoma)\b",
            text,
            re.IGNORECASE,
        )
    }


def _context_has_tied_cue(context: str, target_pattern: str, cue_pattern: str) -> bool:
    return bool(
        re.search(
            rf"(?:{target_pattern})[^.\n]{{0,70}}(?:{cue_pattern})|"
            rf"(?:{cue_pattern})[^.\n]{{0,70}}(?:{target_pattern})",
            context,
            re.IGNORECASE,
        )
    )


def _context_supersedes_value(
    variable_name: CoreVariableName,
    value: str,
    evidence_text: str,
    context_text: str,
) -> bool:
    """Return true only when ``value`` is the outdated side of a revision."""

    candidate_pattern = _value_evidence_pattern(variable_name, value)
    if candidate_pattern is None or not _evidence_text_lexically_states_value(
        variable_name, value, evidence_text
    ):
        return False
    return bool(
        re.search(
            rf"(?:{candidate_pattern})[^.\n]{{0,55}}\b(?:was\s+)?"
            r"(?:superseded|replaced|amended|revised|corrected|updated|changed)"
            r"\s+(?:to|by)\b|"
            r"\b(?:amended|revised|corrected|changed|updated)\s+from\s+"
            rf"(?:{candidate_pattern})|"
            r"\b(?:supersedes|replaces)\s+"
            rf"(?:{candidate_pattern})",
            context_text,
            re.IGNORECASE,
        )
    )


def evidence_text_supports_status(
    variable_name: CoreVariableName,
    status: DocumentationStatus,
    evidence_texts: list[str],
    *,
    value: str | None = None,
    alternatives: list[str] | None = None,
    context_texts: list[str] | None = None,
) -> bool:
    """Check that evidence carries the semantics claimed by a QA status."""

    contexts = context_texts or evidence_texts
    combined = "\n".join(contexts)
    lowered = combined.casefold()
    value_supported = value is not None and any(
        evidence_text_supports_value(
            variable_name,
            value,
            evidence,
            contexts[index] if index < len(contexts) else evidence,
        )
        for index, evidence in enumerate(evidence_texts)
    )

    target_pattern = _status_target_pattern(variable_name, value)
    uncertainty_pattern = (
        r"\b(?:uncertain|approximately|approx\.?|estimated?|likely|possible|"
        r"probable|suspected|equivocal|suggestive|favou?red|provisional|pending)\b"
    )
    if status == DocumentationStatus.SUPPORTED:
        has_target_uncertainty = any(
            _context_has_tied_cue(context, target_pattern, uncertainty_pattern)
            for context in contexts
        )
        has_target_supersession = value is not None and any(
            _context_supersedes_value(variable_name, value, evidence, context)
            for evidence, context in zip(evidence_texts, contexts)
        )
        return value_supported and not has_target_uncertainty and not has_target_supersession
    if status == DocumentationStatus.UNCERTAIN:
        return value_supported and any(
            _context_has_tied_cue(context, target_pattern, uncertainty_pattern)
            for context in contexts
        )
    if status == DocumentationStatus.SUPERSEDED:
        if value is None:
            return False
        return any(
            _context_supersedes_value(variable_name, value, evidence, context)
            for evidence, context in zip(evidence_texts, contexts)
        )
    if status == DocumentationStatus.NEGATED:
        return any(
            re.search(
                rf"\b(?:no\s+evidence\s+of|negative\s+for|absence\s+of|without)\b"
                rf"[^.\n]{{0,60}}(?:{target_pattern})|"
                rf"\bno(?:\s+[a-z]+){{0,3}}\s+(?:{target_pattern})\b|"
                rf"(?:{target_pattern})[^.\n]{{0,55}}\b(?:not\s+"
                r"(?:identified|present|found|detected|documented)|absent|ruled\s+out)\b",
                context,
                re.IGNORECASE,
            )
            for context in contexts
        )
    if status == DocumentationStatus.CANNOT_BE_ASSIGNED:
        explicit_x = (value == "pNX" and "pnx" in lowered) or (
            value == "pTX" and "ptx" in lowered
        )
        assignment_pattern = (
            r"\b(?:cannot\s+be\s+(?:assigned|assessed)|"
            r"not\s+(?:assessed|assignable))\b"
        )
        tied_assignment = any(
            re.search(
                rf"(?:{target_pattern})[^.\n]{{0,70}}(?:{assignment_pattern})|"
                rf"(?:{assignment_pattern})[^.\n]{{0,70}}(?:{target_pattern})",
                context,
                re.IGNORECASE,
            )
            for context in contexts
        )
        no_nodes = variable_name == "pathological_n_category" and bool(
            re.search(
                r"\bno\s+(?:regional\s+)?lymph\s+nodes?[^.\n]{0,45}"
                r"(?:submitted|identified|assessed|evaluated)\b",
                lowered,
            )
        )
        return explicit_x or tied_assignment or no_nodes
    if status == DocumentationStatus.CONFLICTING:
        if alternatives:
            return (
                len(alternatives) >= 2
                and len(alternatives) == len(evidence_texts)
                and len({item.casefold().strip() for item in alternatives}) == len(alternatives)
                and all(
                    evidence_text_supports_value(
                        variable_name,
                        alternative,
                        evidence_texts[index],
                        contexts[index] if index < len(contexts) else evidence_texts[index],
                    )
                    for index, alternative in enumerate(alternatives)
                )
            )
        candidates = _documented_candidates(variable_name, combined)
        return len(candidates) >= 2 and (
            len(evidence_texts) >= 2
            or bool(re.search(r"\b(?:conflicting|conflict|discrepancy|different|versus|vs\.?|another)\b", lowered))
        )
    if status == DocumentationStatus.NOT_DOCUMENTED:
        if not evidence_texts:
            return True
        missing_pattern = r"\b(?:not\s+documented|not\s+stated|not\s+provided)\b"
        return any(
            re.search(
                rf"(?:{target_pattern})[^.\n]{{0,55}}(?:{missing_pattern})|"
                rf"(?:{missing_pattern})[^.\n]{{0,55}}(?:{target_pattern})",
                context,
                re.IGNORECASE,
            )
            for context in contexts
        )
    if status == DocumentationStatus.UNSUPPORTED:
        if not evidence_texts:
            return True
        unsupported_pattern = (
            r"\b(?:unsupported|not\s+supported|insufficient\s+evidence|"
            r"unable\s+to\s+(?:verify|validate|support)|cannot\s+(?:confirm|verify|validate))\b"
        )
        return any(
            _context_has_tied_cue(context, target_pattern, unsupported_pattern)
            for context in contexts
        )
    if status == DocumentationStatus.MANUAL_REVIEW_REQUIRED:
        if not evidence_texts:
            return True
        review_pattern = (
            r"\b(?:manual\s+review\s+(?:is\s+)?required|requires?\s+manual\s+review|"
            r"review\s+manually|needs?\s+(?:manual\s+)?review|refer(?:red)?\s+for\s+review)\b"
        )
        return any(
            _context_has_tied_cue(context, target_pattern, review_pattern)
            for context in contexts
        )
    return False


class EvidenceSpan(StrictModel):
    """An exact character span copied from the submitted report."""

    text: str = Field(..., min_length=1)
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., gt=0)
    page_number: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def valid_character_span(self) -> "EvidenceSpan":
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("offset span length must equal the evidence text length")
        return self


class TextTransformation(StrictModel):
    """One conservative, reviewable text-normalization operation."""

    transformation: str = Field(..., min_length=1)
    count: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)


class OCRQualityAssessment(StrictModel):
    """Transparent page-level extraction quality assessment."""

    label: Literal["Good", "Review recommended", "Poor"]
    score: float = Field(..., ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


class ProcessedPage(StrictModel):
    """Extracted and reviewer-controlled state for a single PDF page."""

    page_number: int = Field(..., ge=1)
    extraction_method: Literal["native_text", "ocr"]
    raw_text: str = ""
    normalized_text: str = ""
    corrected_text: str | None = None
    accepted: bool = False
    excluded_as_blank: bool = False
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    ocr_language: str | None = None
    ocr_page_segmentation: int | None = Field(default=None, ge=3, le=13)
    character_count: int = Field(..., ge=0)
    quality: OCRQualityAssessment
    warning_flags: list[str] = Field(default_factory=list)
    transformations: list[TextTransformation] = Field(default_factory=list)

    @property
    def authoritative_text(self) -> str:
        if self.excluded_as_blank:
            return ""
        return self.corrected_text if self.corrected_text is not None else self.raw_text

    @model_validator(mode="after")
    def coherent_review_state(self) -> "ProcessedPage":
        if self.excluded_as_blank and not self.accepted:
            raise ValueError("a page excluded as blank must be explicitly accepted")
        return self


class DocumentProvenance(StrictModel):
    """Document-level provenance retained through extraction and review."""

    source_report_digest: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")
    extraction_timestamp: datetime
    processor_version: str = Field(..., min_length=1)
    page_count: int = Field(..., ge=1)
    native_text_pages: int = Field(..., ge=0)
    ocr_pages: int = Field(..., ge=0)
    reviewer_accepted_at: datetime | None = None


class ProcessedDocument(StrictModel):
    """Validated, page-aware PDF processing result."""

    document_id: str = Field(..., pattern=r"^TCGA-PDF-[0-9A-F]{12}$")
    pages: list[ProcessedPage] = Field(..., min_length=1)
    provenance: DocumentProvenance

    @model_validator(mode="after")
    def coherent_pages(self) -> "ProcessedDocument":
        expected = list(range(1, len(self.pages) + 1))
        if [page.page_number for page in self.pages] != expected:
            raise ValueError("processed pages must be sequential and one-indexed")
        if self.provenance.page_count != len(self.pages):
            raise ValueError("provenance page_count must match processed pages")
        return self


class VariableExtraction(StrictModel):
    """Extraction result for one canonical variable."""

    variable_name: CoreVariableName
    extracted_value: str | None = None
    documentation_status: DocumentationStatus
    evidence: list[EvidenceSpan] | None = Field(default=None, min_length=1)
    alternatives: list[str] = Field(default_factory=list, max_length=8)
    notes: str | None = None

    @model_validator(mode="after")
    def valid_status_contract(self) -> "VariableExtraction":
        value = self.extracted_value
        status = self.documentation_status
        alternatives: list[str] = []
        for alternative in self.alternatives:
            normalized = alternative.strip()
            if not normalized:
                raise ValueError("conflicting alternative values cannot be blank")
            alternatives.append(normalized)
        object.__setattr__(self, "alternatives", alternatives)

        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("extracted_value cannot be blank")
            object.__setattr__(self, "extracted_value", value)
            if not self.evidence:
                raise ValueError("every non-null extracted_value requires evidence")

        if status in {
            DocumentationStatus.SUPPORTED,
            DocumentationStatus.UNCERTAIN,
            DocumentationStatus.SUPERSEDED,
        } and value is None:
            raise ValueError(f"{status.value} status requires a non-null extracted_value")
        if status in {
            DocumentationStatus.NOT_DOCUMENTED,
            DocumentationStatus.NEGATED,
            DocumentationStatus.CONFLICTING,
            DocumentationStatus.UNSUPPORTED,
            DocumentationStatus.MANUAL_REVIEW_REQUIRED,
        } and value is not None:
            raise ValueError(f"{status.value} status requires abstention")
        if status == DocumentationStatus.CANNOT_BE_ASSIGNED and not self.evidence:
            raise ValueError("cannot_be_assigned status requires exact evidence")
        if status == DocumentationStatus.NEGATED and not self.evidence:
            raise ValueError("negated status requires exact negation evidence")
        if status == DocumentationStatus.CONFLICTING:
            evidence = self.evidence or []
            if len(alternatives) < 2 or len(evidence) < 2:
                raise ValueError(
                    "conflicting status requires at least two paired alternatives and evidence spans"
                )
            if len(alternatives) != len(evidence):
                raise ValueError(
                    "conflicting status requires one evidence span per alternative"
                )
            if len({item.casefold() for item in alternatives}) != len(alternatives):
                raise ValueError("conflicting alternative values must be distinct")
            if len({(span.start_offset, span.end_offset) for span in evidence}) != len(evidence):
                raise ValueError("conflicting evidence spans must be distinct")
            for index, (alternative, span) in enumerate(
                zip(alternatives, evidence), start=1
            ):
                if self.variable_name == "tumor_size" and not re.fullmatch(
                    r"\d+(?:\.\d+)? cm", alternative
                ):
                    raise ValueError(
                        f"conflicting alternative {index} must use canonical '<number> cm' notation"
                    )
                if (
                    self.variable_name == "pathological_t_category"
                    and alternative not in LUNG_PT_CATEGORIES
                ):
                    raise ValueError(
                        f"conflicting alternative {index} must use canonical lung pT notation"
                    )
                if (
                    self.variable_name == "pathological_n_category"
                    and alternative not in LUNG_PN_CATEGORIES
                ):
                    raise ValueError(
                        f"conflicting alternative {index} must use canonical lung pN notation"
                    )
                if not evidence_text_supports_value(
                    self.variable_name, alternative, span.text
                ):
                    raise ValueError(
                        f"conflicting evidence span {index} does not support its paired alternative"
                    )
        elif alternatives:
            raise ValueError("alternatives are only valid for conflicting status")
        if status in {DocumentationStatus.UNCERTAIN, DocumentationStatus.SUPERSEDED} and not self.evidence:
            raise ValueError(f"{status.value} status requires evidence")

        if self.variable_name == "pathological_t_category" and value is not None:
            if value not in LUNG_PT_CATEGORIES:
                raise ValueError("pathological T category must use canonical pT notation")
            if status == DocumentationStatus.CANNOT_BE_ASSIGNED and value != "pTX":
                raise ValueError("only explicit pTX may carry a value with cannot_be_assigned")
            if value == "pTX" and status != DocumentationStatus.CANNOT_BE_ASSIGNED:
                raise ValueError("explicit pTX must use cannot_be_assigned status")
        if self.variable_name == "tumor_size" and value is not None:
            if not re.fullmatch(r"\d+(?:\.\d+)? cm", value):
                raise ValueError("tumor size must use canonical '<number> cm' notation")
        if self.variable_name == "pathological_n_category" and value is not None:
            if value not in LUNG_PN_CATEGORIES:
                raise ValueError("pathological N category must use canonical pN notation")
            if status == DocumentationStatus.CANNOT_BE_ASSIGNED and value != "pNX":
                raise ValueError("only explicit pNX may carry a value with cannot_be_assigned")
            if value == "pNX" and status != DocumentationStatus.CANNOT_BE_ASSIGNED:
                raise ValueError("explicit pNX must use cannot_be_assigned status")
        if self.variable_name not in {
            "pathological_t_category",
            "pathological_n_category",
        } and status == DocumentationStatus.CANNOT_BE_ASSIGNED and value is not None:
            raise ValueError("cannot_be_assigned values are only supported for explicit pTX or pNX")
        return self


class QAIssueType(str, Enum):
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"
    NEGATED = "negated"
    UNSUPPORTED_EVIDENCE = "unsupported_evidence"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    OFFSET_MISMATCH = "offset_mismatch"
    MANUAL_REVIEW = "manual_review_required"
    OCR_QUALITY = "ocr_quality"


class QAIssue(StrictModel):
    issue_type: QAIssueType
    variable_name: CoreVariableName | None = None
    description: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"]
    evidence: list[EvidenceSpan] | None = Field(default=None, min_length=1)
    suggestion: str | None = None


class ExtractionResult(StrictModel):
    # Container updates are assembled by the QA pipeline and then round-trip
    # validated at its boundary. Disabling per-assignment validation avoids
    # rejecting a safe transient state while priority and referral flags are
    # updated together.
    model_config = ConfigDict(extra="forbid", validate_assignment=False)

    report_id: str = Field(..., min_length=1)
    cancer_type: CancerType | None = None
    method: Literal["baseline", "evidence_first", "ml"]
    model_version: str = Field(default="unversioned", min_length=1)
    source_report_digest: str | None = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    document_provenance: DocumentProvenance | None = None
    variables: list[VariableExtraction] = Field(..., min_length=len(CORE_VARIABLES), max_length=len(CORE_VARIABLES))
    qa_issues: list[QAIssue] = Field(default_factory=list)
    manual_review_required: bool = False
    review_priority: ReviewPriority = "low"
    review_priority_reason: str = Field(default="No documentation issues detected.", min_length=1)
    timestamp: datetime | None = None

    @model_validator(mode="after")
    def exactly_one_of_each_core_variable(self) -> "ExtractionResult":
        names = [variable.variable_name for variable in self.variables]
        if len(set(names)) != len(names) or set(names) != set(CORE_VARIABLES):
            raise ValueError("variables must contain exactly one result for each core variable")
        has_review_signal = bool(self.qa_issues) or any(
            variable.documentation_status != DocumentationStatus.SUPPORTED
            for variable in self.variables
        )
        if has_review_signal and not self.manual_review_required:
            raise ValueError("non-routine field statuses or QA issues require manual review")
        if self.review_priority != "low" and not self.manual_review_required:
            raise ValueError("non-low priority requires manual_review_required")
        if self.manual_review_required and self.review_priority == "low":
            raise ValueError("manual_review_required requires a non-low review priority")
        return self


class GoldAnnotation(StrictModel):
    report_id: str = Field(..., min_length=1)
    cancer_type: CancerType | None = None
    variable_name: CoreVariableName
    gold_value: str | None = None
    evidence_text: str | None = None
    evidence_start_offset: int | None = Field(default=None, ge=0)
    evidence_end_offset: int | None = Field(default=None, gt=0)
    documentation_status: DocumentationStatus
    manual_review_required: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def complete_evidence_coordinates(self) -> "GoldAnnotation":
        value = self.gold_value
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("gold_value cannot be blank")
            object.__setattr__(self, "gold_value", value)

        supplied = (
            self.evidence_text is not None,
            self.evidence_start_offset is not None,
            self.evidence_end_offset is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError("gold evidence text and both offsets must be supplied together")
        if all(supplied):
            assert self.evidence_text is not None
            assert self.evidence_start_offset is not None
            assert self.evidence_end_offset is not None
            if self.evidence_end_offset <= self.evidence_start_offset:
                raise ValueError("gold evidence end offset must be greater than start offset")
            if self.evidence_end_offset - self.evidence_start_offset != len(self.evidence_text):
                raise ValueError("gold evidence offset span must equal evidence text length")

        has_evidence = all(supplied)
        status = self.documentation_status
        if value is not None and not has_evidence:
            raise ValueError("every non-null gold_value requires exact evidence")
        if status == DocumentationStatus.SUPPORTED and value is None:
            raise ValueError("supported gold status requires a non-null gold_value")
        if status in {
            DocumentationStatus.NOT_DOCUMENTED,
            DocumentationStatus.NEGATED,
            DocumentationStatus.CONFLICTING,
            DocumentationStatus.UNSUPPORTED,
            DocumentationStatus.MANUAL_REVIEW_REQUIRED,
        } and value is not None:
            raise ValueError(f"{status.value} gold status requires abstention")
        if status in {
            DocumentationStatus.SUPPORTED,
            DocumentationStatus.CANNOT_BE_ASSIGNED,
            DocumentationStatus.NEGATED,
            DocumentationStatus.UNCERTAIN,
            DocumentationStatus.CONFLICTING,
            DocumentationStatus.SUPERSEDED,
        } and not has_evidence:
            raise ValueError(f"{status.value} gold status requires exact evidence")
        if status in {DocumentationStatus.UNCERTAIN, DocumentationStatus.SUPERSEDED} and value is None:
            raise ValueError(f"{status.value} gold status requires a candidate value")
        if status != DocumentationStatus.SUPPORTED and not self.manual_review_required:
            raise ValueError("non-supported gold statuses require manual_review_required=true")

        if self.variable_name == "tumor_size" and self.gold_value is not None:
            if not re.fullmatch(r"\d+(?:\.\d+)? cm", self.gold_value):
                raise ValueError("gold tumor size must use canonical '<number> cm' notation")
        if self.variable_name == "pathological_t_category" and value is not None:
            if value not in LUNG_PT_CATEGORIES:
                raise ValueError("gold pathological T category must use canonical lung pT notation")
            if status == DocumentationStatus.CANNOT_BE_ASSIGNED and value != "pTX":
                raise ValueError("only explicit pTX may carry a cannot-be-assigned gold value")
            if value == "pTX" and status != DocumentationStatus.CANNOT_BE_ASSIGNED:
                raise ValueError("explicit gold pTX must use cannot_be_assigned status")
        if self.variable_name == "pathological_n_category" and value is not None:
            if value not in LUNG_PN_CATEGORIES:
                raise ValueError("gold pathological N category must use canonical lung pN notation")
            if status == DocumentationStatus.CANNOT_BE_ASSIGNED and value != "pNX":
                raise ValueError("only explicit pNX may carry a cannot-be-assigned gold value")
            if value == "pNX" and status != DocumentationStatus.CANNOT_BE_ASSIGNED:
                raise ValueError("explicit gold pNX must use cannot_be_assigned status")
        if self.variable_name not in {
            "pathological_t_category",
            "pathological_n_category",
        } and status == DocumentationStatus.CANNOT_BE_ASSIGNED and value is not None:
            raise ValueError("cannot-be-assigned gold values are limited to explicit pTX or pNX")
        if value is not None and self.evidence_text is not None:
            states_value = (
                _evidence_text_lexically_states_value(
                    self.variable_name, value, self.evidence_text
                )
                if status == DocumentationStatus.SUPERSEDED
                else evidence_text_supports_value(
                    self.variable_name, value, self.evidence_text
                )
            )
            if not states_value:
                raise ValueError("gold evidence does not state the gold value")

        semantic_statuses = {
            DocumentationStatus.SUPPORTED,
            DocumentationStatus.NOT_DOCUMENTED,
            DocumentationStatus.CANNOT_BE_ASSIGNED,
            DocumentationStatus.NEGATED,
            DocumentationStatus.UNCERTAIN,
            DocumentationStatus.CONFLICTING,
            DocumentationStatus.SUPERSEDED,
            DocumentationStatus.UNSUPPORTED,
            DocumentationStatus.MANUAL_REVIEW_REQUIRED,
        }
        if status in semantic_statuses and not evidence_text_supports_status(
            self.variable_name,
            status,
            [self.evidence_text] if self.evidence_text is not None else [],
            value=value,
        ):
            raise ValueError(
                f"gold evidence does not support {status.value} documentation status"
            )
        return self


class VariableMetrics(StrictModel):
    variable_name: CoreVariableName
    precision: float | None = Field(None, ge=0.0, le=1.0)
    recall: float | None = Field(None, ge=0.0, le=1.0)
    f1_score: float | None = Field(None, ge=0.0, le=1.0)
    true_positives: int = Field(0, ge=0)
    false_positives: int = Field(0, ge=0)
    false_negatives: int = Field(0, ge=0)
    true_negatives: int = Field(0, ge=0)
    unsupported_count: int = Field(0, ge=0)
    abstained_count: int = Field(0, ge=0)
    evidence_correct_count: int = Field(0, ge=0)
    evidence_total_count: int = Field(0, ge=0)
    status_correct_count: int = Field(0, ge=0)
    status_total_count: int = Field(0, ge=0)


class OverallMetrics(StrictModel):
    method: Literal["baseline", "evidence_first", "ml"]
    report_count: int = Field(0, ge=0)
    fields_evaluated: int = Field(0, ge=0)
    returned_value_count: int = Field(0, ge=0)
    gold_value_count: int = Field(0, ge=0)
    expected_abstention_count: int = Field(0, ge=0)
    appropriate_abstention_count: int = Field(0, ge=0)
    undocumented_gold_field_count: int = Field(0, ge=0)
    undocumented_false_positive_count: int = Field(0, ge=0)
    manual_review_count: int = Field(0, ge=0)
    schema_valid_count: int = Field(0, ge=0)
    unsupported_extraction_rate: float | None = Field(None, ge=0.0, le=1.0)
    overall_precision: float | None = Field(None, ge=0.0, le=1.0)
    overall_recall: float | None = Field(None, ge=0.0, le=1.0)
    overall_f1: float | None = Field(None, ge=0.0, le=1.0)
    appropriate_abstention_rate: float | None = Field(None, ge=0.0, le=1.0)
    false_positive_rate_undocumented: float | None = Field(None, ge=0.0, le=1.0)
    coverage: float = Field(0.0, ge=0.0, le=1.0)
    manual_review_referral_rate: float = Field(0.0, ge=0.0, le=1.0)
    schema_validity_rate: float = Field(0.0, ge=0.0, le=1.0)
    documentation_status_accuracy: float = Field(0.0, ge=0.0, le=1.0)
    documentation_status_correct_count: int = Field(0, ge=0)
    exact_evidence_rate: float | None = Field(None, ge=0.0, le=1.0)
    exact_evidence_correct_count: int = Field(0, ge=0)
    exact_evidence_total_count: int = Field(0, ge=0)
    gold_manual_review_report_count: int = Field(0, ge=0)
    review_referral_true_positive_count: int = Field(0, ge=0)
    review_referral_false_positive_count: int = Field(0, ge=0)
    review_referral_false_negative_count: int = Field(0, ge=0)
    manual_review_referral_precision: float | None = Field(None, ge=0.0, le=1.0)
    manual_review_referral_recall: float | None = Field(None, ge=0.0, le=1.0)
    variable_metrics: list[VariableMetrics] = Field(default_factory=list)


class ComparisonResult(StrictModel):
    baseline_metrics: OverallMetrics
    evidence_first_metrics: OverallMetrics
    unsupported_rate_reduction: float | None = None
    precision_improvement: float | None = None
    recall_change: float | None = None
    coverage_change: float = 0.0
