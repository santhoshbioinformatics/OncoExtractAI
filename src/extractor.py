"""Deterministic, evidence-anchored lung pathology abstraction."""

from __future__ import annotations

import re
from typing import NamedTuple

from .schemas import (
    CORE_VARIABLES,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    LUNG_PN_SUFFIX_PATTERN,
    LUNG_PT_SUFFIX_PATTERN,
    QAIssue,
    QAIssueType,
    VariableExtraction,
)
from .utils import get_iso_timestamp


Match = tuple[str, str, int, int]


class Section(NamedTuple):
    kind: str
    start: int
    end: int


PATTERNS: dict[str, list[str]] = {
    "tumor_size": [
        r"tumou?r\s+size[^\n]{0,80}?greatest\s+dimension(?:\s+of\s+(?:the\s+)?tumou?r)?\s*"
        r"(?:is|:|=)?\s*(\d+(?:\.\d+)?)\s*cm\b",
        r"greatest\s+dimension(?:\s+of\s+(?:the\s+)?tumou?r)?\s*(?:is|:|=)?\s*"
        r"(\d+(?:\.\d+)?)\s*cm\b",
        r"greatest\s+diameter(?:\s+of\s+(?:the\s+)?tumou?r)?\s*(?:is|:|=)?\s*"
        r"(\d+(?:\.\d+)?)\s*cm\b",
        r"(?:size|dimension)\s+of\s+(?:invasive\s+)?(?:carcinoma|tumou?r)\s*"
        r"(?:is|:|=)?\s*(\d+(?:\.\d+)?)\s*cm\b",
        r"tumou?r\s+size\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)\s*[x×]\s*"
        r"\d+(?:\.\d+)?(?:\s*[x×]\s*\d+(?:\.\d+)?)?\s*cm\b",
        r"\b(?:tumou?r|mass|lesion|carcinoma)\s+(?:measuring|measures?|measured)\s+"
        r"(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*cm\b",
        r"(\d+(?:\.\d+)?)\s*cm\s*(?:in\s*)?(?:greatest\s*)?dimension\b",
        r"tumou?r\s+sizes?\s*(?:is|:)?\s*(?:approximately\s+|about\s+)?(\d+(?:\.\d+)?)\s*cm\b",
        r"maximum\s+tumou?r\s+dimension\s*(?:is|:|measured\s+)?\s*(\d+(?:\.\d+)?)\s*cm\b",
        r"(?:gross|microscopic|synoptic|narrative)\s+(?:section\s+)?records\s+(\d+(?:\.\d+)?)\s*cm\b",
        r"(\d+(?:\.\d+)?)\s*cm\s*(?:tumou?r|mass|lesion)\b",
    ],
    "pathological_t_category": [
        rf"\b(pT{LUNG_PT_SUFFIX_PATTERN})\b",
        rf"pathological\s+T\s+category\s*(?:is|:)?\s*(pT{LUNG_PT_SUFFIX_PATTERN})\b",
        rf"consistent\s+with\s+(?:pathological\s+T\s+category\s+)?(pT{LUNG_PT_SUFFIX_PATTERN})\b",
    ],
    "pathological_n_category": [
        rf"\b(pN{LUNG_PN_SUFFIX_PATTERN})\b",
        rf"pathological\s+N\s+category\s*(?:is|:)?\s*(pN{LUNG_PN_SUFFIX_PATTERN})\b",
        rf"consistent\s+with\s+(?:pathological\s+N\s+category\s+)?(pN{LUNG_PN_SUFFIX_PATTERN})\b",
    ],
    "histologic_diagnosis": [
        r"\b(minimally\s+invasive\s+adenocarcinoma)\b",
        r"\b(basaloid\s+squamous\s+cell\s+carcinoma)\b",
        r"\b(adenocarcinoma\s*,?\s*(?:acinar|papillary|micropapillary|lepidic|solid)"
        r"(?:\s+predominant)?(?:\s+pattern)?)\b",
        r"\b(squamous\s+cell\s+carcinoma\s*,?\s*(?:keratinizing|non[-\s]?keratinizing))\b",
        r"\b((?:well|moderately|poorly)[-\s]+differentiated\s+(?:invasive\s+)?squamous\s+cell\s+carcinoma)\b",
        r"\b((?:invasive\s+)?squamous\s+cell\s+carcinoma)\b",
        r"\b((?:invasive\s+)?adenocarcinoma(?:\s+with\s+(?:predominant\s+)?(?:lepidic|acinar|papillary|micropapillary|solid)(?:\s+and\s+(?:lepidic|acinar|papillary|micropapillary|solid))*\s+growth\s+pattern)?)\b",
        r"\b((?:invasive\s+)?adenocarcinoma(?:,\s*(?:lepidic|acinar|papillary|micropapillary|solid)(?:\s+and\s+(?:lepidic|acinar|papillary|micropapillary|solid))*\s+growth\s+pattern)?)\b",
        r"\b(poorly\s+differentiated\s+carcinoma)\b",
        r"\b((?:invasive\s+)?large\s+cell\s+carcinoma)\b",
    ],
}

UNCERTAINTY_PATTERNS = [
    r"\buncertain(?:ty)?\b",
    r"\bapproximately\b",
    r"\bapprox\.?\b",
    r"\bestimated?\b",
    r"\blikely\b",
    r"\bpossible\b",
    r"\bprobable\b",
    r"\bsuspected\b",
    r"\bequivocal\b",
    r"\bsuggestive\b",
    r"\bdiscrepancy\b",
    r"\bconflicting\b",
    r"\branging\b",
    r"\bpending\b",
    r"\bprovisional\b",
]

EXPLICIT_UPDATE_PATTERN = re.compile(
    r"\b(?:amended?|amendment|revised?|revision|reclassified?|reclassification|"
    r"corrected?|correction|supersed(?:e|ed|es)|updated?|changed\s+from)\b",
    re.IGNORECASE,
)

SECTION_HEADING_PATTERN = re.compile(
    r"^[ \t]*(?:(?P<revised>REVISED\s+FINAL\s+DIAGNOSIS)"
    r"|(?P<preliminary>PRELIMINARY\s+DIAGNOSIS)"
    r"|(?P<amended>AMENDED(?:\s+FINAL)?\s+DIAGNOSIS)"
    r"|(?P<final>FINAL\s+DIAGNOSIS)"
    r"|(?P<addendum>ADDENDUM[^\n:]*)"
    r"|(?P<staging>STAGING))\s*:?[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)

PRIORITY_ORDER = ("low", "medium", "high", "critical")


def _raise_priority(current: str, new: str) -> str:
    return new if PRIORITY_ORDER.index(new) > PRIORITY_ORDER.index(current) else current


def _find_all_matches(text: str, patterns: list[str]) -> list[Match]:
    """Return unique ``(exact text, captured value, start, end)`` matches."""

    found: list[Match] = []
    seen: set[tuple[int, int, str]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
            value = match.group(1).strip() if match.lastindex and match.group(1) else match.group(0).strip()
            key = (match.start(), match.end(), value.casefold())
            if not value or key in seen:
                continue
            seen.add(key)
            found.append((match.group(0), value, match.start(), match.end()))
    return sorted(found, key=lambda item: (item[2], -(item[3] - item[2])))


def _normalize_value(variable_name: str, value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if variable_name == "tumor_size":
        match = re.search(r"(\d+(?:\.\d+)?)\s*cm", value, re.IGNORECASE)
        if match:
            return f"{match.group(1)} cm"
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            return f"{value} cm"
        return value
    if variable_name == "pathological_t_category":
        match = re.search(rf"pT({LUNG_PT_SUFFIX_PATTERN})\b", value, re.IGNORECASE)
        if match:
            suffix = match.group(1)
            if suffix.casefold() in {"is", "1mi"}:
                suffix = suffix.casefold()
            elif suffix.casefold() == "x":
                suffix = "X"
            else:
                suffix = suffix[0] + suffix[1:].casefold()
            return f"pT{suffix}"
    if variable_name == "pathological_n_category":
        match = re.search(rf"pN({LUNG_PN_SUFFIX_PATTERN})\b", value, re.IGNORECASE)
        if match:
            suffix = match.group(1)
            suffix = "X" if suffix.casefold() == "x" else suffix[0] + suffix[1:].casefold()
            return f"pN{suffix}"
    if variable_name == "histologic_diagnosis" and value:
        if value.isupper():
            value = value.lower()
        return value[0].upper() + value[1:]
    return value


def _conflict_key(variable_name: str, value: str) -> str:
    normalized = _normalize_value(variable_name, value).casefold()
    if variable_name != "histologic_diagnosis":
        return normalized
    if "minimally invasive adenocarcinoma" in normalized:
        return "minimally invasive adenocarcinoma"
    if "squamous cell carcinoma" in normalized:
        return "squamous cell carcinoma"
    if "adenocarcinoma" in normalized:
        return "adenocarcinoma"
    if "large cell carcinoma" in normalized:
        return "large cell carcinoma"
    if "poorly differentiated carcinoma" in normalized:
        return "poorly differentiated carcinoma"
    return normalized


def _sections(report_text: str) -> list[Section]:
    headings = list(SECTION_HEADING_PATTERN.finditer(report_text))
    sections: list[Section] = []
    for index, heading in enumerate(headings):
        kind = next(name for name, value in heading.groupdict().items() if value is not None)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(report_text)
        sections.append(Section(kind, heading.start(), end))
    return sections


def _section_for(offset: int, sections: list[Section]) -> Section | None:
    return next((section for section in sections if section.start <= offset < section.end), None)


def _is_historical_mention(report_text: str, start: int, end: int) -> bool:
    prefix = report_text[max(0, start - 60):start]
    suffix = report_text[end:min(len(report_text), end + 60)]
    old_prefix = re.search(
        r"(?:amended|revised|changed|corrected|updated)\s+from\s*[(:]?\s*$|"
        r"\b(?:previously|formerly|originally|prior)\s*[(:]?\s*$",
        prefix,
        re.IGNORECASE,
    )
    old_suffix = re.match(
        r"\s*(?:was\s+)?(?:amended|revised|changed|corrected|updated)\s+to\b",
        suffix,
        re.IGNORECASE,
    )
    return bool(old_prefix or old_suffix)


def _resolve_current_matches(report_text: str, matches: list[Match]) -> tuple[list[Match], list[Match]]:
    """Return current and explicitly superseded matches without guessing chronology."""

    if not matches:
        return [], []

    inline_historical = [match for match in matches if _is_historical_mention(report_text, match[2], match[3])]
    eligible = [match for match in matches if match not in inline_historical]
    sections = _sections(report_text)

    candidate_sections: list[Section] = []
    for section in sections:
        section_matches = [match for match in eligible if _section_for(match[2], sections) == section]
        if not section_matches:
            continue
        if section.kind in {"revised", "amended"}:
            candidate_sections.append(section)
        elif section.kind == "addendum" and EXPLICIT_UPDATE_PATTERN.search(
            report_text[section.start:section.end]
        ):
            candidate_sections.append(section)
        elif section.kind == "final":
            has_earlier_preliminary = any(
                prior.kind == "preliminary" and prior.start < section.start for prior in sections
            )
            if has_earlier_preliminary:
                candidate_sections.append(section)

    if not candidate_sections:
        return eligible, inline_historical

    current_section = max(candidate_sections, key=lambda section: section.start)
    current = [
        match
        for match in eligible
        if match[2] >= current_section.start
        and (_section_for(match[2], sections) or current_section).kind != "preliminary"
    ]
    if not current:
        return eligible, inline_historical
    superseded = [match for match in eligible if match not in current] + inline_historical
    return current, superseded


def _choose_best_match(matches: list[Match], variable_name: str) -> Match | None:
    if not matches:
        return None

    def score(item: Match) -> tuple[int, int, int]:
        full, _, start, _ = item
        explicit = int(
            "pathological" in full.casefold()
            or "final diagnosis" in full.casefold()
            or variable_name == "histologic_diagnosis"
        )
        return explicit, len(full), start

    return max(matches, key=score)


def _make_evidence(text: str, start: int, end: int) -> EvidenceSpan | None:
    if start < 0 or end <= start or end > len(text):
        return None
    snippet = text[start:end]
    if not snippet:
        return None
    return EvidenceSpan(text=snippet, start_offset=start, end_offset=end)


def _local_context(text: str, start: int, end: int, window: int = 100) -> str:
    left_bound = max(text.rfind("\n", 0, start), text.rfind(".", 0, start), start - window)
    right_candidates = [position for position in (text.find("\n", end), text.find(".", end)) if position >= 0]
    right_bound = min(right_candidates) + 1 if right_candidates else min(len(text), end + window)
    return text[max(0, left_bound):min(len(text), right_bound)]


def _has_uncertainty(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in UNCERTAINTY_PATTERNS)


def _negation_evidence(
    variable_name: str,
    value: str,
    report_text: str,
    start: int,
    end: int,
) -> EvidenceSpan | None:
    escaped = re.escape(value)
    patterns = [
        rf"\bno\s+evidence\s+of\s+{escaped}\b",
        rf"\bnegative\s+for\s+{escaped}\b",
        rf"\babsence\s+of\s+{escaped}\b",
        rf"\bwithout\s+{escaped}\b",
        rf"\bnot\s+{escaped}\b",
        rf"\b{escaped}\s+(?:is|was|are|were)\s+not\s+(?:identified|present|found|detected)\b",
        rf"\brule[sd]?\s+out\s+{escaped}\b",
    ]
    if variable_name == "histologic_diagnosis":
        patterns.append(rf"\bconcern\s+for\s+{escaped}.{{0,60}}?ruled\s+out\b")

    context_start = max(0, start - 100)
    context_end = min(len(report_text), end + 100)
    context = report_text[context_start:context_end]
    for pattern in patterns:
        match = re.search(pattern, context, re.IGNORECASE)
        if match:
            return _make_evidence(
                report_text,
                context_start + match.start(),
                context_start + match.end(),
            )
    return None


def _first_status_evidence(
    report_text: str,
    patterns: list[str],
) -> EvidenceSpan | None:
    for pattern in patterns:
        match = re.search(pattern, report_text, re.IGNORECASE | re.MULTILINE)
        if match:
            return _make_evidence(report_text, match.start(), match.end())
    return None


def _documented_no_value(
    variable_name: str,
    report_text: str,
) -> tuple[DocumentationStatus, EvidenceSpan | None, str]:
    """Classify explicit no-value language without manufacturing a category."""

    labels = {
        "histologic_diagnosis": r"(?:histologic\s+diagnosis|histology)",
        "tumor_size": r"tumou?r\s+size",
        "pathological_t_category": r"(?:pathological\s+T\s+category|pT)",
        "pathological_n_category": r"(?:pathological\s+N\s+category|pN)",
    }
    label = labels[variable_name]

    not_documented = _first_status_evidence(
        report_text,
        [rf"\b{label}\s*(?::|is)?\s*not\s+documented\b[^\n.]*\.?"],
    )
    if not_documented:
        return (
            DocumentationStatus.NOT_DOCUMENTED,
            not_documented,
            f"{variable_name.replace('_', ' ').title()} is explicitly recorded as not documented.",
        )

    cannot_assign = _first_status_evidence(
        report_text,
        [
            rf"\b{label}\s*(?::|is)?\s*cannot\s+be\s+assigned\b[^\n.]*\.?",
            rf"\b{label}\s*(?::|is)?\s*cannot\s+be\s+assessed\b[^\n.]*\.?",
        ],
    )
    if cannot_assign:
        return (
            DocumentationStatus.CANNOT_BE_ASSIGNED,
            cannot_assign,
            f"{variable_name.replace('_', ' ').title()} is explicitly documented as not assignable.",
        )

    if variable_name == "tumor_size":
        no_tumor = _first_status_evidence(
            report_text,
            [
                r"\bNo\s+invasive\s+tumou?r\s+is\s+identified\s+for\s+measurement\b[^\n.]*\.?",
                r"\bNo\s+tumou?r\s+is\s+identified\s+for\s+measurement\b[^\n.]*\.?",
            ],
        )
        if no_tumor:
            return (
                DocumentationStatus.NEGATED,
                no_tumor,
                "No invasive tumor is identified for measurement.",
            )

    if variable_name == "pathological_n_category":
        no_nodes = _first_status_evidence(
            report_text,
            [
                r"\bno\s+(?:regional\s+)?lymph\s+nodes?\s+(?:were\s+)?(?:submitted|identified|assessed|evaluated)\b[^\n.]*\.?",
                r"\bregional\s+lymph\s+nodes?\s+cannot\s+be\s+assessed\b[^\n.]*\.?",
                r"\bnodes?\s+cannot\s+be\s+assessed\b[^\n.]*\.?",
            ],
        )
        if no_nodes:
            return (
                DocumentationStatus.CANNOT_BE_ASSIGNED,
                no_nodes,
                "Regional lymph nodes were not assessed; no pN category was inferred.",
            )

    return (
        DocumentationStatus.NOT_DOCUMENTED,
        None,
        f"{variable_name.replace('_', ' ').title()} is not explicitly documented.",
    )


def _issue_priority(issues: list[QAIssue]) -> tuple[str, str]:
    if not issues:
        return "low", "No documentation issues detected."
    if any(issue.severity == "critical" for issue in issues):
        return "critical", "Critical evidence validation failure requires manual review."
    conflicts = [issue for issue in issues if issue.issue_type == QAIssueType.CONFLICTING]
    if conflicts:
        fields = ", ".join(sorted({issue.variable_name or "report" for issue in conflicts}))
        return "high", f"Conflicting documentation requires review: {fields}."
    superseded = [issue for issue in issues if issue.issue_type == QAIssueType.SUPERSEDED]
    if superseded:
        return "high", "An explicit amendment or final revision supersedes older documentation."
    high = [issue for issue in issues if issue.severity == "high"]
    if high:
        fields = ", ".join(sorted({issue.variable_name or "report" for issue in high}))
        return "high", f"High-impact documentation issue requires review: {fields}."
    uncertain = [issue for issue in issues if issue.issue_type == QAIssueType.UNCERTAIN]
    if uncertain:
        fields = ", ".join(sorted({issue.variable_name or "report" for issue in uncertain}))
        return "medium", f"Uncertain documentation requires review: {fields}."
    fields = ", ".join(sorted({issue.variable_name or "report" for issue in issues}))
    return "medium", f"Documentation is incomplete or requires review: {fields}."


def _distinct_match_groups(variable_name: str, matches: list[Match]) -> dict[str, list[Match]]:
    groups: dict[str, list[Match]] = {}
    for match in matches:
        key = _conflict_key(variable_name, match[1])
        groups.setdefault(key, []).append(match)
    return groups


def _extract(report_text: str, report_id: str, method: str) -> ExtractionResult:
    if not isinstance(report_text, str):
        raise TypeError("report_text must be a string")

    variables: list[VariableExtraction] = []
    issues: list[QAIssue] = []

    for variable_name in CORE_VARIABLES:
        all_matches = _find_all_matches(report_text, PATTERNS[variable_name])
        if method == "evidence_first":
            active_matches, superseded_matches = _resolve_current_matches(
                report_text, all_matches
            )
        else:
            # The safe baseline remains chronology-naive. Contradictions therefore
            # abstain rather than being silently resolved as an amendment.
            active_matches = all_matches
            superseded_matches = []

        if not active_matches:
            status, status_evidence, note = _documented_no_value(
                variable_name, report_text
            )
            evidence_spans = [status_evidence] if status_evidence else None
            variables.append(
                VariableExtraction(
                    variable_name=variable_name,
                    extracted_value=None,
                    documentation_status=status,
                    evidence=evidence_spans,
                    notes=note,
                )
            )
            severity = (
                "high"
                if variable_name
                in {"histologic_diagnosis", "pathological_n_category"}
                else "medium"
            )
            issue_type = (
                QAIssueType.NEGATED
                if status == DocumentationStatus.NEGATED
                else QAIssueType.MISSING
            )
            issues.append(
                QAIssue(
                    issue_type=issue_type,
                    variable_name=variable_name,
                    description=note,
                    severity=severity,
                    evidence=evidence_spans,
                    suggestion="Confirm the documentation manually; do not infer a category.",
                )
            )
            continue

        groups = _distinct_match_groups(variable_name, active_matches)
        if len(groups) > 1:
            alternatives: list[tuple[str, Match]] = []
            for group_matches in groups.values():
                best_alternative = (
                    group_matches[0]
                    if method == "baseline"
                    else _choose_best_match(group_matches, variable_name)
                )
                if best_alternative:
                    alternatives.append(
                        (_normalize_value(variable_name, best_alternative[1]), best_alternative)
                    )
            alternatives.sort(key=lambda item: item[1][2])
            evidence = [
                span
                for _, match in alternatives
                if (span := _make_evidence(report_text, match[2], match[3])) is not None
            ]
            values = [value for value, _ in alternatives]
            variables.append(
                VariableExtraction(
                    variable_name=variable_name,
                    extracted_value=None,
                    documentation_status=DocumentationStatus.CONFLICTING,
                    evidence=evidence,
                    alternatives=values,
                    notes=f"Conflicting explicit values: {', '.join(values)}",
                )
            )
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.CONFLICTING,
                    variable_name=variable_name,
                    description=f"Conflicting explicit values detected: {', '.join(values)}.",
                    severity="high",
                    evidence=evidence,
                    suggestion="Resolve the alternatives manually; no value was selected.",
                )
            )
            continue

        best = _choose_best_match(active_matches, variable_name)
        assert best is not None
        _, raw_value, start, end = best
        value = _normalize_value(variable_name, raw_value)
        evidence = _make_evidence(report_text, start, end)
        if evidence is None:
            variables.append(
                VariableExtraction(
                    variable_name=variable_name,
                    extracted_value=None,
                    documentation_status=DocumentationStatus.UNSUPPORTED,
                    evidence=None,
                    notes="Candidate could not be anchored to the report.",
                )
            )
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                    variable_name=variable_name,
                    description="Candidate could not be anchored to exact report evidence.",
                    severity="critical",
                )
            )
            continue

        negation = _negation_evidence(variable_name, value, report_text, start, end)
        context = _local_context(report_text, start, end)
        uncertain = _has_uncertainty(context)

        if negation is not None:
            status = DocumentationStatus.NEGATED
            selected_value = None
            selected_evidence = [negation]
            note = "The candidate is explicitly negated in the report."
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.NEGATED,
                    variable_name=variable_name,
                    description=note,
                    severity="high" if variable_name == "histologic_diagnosis" else "medium",
                    evidence=selected_evidence,
                    suggestion="Confirm the negated finding during review.",
                )
            )
        elif (
            variable_name == "pathological_n_category" and value == "pNX"
        ) or (
            variable_name == "pathological_t_category" and value == "pTX"
        ):
            status = DocumentationStatus.CANNOT_BE_ASSIGNED
            selected_value = value
            selected_evidence = [evidence]
            note = (
                "Explicit pNX documented; it is distinct from pN0 and from an undocumented pN."
                if value == "pNX"
                else "Explicit pTX documented; no pathological T category was inferred."
            )
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.MANUAL_REVIEW,
                    variable_name=variable_name,
                    description=note,
                    severity="medium",
                    evidence=selected_evidence,
                    suggestion=(
                        "Retain pNX as explicitly documented; do not convert it to pN0."
                        if value == "pNX"
                        else "Retain pTX as explicitly documented; do not infer pT from size."
                    ),
                )
            )
        elif uncertain:
            status = DocumentationStatus.UNCERTAIN
            selected_value = value
            selected_evidence = [evidence]
            note = "Uncertainty language occurs in the same local statement."
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.UNCERTAIN,
                    variable_name=variable_name,
                    description=f"Uncertainty language qualifies the proposed {variable_name.replace('_', ' ')}.",
                    severity="medium",
                    evidence=selected_evidence,
                    suggestion="Review the candidate before acceptance.",
                )
            )
        else:
            status = DocumentationStatus.SUPPORTED
            selected_value = value
            selected_evidence = [evidence]
            note = "Candidate is anchored to exact report evidence."

        variables.append(
            VariableExtraction(
                variable_name=variable_name,
                extracted_value=selected_value,
                documentation_status=status,
                evidence=selected_evidence,
                notes=note,
            )
        )

        current_key = _conflict_key(variable_name, value)
        prior_alternatives = [
            match
            for match in superseded_matches
            if _conflict_key(variable_name, match[1]) != current_key
        ]
        if prior_alternatives:
            prior = _choose_best_match(prior_alternatives, variable_name)
            prior_evidence = _make_evidence(report_text, prior[2], prior[3]) if prior else None
            issue_evidence = [span for span in (prior_evidence, evidence) if span is not None]
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.SUPERSEDED,
                    variable_name=variable_name,
                    description=(
                        f"An explicit final/amended relationship replaces "
                        f"{_normalize_value(variable_name, prior[1]) if prior else 'an older value'} with {value}."
                    ),
                    severity="high",
                    evidence=issue_evidence,
                    suggestion="Use the current value and retain the older span in the audit trail.",
                )
            )

    priority, reason = _issue_priority(issues)
    return ExtractionResult(
        report_id=report_id,
        method=method,
        variables=variables,
        qa_issues=issues,
        manual_review_required=priority != "low",
        review_priority=priority,
        review_priority_reason=reason,
        timestamp=get_iso_timestamp(),
    )


def baseline_extract(report_text: str, report_id: str) -> ExtractionResult:
    """Run deterministic direct extraction without inferring pT or pN."""

    return _extract(report_text, report_id, "baseline")


def evidence_first_extract(report_text: str, report_id: str) -> ExtractionResult:
    """Run evidence-first extraction with abstention and documentation QA."""

    return _extract(report_text, report_id, "evidence_first")
