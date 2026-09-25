"""Human-review state, validation, saved snapshots, and audit records."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    CORE_VARIABLES,
    CoreVariableName,
    DocumentationStatus,
    DocumentProvenance,
    EvidenceSpan,
    ExtractionResult,
    ReviewAction,
    ReviewPriority,
    VariableExtraction,
    evidence_text_supports_status,
    evidence_text_supports_value,
    local_evidence_context,
)


def _validate_conflict_alternative(
    variable_name: CoreVariableName,
    alternative: str,
    evidence: EvidenceSpan,
    context_text: str | None = None,
) -> None:
    if not evidence_text_supports_value(
        variable_name, alternative, evidence.text, context_text
    ):
        raise ValueError("conflict evidence does not state its paired alternative")
    alternative_status = (
        DocumentationStatus.CANNOT_BE_ASSIGNED
        if alternative in {"pTX", "pNX"}
        else DocumentationStatus.SUPPORTED
    )
    VariableExtraction(
        variable_name=variable_name,
        extracted_value=alternative,
        documentation_status=alternative_status,
        evidence=[evidence],
    )


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable_name: CoreVariableName
    action: ReviewAction = ReviewAction.PENDING
    corrected_value: str | None = None
    corrected_status: DocumentationStatus | None = None
    conflicting_alternatives: list[str] = Field(default_factory=list, max_length=8)
    corrected_evidence_texts: list[str] = Field(default_factory=list, max_length=8)
    reason: str = Field(default="", max_length=2_000)


class ReviewedField(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable_name: CoreVariableName
    original_output: VariableExtraction
    reviewed_value: str | None = None
    reviewed_status: DocumentationStatus
    review_action: ReviewAction
    review_reason: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reviewed_alternatives: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def coherent_reviewed_field(self) -> "ReviewedField":
        if self.original_output.variable_name != self.variable_name:
            raise ValueError("original output variable must match reviewed field")
        if self.review_action == ReviewAction.PENDING:
            raise ValueError("saved reviewed fields cannot remain pending")
        if not self.review_reason.strip():
            raise ValueError("saved reviewed fields require a review reason")

        VariableExtraction(
            variable_name=self.variable_name,
            extracted_value=self.reviewed_value,
            documentation_status=self.reviewed_status,
            evidence=self.evidence or None,
            alternatives=self.reviewed_alternatives,
        )
        if self.reviewed_status == DocumentationStatus.CONFLICTING:
            if len(self.reviewed_alternatives) < 2 or len(self.reviewed_alternatives) != len(self.evidence):
                raise ValueError("reviewed conflict requires an alternative value per evidence span")
            if len({item.casefold().strip() for item in self.reviewed_alternatives}) != len(self.reviewed_alternatives):
                raise ValueError("reviewed conflict alternatives must be distinct")
            for alternative, evidence in zip(self.reviewed_alternatives, self.evidence):
                _validate_conflict_alternative(self.variable_name, alternative, evidence)
        elif self.reviewed_alternatives:
            raise ValueError("reviewed alternatives are only valid for a conflict")

        if self.review_action == ReviewAction.ACCEPTED:
            if (
                self.reviewed_value != self.original_output.extracted_value
                or self.reviewed_status != self.original_output.documentation_status
                or self.evidence != (self.original_output.evidence or [])
                or self.reviewed_alternatives != self.original_output.alternatives
            ):
                raise ValueError(
                    "accepted review must preserve original value, status, evidence, and alternatives"
                )
        elif self.review_action == ReviewAction.REJECTED:
            if (
                self.reviewed_value is not None
                or self.reviewed_status != DocumentationStatus.UNSUPPORTED
                or self.evidence
            ):
                raise ValueError("rejected review must withhold the value as unsupported")
        elif self.review_action == ReviewAction.FLAGGED:
            if (
                self.reviewed_value is not None
                or self.reviewed_status != DocumentationStatus.MANUAL_REVIEW_REQUIRED
                or self.evidence
            ):
                raise ValueError("flagged review must withhold the value for manual review")
        return self


class ReviewSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    report_id: str
    method: Literal["baseline", "evidence_first", "ml"]
    model_version: str = "unversioned"
    source_report_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    extraction_timestamp: str | None = None
    document_provenance: DocumentProvenance | None = None
    reviewer_identity: str = Field(default="Unassigned", min_length=1)
    reviewer_identity_verified: bool = False
    reviewed_at: str
    review_started_at: str | None = None
    review_duration_seconds: float | None = Field(default=None, ge=0)
    returned_for_clarification: bool = False
    original_review_priority: ReviewPriority
    original_review_priority_reason: str
    review_priority: ReviewPriority
    review_priority_reason: str
    overall_note: str = ""
    fields: list[ReviewedField] = Field(
        ..., min_length=len(CORE_VARIABLES), max_length=len(CORE_VARIABLES)
    )

    @model_validator(mode="after")
    def exactly_one_review_per_core_variable(self) -> "ReviewSnapshot":
        names = [field.variable_name for field in self.fields]
        if len(set(names)) != len(names) or set(names) != set(CORE_VARIABLES):
            raise ValueError("snapshot must contain exactly one reviewed core field")
        return self


class AuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    review_id: str
    report_id: str
    method: Literal["baseline", "evidence_first", "ml"]
    model_version: str = "unversioned"
    source_report_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    extraction_timestamp: str | None = None
    reviewer_identity: str = Field(default="Unassigned", min_length=1)
    reviewer_identity_verified: bool = False
    variable_name: CoreVariableName
    original_output: VariableExtraction
    reviewer_action: ReviewAction
    corrected_value: str | None = None
    resulting_value: str | None = None
    resulting_status: DocumentationStatus
    resulting_evidence: list[EvidenceSpan] = Field(default_factory=list)
    resulting_alternatives: list[str] = Field(default_factory=list, max_length=8)
    review_reason: str
    timestamp: str

    @model_validator(mode="after")
    def coherent_audit_event(self) -> "AuditRecord":
        if self.original_output.variable_name != self.variable_name:
            raise ValueError("audit original output must match variable_name")
        if self.reviewer_action == ReviewAction.PENDING:
            raise ValueError("saved audit actions cannot remain pending")
        if not self.review_reason.strip():
            raise ValueError("audit actions require a review reason")

        VariableExtraction(
            variable_name=self.variable_name,
            extracted_value=self.resulting_value,
            documentation_status=self.resulting_status,
            evidence=self.resulting_evidence or None,
            alternatives=self.resulting_alternatives,
        )
        if self.resulting_status == DocumentationStatus.CONFLICTING:
            if len(self.resulting_alternatives) < 2 or len(self.resulting_alternatives) != len(self.resulting_evidence):
                raise ValueError("conflict audit requires alternative values and evidence")
            if len({item.casefold().strip() for item in self.resulting_alternatives}) != len(self.resulting_alternatives):
                raise ValueError("audit conflict alternatives must be distinct")
            for alternative, evidence in zip(self.resulting_alternatives, self.resulting_evidence):
                _validate_conflict_alternative(self.variable_name, alternative, evidence)
        elif self.resulting_alternatives:
            raise ValueError("audit alternatives are only valid for a conflict")
        if self.reviewer_action == ReviewAction.CORRECTED:
            if self.corrected_value != self.resulting_value:
                raise ValueError("corrected audit value must equal resulting value")
        elif self.corrected_value is not None:
            raise ValueError("corrected_value is only valid for a correction action")
        if self.reviewer_action == ReviewAction.ACCEPTED and (
            self.resulting_value != self.original_output.extracted_value
            or self.resulting_status != self.original_output.documentation_status
            or self.resulting_evidence != (self.original_output.evidence or [])
            or self.resulting_alternatives != self.original_output.alternatives
        ):
            raise ValueError("accepted audit event must preserve the original output")
        if self.reviewer_action == ReviewAction.REJECTED and (
            self.resulting_value is not None
            or self.resulting_status != DocumentationStatus.UNSUPPORTED
            or self.resulting_evidence
        ):
            raise ValueError("rejected audit event must withhold the value as unsupported")
        if self.reviewer_action == ReviewAction.FLAGGED and (
            self.resulting_value is not None
            or self.resulting_status != DocumentationStatus.MANUAL_REVIEW_REQUIRED
            or self.resulting_evidence
        ):
            raise ValueError("flagged audit event must withhold the value for manual review")
        return self


def _reviewed_priority(fields: list[ReviewedField]) -> tuple[ReviewPriority, str]:
    high_statuses = {
        DocumentationStatus.CONFLICTING,
        DocumentationStatus.UNSUPPORTED,
        DocumentationStatus.MANUAL_REVIEW_REQUIRED,
    }
    medium_statuses = {
        DocumentationStatus.NOT_DOCUMENTED,
        DocumentationStatus.CANNOT_BE_ASSIGNED,
        DocumentationStatus.NEGATED,
        DocumentationStatus.UNCERTAIN,
        DocumentationStatus.SUPERSEDED,
    }
    high_fields = [
        field.variable_name for field in fields if field.reviewed_status in high_statuses
    ]
    if high_fields:
        labels = ", ".join(name.replace("_", " ") for name in high_fields)
        return "high", f"Reviewer decisions leave unresolved high-impact fields: {labels}."
    medium_fields = [
        field.variable_name for field in fields if field.reviewed_status in medium_statuses
    ]
    if medium_fields:
        labels = ", ".join(name.replace("_", " ") for name in medium_fields)
        return "medium", f"Reviewer decisions retain non-routine documentation states: {labels}."
    return "low", "Reviewer decisions leave all four fields explicitly supported."


def initial_review_decisions(result: ExtractionResult) -> dict[str, dict[str, Any]]:
    """Create JSON-safe editable state for each extracted field."""

    return {
        variable.variable_name: ReviewDecision(
            variable_name=variable.variable_name,
            action=ReviewAction.PENDING,
            corrected_value=variable.extracted_value,
            corrected_status=variable.documentation_status,
            conflicting_alternatives=list(variable.alternatives),
            corrected_evidence_texts=[],
            reason="",
        ).model_dump(mode="json")
        for variable in result.variables
    }


def copy_decisions(decisions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return deepcopy(decisions)


def validate_review_decisions(
    result: ExtractionResult,
    decisions: dict[str, dict[str, Any]],
    report_text: str | None = None,
) -> list[str]:
    """Return reviewer-facing validation errors without changing state."""

    errors: list[str] = []
    expected = {variable.variable_name for variable in result.variables}
    received = set(decisions)
    if received != expected:
        missing = sorted(expected - received)
        extra = sorted(received - expected)
        if missing:
            errors.append(f"Missing review decisions for: {', '.join(missing)}.")
        if extra:
            errors.append(f"Unexpected review decisions for: {', '.join(extra)}.")

    for variable in result.variables:
        raw = decisions.get(variable.variable_name)
        if raw is None:
            continue
        try:
            decision = ReviewDecision.model_validate(raw)
        except Exception as error:
            errors.append(f"{variable.variable_name}: invalid review decision ({error}).")
            continue

        if decision.variable_name != variable.variable_name:
            errors.append(
                f"{variable.variable_name}: decision variable does not match its field key."
            )

        if decision.action == ReviewAction.PENDING:
            errors.append(f"{variable.variable_name}: choose Accept, Correct, Reject, or Flag.")
        if decision.action == ReviewAction.CORRECTED:
            if decision.corrected_status is None:
                errors.append(f"{variable.variable_name}: choose a corrected QA status.")
                continue

            value = (decision.corrected_value or "").strip() or None
            evidence_texts = [
                text.strip() for text in decision.corrected_evidence_texts if text.strip()
            ]
            submitted_alternatives = [
                item.strip() for item in decision.conflicting_alternatives if item.strip()
            ]
            alternatives = (
                submitted_alternatives
                if decision.corrected_status == DocumentationStatus.CONFLICTING
                else []
            )
            if any(len(text) > 10_000 for text in evidence_texts):
                errors.append(
                    f"{variable.variable_name}: each corrected evidence span must be 10,000 characters or fewer."
                )
            if len(evidence_texts) != len(set(evidence_texts)):
                errors.append(
                    f"{variable.variable_name}: corrected evidence spans must be distinct."
                )
            if decision.corrected_status == DocumentationStatus.CONFLICTING:
                if len(alternatives) < 2 or len(alternatives) != len(evidence_texts):
                    errors.append(
                        f"{variable.variable_name}: provide one distinct alternative value for each of at least two evidence spans."
                    )
                if len({item.casefold() for item in alternatives}) != len(alternatives):
                    errors.append(
                        f"{variable.variable_name}: conflicting alternative values must be distinct."
                    )
            abstention_statuses = {
                DocumentationStatus.NOT_DOCUMENTED,
                DocumentationStatus.NEGATED,
                DocumentationStatus.CONFLICTING,
                DocumentationStatus.UNSUPPORTED,
                DocumentationStatus.MANUAL_REVIEW_REQUIRED,
            }
            value_required_statuses = {
                DocumentationStatus.SUPPORTED,
                DocumentationStatus.UNCERTAIN,
                DocumentationStatus.SUPERSEDED,
            }
            evidence_required_statuses = {
                DocumentationStatus.SUPPORTED,
                DocumentationStatus.CANNOT_BE_ASSIGNED,
                DocumentationStatus.NEGATED,
                DocumentationStatus.UNCERTAIN,
                DocumentationStatus.CONFLICTING,
                DocumentationStatus.SUPERSEDED,
            }

            if decision.corrected_status in value_required_statuses and value is None:
                errors.append(f"{variable.variable_name}: a corrected value is required.")
            if decision.corrected_status in abstention_statuses and value is not None:
                errors.append(
                    f"{variable.variable_name}: {decision.corrected_status.value} requires an empty corrected value."
                )
            if decision.corrected_status == DocumentationStatus.CANNOT_BE_ASSIGNED and value:
                allowed = (
                    variable.variable_name == "pathological_n_category" and value == "pNX"
                ) or (
                    variable.variable_name == "pathological_t_category" and value == "pTX"
                )
                if not allowed:
                    errors.append(
                        f"{variable.variable_name}: only explicit pNX or pTX may carry a cannot-be-assigned value."
                    )
            required_evidence_count = (
                2 if decision.corrected_status == DocumentationStatus.CONFLICTING
                else 1 if decision.corrected_status in evidence_required_statuses
                else 0
            )
            if len(evidence_texts) < required_evidence_count:
                errors.append(
                    f"{variable.variable_name}: {required_evidence_count} distinct exact report "
                    "evidence span(s) are required for the corrected status."
                )
            validated_evidence: list[EvidenceSpan] = []
            validated_contexts: list[str] = []
            for evidence_text in evidence_texts:
                if report_text is None:
                    errors.append(
                        f"{variable.variable_name}: report text is required to validate corrected evidence."
                    )
                elif evidence_text not in report_text:
                    errors.append(
                        f"{variable.variable_name}: corrected evidence must be an exact report substring."
                    )
                else:
                    occurrence_count = report_text.count(evidence_text)
                    if occurrence_count != 1:
                        errors.append(
                            f"{variable.variable_name}: corrected evidence occurs {occurrence_count} times; "
                            "include enough exact context to identify one span."
                        )
                    else:
                        start = report_text.find(evidence_text)
                        validated_evidence.append(
                            EvidenceSpan(
                                text=evidence_text,
                                start_offset=start,
                                end_offset=start + len(evidence_text),
                            )
                        )
                        validated_contexts.append(
                            local_evidence_context(
                                report_text, start, start + len(evidence_text)
                            )
                        )

            value_supported = value is not None and any(
                evidence_text_supports_value(
                    variable.variable_name,
                    value,
                    span.text,
                    validated_contexts[index],
                )
                for index, span in enumerate(validated_evidence)
            )
            if (
                value is not None
                and decision.corrected_status != DocumentationStatus.SUPERSEDED
                and evidence_texts
                and not value_supported
            ):
                errors.append(
                    f"{variable.variable_name}: corrected evidence does not state the corrected value."
                )
            if (
                decision.corrected_status == DocumentationStatus.CONFLICTING
                and len(alternatives) == len(evidence_texts)
            ):
                evidence_by_text = {span.text: span for span in validated_evidence}
                for index, (alternative, evidence_text) in enumerate(
                    zip(alternatives, evidence_texts), start=1
                ):
                    span = evidence_by_text.get(evidence_text)
                    if span is None:
                        continue
                    context = local_evidence_context(
                        report_text or "", span.start_offset, span.end_offset
                    )
                    try:
                        _validate_conflict_alternative(
                            variable.variable_name, alternative, span, context
                        )
                    except Exception:
                        errors.append(
                            f"{variable.variable_name}: conflict evidence {index} does not state "
                            f"a canonical alternative {index}."
                        )
            semantic_statuses = {
                DocumentationStatus.SUPPORTED,
                DocumentationStatus.NOT_DOCUMENTED,
                DocumentationStatus.CANNOT_BE_ASSIGNED,
                DocumentationStatus.NEGATED,
                DocumentationStatus.UNCERTAIN,
                DocumentationStatus.CONFLICTING,
                DocumentationStatus.SUPERSEDED,
            }
            exact_evidence_complete = len(validated_evidence) == len(evidence_texts)
            if (
                decision.corrected_status in semantic_statuses
                and exact_evidence_complete
                and len(validated_evidence) >= required_evidence_count
                and not evidence_text_supports_status(
                    variable.variable_name,
                    decision.corrected_status,
                    [span.text for span in validated_evidence],
                    value=value,
                    alternatives=alternatives or None,
                    context_texts=validated_contexts,
                )
            ):
                errors.append(
                    f"{variable.variable_name}: corrected evidence does not support the "
                    f"{decision.corrected_status.value} documentation status."
                )
            can_check_schema = (
                (decision.corrected_status not in value_required_statuses or value is not None)
                and (
                    decision.corrected_status not in evidence_required_statuses
                    or len(validated_evidence) >= required_evidence_count
                )
                and exact_evidence_complete
            )
            if can_check_schema:
                try:
                    VariableExtraction(
                        variable_name=variable.variable_name,
                        extracted_value=value,
                        documentation_status=decision.corrected_status,
                        evidence=validated_evidence or None,
                        alternatives=alternatives,
                    )
                except Exception:
                    errors.append(
                        f"{variable.variable_name}: corrected value or status violates the canonical field schema."
                    )
        if decision.action in {
            ReviewAction.CORRECTED,
            ReviewAction.REJECTED,
            ReviewAction.FLAGGED,
        } and not decision.reason.strip():
            errors.append(f"{variable.variable_name}: explain this reviewer action.")

    return errors


def create_review_snapshot(
    result: ExtractionResult,
    decisions: dict[str, dict[str, Any]],
    *,
    report_text: str,
    overall_note: str = "",
    timestamp: str | None = None,
    review_started_at: str | None = None,
    review_duration_seconds: float | None = None,
    returned_for_clarification: bool = False,
    reviewer_identity: str = "Unassigned",
    reviewer_identity_verified: bool = False,
) -> tuple[ReviewSnapshot, list[AuditRecord]]:
    """Validate decisions and create deep-copied snapshot and audit event values."""

    errors = validate_review_decisions(result, decisions, report_text)
    if errors:
        raise ValueError("\n".join(errors))

    reviewed_at = timestamp or datetime.now(timezone.utc).isoformat()
    source_report_digest = result.source_report_digest or (
        "sha256:" + hashlib.sha256(report_text.encode("utf-8")).hexdigest()
    )
    extraction_timestamp = result.timestamp.isoformat() if result.timestamp else None
    review_id = f"REV-{uuid4().hex[:12].upper()}"
    fields: list[ReviewedField] = []
    records: list[AuditRecord] = []

    for variable in result.variables:
        decision = ReviewDecision.model_validate(decisions[variable.variable_name])
        reason = decision.reason.strip()
        reviewed_evidence: list[EvidenceSpan]
        reviewed_alternatives: list[str] = []

        if decision.action == ReviewAction.ACCEPTED:
            reviewed_value = variable.extracted_value
            reviewed_status = variable.documentation_status
            reviewed_evidence = [
                item.model_copy(deep=True) for item in variable.evidence or []
            ]
            reviewed_alternatives = list(variable.alternatives)
            reason = reason or "Reviewer accepted the proposed value and QA status."
        elif decision.action == ReviewAction.CORRECTED:
            reviewed_value = (decision.corrected_value or "").strip() or None
            reviewed_status = decision.corrected_status or DocumentationStatus.SUPPORTED
            corrected_evidence = [
                text.strip() for text in decision.corrected_evidence_texts if text.strip()
            ]
            if reviewed_status == DocumentationStatus.CONFLICTING:
                reviewed_alternatives = [
                    item.strip()
                    for item in decision.conflicting_alternatives
                    if item.strip()
                ]
            if corrected_evidence:
                reviewed_evidence = [
                    EvidenceSpan(
                        text=evidence_text,
                        start_offset=report_text.find(evidence_text),
                        end_offset=report_text.find(evidence_text) + len(evidence_text),
                    )
                    for evidence_text in corrected_evidence
                ]
            else:
                reviewed_evidence = []
        elif decision.action == ReviewAction.REJECTED:
            reviewed_value = None
            reviewed_status = DocumentationStatus.UNSUPPORTED
            reviewed_evidence = []
        else:  # flagged_for_review
            reviewed_value = None
            reviewed_status = DocumentationStatus.MANUAL_REVIEW_REQUIRED
            reviewed_evidence = []

        reviewed_field = ReviewedField(
            variable_name=variable.variable_name,
            original_output=variable.model_copy(deep=True),
            reviewed_value=reviewed_value,
            reviewed_status=reviewed_status,
            review_action=decision.action,
            review_reason=reason,
            evidence=reviewed_evidence,
            reviewed_alternatives=reviewed_alternatives,
        )
        fields.append(reviewed_field)
        records.append(
            AuditRecord(
                audit_id=f"AUD-{uuid4().hex[:12].upper()}",
                review_id=review_id,
                report_id=result.report_id,
                method=result.method,
                model_version=result.model_version,
                source_report_digest=source_report_digest,
                extraction_timestamp=extraction_timestamp,
                reviewer_identity=reviewer_identity.strip() or "Unassigned",
                reviewer_identity_verified=reviewer_identity_verified,
                variable_name=variable.variable_name,
                original_output=variable.model_copy(deep=True),
                reviewer_action=decision.action,
                corrected_value=(
                    reviewed_value if decision.action == ReviewAction.CORRECTED else None
                ),
                resulting_value=reviewed_value,
                resulting_status=reviewed_status,
                resulting_evidence=[item.model_copy(deep=True) for item in reviewed_evidence],
                resulting_alternatives=list(reviewed_alternatives),
                review_reason=reason,
                timestamp=reviewed_at,
            )
        )

    reviewed_priority, reviewed_priority_reason = _reviewed_priority(fields)
    snapshot = ReviewSnapshot(
        review_id=review_id,
        report_id=result.report_id,
        method=result.method,
        model_version=result.model_version,
        source_report_digest=source_report_digest,
        extraction_timestamp=extraction_timestamp,
        document_provenance=(
            result.document_provenance.model_copy(deep=True)
            if result.document_provenance else None
        ),
        reviewer_identity=reviewer_identity.strip() or "Unassigned",
        reviewer_identity_verified=reviewer_identity_verified,
        reviewed_at=reviewed_at,
        review_started_at=review_started_at,
        review_duration_seconds=review_duration_seconds,
        returned_for_clarification=returned_for_clarification,
        original_review_priority=result.review_priority,
        original_review_priority_reason=result.review_priority_reason,
        review_priority=reviewed_priority,
        review_priority_reason=reviewed_priority_reason,
        overall_note=overall_note.strip(),
        fields=fields,
    )
    return snapshot, records


def decisions_complete(
    result: ExtractionResult,
    decisions: dict[str, dict[str, Any]],
    report_text: str | None = None,
) -> bool:
    return not validate_review_decisions(result, decisions, report_text)


def review_progress(
    result: ExtractionResult,
    decisions: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    reviewed = 0
    for variable in result.variables:
        raw = decisions.get(variable.variable_name, {})
        if raw.get("action", ReviewAction.PENDING.value) != ReviewAction.PENDING.value:
            reviewed += 1
    return reviewed, len(result.variables or CORE_VARIABLES)
