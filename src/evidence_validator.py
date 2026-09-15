"""Fail-closed validation of evidence spans and their proposed values."""

from __future__ import annotations

import re

from pydantic import ValidationError

from .schemas import (
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    QAIssue,
    QAIssueType,
    VariableExtraction,
    evidence_text_supports_status,
    evidence_text_supports_value,
    local_evidence_context,
)


class EvidenceValidator:
    """Validate all evidence, offsets, and value-to-evidence relationships."""

    def validate(self, result: ExtractionResult, report_text: str) -> list[QAIssue]:
        if not isinstance(report_text, str):
            raise TypeError("report_text must be a string")

        issues: list[QAIssue] = []
        for variable in result.variables:
            if variable.extracted_value is not None and not variable.evidence:
                issues.append(
                    QAIssue(
                        issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                        variable_name=variable.variable_name,
                        description="A non-null candidate has no evidence span.",
                        severity="critical",
                        suggestion="Abstain unless exact supporting evidence is supplied.",
                    )
                )

            for evidence in variable.evidence or []:
                substring_issue = self._check_substring(evidence, report_text, variable)
                if substring_issue:
                    issues.append(substring_issue)
                offset_issue = self._check_offsets(evidence, report_text, variable)
                if offset_issue:
                    issues.append(offset_issue)

            negation_issue = self._check_negation(variable, report_text)
            if negation_issue:
                issues.append(negation_issue)

            # A supported candidate that is actually negated is handled as a
            # semantic negation rather than producing a duplicate support error.
            if negation_issue is None:
                if variable.documentation_status == DocumentationStatus.SUPERSEDED:
                    status_issue = self._check_status_supported(variable, report_text)
                    if status_issue:
                        issues.append(status_issue)
                else:
                    support_issue = self._check_value_supported(variable, report_text)
                    if support_issue:
                        issues.append(support_issue)
                    else:
                        status_issue = self._check_status_supported(variable, report_text)
                        if status_issue:
                            issues.append(status_issue)

        schema_issue = self._check_schema_validity(result)
        if schema_issue:
            issues.append(schema_issue)
        return issues

    def _check_substring(
        self,
        evidence: EvidenceSpan,
        report_text: str,
        variable: VariableExtraction,
    ) -> QAIssue | None:
        if evidence.text not in report_text:
            return QAIssue(
                issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                variable_name=variable.variable_name,
                description=f"Evidence is not an exact report substring: {evidence.text[:100]!r}.",
                severity="critical",
                suggestion="Copy evidence verbatim from the submitted report.",
            )
        return None

    def _check_offsets(
        self,
        evidence: EvidenceSpan,
        report_text: str,
        variable: VariableExtraction,
    ) -> QAIssue | None:
        if not (
            0 <= evidence.start_offset < evidence.end_offset <= len(report_text)
        ):
            return QAIssue(
                issue_type=QAIssueType.OFFSET_MISMATCH,
                variable_name=variable.variable_name,
                description=(
                    "Evidence offsets are outside the report: "
                    f"start={evidence.start_offset}, end={evidence.end_offset}, "
                    f"report_length={len(report_text)}."
                ),
                severity="critical",
                suggestion="Use zero-based, end-exclusive character offsets.",
            )

        at_offsets = report_text[evidence.start_offset:evidence.end_offset]
        if at_offsets != evidence.text:
            return QAIssue(
                issue_type=QAIssueType.EVIDENCE_MISMATCH,
                variable_name=variable.variable_name,
                description=(
                    f"Evidence offsets resolve to {at_offsets[:50]!r}, "
                    f"not {evidence.text[:50]!r}."
                ),
                severity="critical",
                suggestion="Recompute offsets against the unmodified report text.",
            )
        return None

    def _check_value_supported(
        self, variable: VariableExtraction, report_text: str
    ) -> QAIssue | None:
        value = variable.extracted_value
        if value is None:
            return None
        if not variable.evidence:
            return None  # The missing-evidence check emits the critical issue.

        supported = any(
            evidence_text_supports_value(
                variable.variable_name,
                value,
                span.text,
                local_evidence_context(
                    report_text, span.start_offset, span.end_offset
                ),
            )
            for span in variable.evidence
        )

        if not supported:
            return QAIssue(
                issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                variable_name=variable.variable_name,
                description=(
                    f"The proposed value {value!r} is not supported by its exact "
                    "evidence in local context."
                ),
                severity="critical",
                evidence=variable.evidence,
                suggestion=(
                    "Abstain or attach evidence that explicitly states the value "
                    "for this variable in the current context."
                ),
            )
        return None

    def _check_status_supported(
        self, variable: VariableExtraction, report_text: str
    ) -> QAIssue | None:
        evidence = variable.evidence or []
        contexts = [
            local_evidence_context(report_text, span.start_offset, span.end_offset)
            for span in evidence
        ]
        if evidence_text_supports_status(
            variable.variable_name,
            variable.documentation_status,
            [span.text for span in evidence],
            value=variable.extracted_value,
            alternatives=getattr(variable, "alternatives", None),
            context_texts=contexts,
        ):
            return None
        return QAIssue(
            issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
            variable_name=variable.variable_name,
            description=(
                f"The evidence does not support documentation status "
                f"{variable.documentation_status.value!r} in local context."
            ),
            severity="critical",
            evidence=variable.evidence,
            suggestion="Correct the status or attach exact evidence with the required semantics.",
        )

    def _check_negation(
        self,
        variable: VariableExtraction,
        report_text: str,
    ) -> QAIssue | None:
        value = variable.extracted_value
        if value is None or variable.documentation_status == DocumentationStatus.NEGATED:
            return None
        if variable.variable_name == "pathological_n_category" and value == "pN0":
            return None  # Negative nodes are affirmative evidence for an explicit pN0.

        escaped = re.escape(value)
        patterns = [
            rf"\bno\s+evidence\s+of\s+{escaped}\b",
            rf"\bnegative\s+for\s+{escaped}\b",
            rf"\babsence\s+of\s+{escaped}\b",
            rf"\bnot\s+{escaped}\b",
            rf"\b{escaped}\s+(?:is|was|are|were)\s+not\s+(?:identified|present|found|detected)\b",
            rf"\bconcern\s+for\s+{escaped}.{{0,60}}?ruled\s+out\b",
        ]

        for evidence in variable.evidence or []:
            context = local_evidence_context(
                report_text, evidence.start_offset, evidence.end_offset
            )
            context_start = -1
            search_from = 0
            while context:
                candidate_start = report_text.find(context, search_from)
                if candidate_start < 0:
                    break
                if (
                    candidate_start <= evidence.start_offset
                    and candidate_start + len(context) >= evidence.end_offset
                ):
                    context_start = candidate_start
                    break
                search_from = candidate_start + 1
            if context_start < 0:
                continue
            for pattern in patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if not match:
                    continue
                start = context_start + match.start()
                end = context_start + match.end()
                negation_span = EvidenceSpan(
                    text=report_text[start:end],
                    start_offset=start,
                    end_offset=end,
                )
                return QAIssue(
                    issue_type=QAIssueType.NEGATED,
                    variable_name=variable.variable_name,
                    description="The proposed value is explicitly negated in its local context.",
                    severity="high",
                    evidence=[negation_span],
                    suggestion="Abstain and classify the field as negated.",
                )
        return None

    def _check_schema_validity(self, result: ExtractionResult) -> QAIssue | None:
        try:
            ExtractionResult.model_validate(result.model_dump())
        except (ValidationError, ValueError, TypeError) as error:
            return QAIssue(
                issue_type=QAIssueType.UNSUPPORTED_EVIDENCE,
                description=f"Extraction output violates the schema: {error}.",
                severity="critical",
                suggestion="Reject malformed structured output before display.",
            )
        return None

    def validate_and_flag(
        self,
        result: ExtractionResult,
        report_text: str,
    ) -> ExtractionResult:
        """Validate in place and force unsafe candidates to abstain."""

        issues = self.validate(result, report_text)
        if any(
            issue.severity == "critical" and issue.variable_name is None
            for issue in issues
        ):
            raise ValueError(
                "Extraction result failed the structured schema and cannot be returned safely."
            )
        existing = {
            (issue.issue_type, issue.variable_name, issue.description)
            for issue in result.qa_issues
        }
        for issue in issues:
            key = (issue.issue_type, issue.variable_name, issue.description)
            if key not in existing:
                result.qa_issues.append(issue)
                existing.add(key)

        for index, variable in enumerate(result.variables):
            variable_issues = [
                issue for issue in issues if issue.variable_name == variable.variable_name
            ]
            critical = any(issue.severity == "critical" for issue in variable_issues)
            negated = next(
                (issue for issue in variable_issues if issue.issue_type == QAIssueType.NEGATED),
                None,
            )
            if critical:
                result.variables[index] = VariableExtraction(
                    variable_name=variable.variable_name,
                    extracted_value=None,
                    documentation_status=DocumentationStatus.UNSUPPORTED,
                    evidence=variable.evidence,
                    notes="Candidate removed because deterministic evidence validation failed.",
                )
            elif negated is not None:
                result.variables[index] = VariableExtraction(
                    variable_name=variable.variable_name,
                    extracted_value=None,
                    documentation_status=DocumentationStatus.NEGATED,
                    evidence=negated.evidence,
                    notes="Candidate removed because it is explicitly negated.",
                )

        if issues:
            result.manual_review_required = True
            if any(issue.severity == "critical" for issue in issues):
                result.review_priority = "critical"
                result.review_priority_reason = (
                    "Critical evidence validation failure requires manual review."
                )
            elif any(issue.issue_type == QAIssueType.NEGATED for issue in issues):
                if result.review_priority == "low":
                    result.review_priority = "high"
                result.review_priority_reason = (
                    "A proposed value was explicitly negated and was removed."
                )
        return result
