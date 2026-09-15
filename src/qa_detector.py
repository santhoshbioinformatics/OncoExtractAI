"""Documentation QA and deterministic review prioritization."""

from __future__ import annotations

import re

from .extractor import EXPLICIT_UPDATE_PATTERN, SECTION_HEADING_PATTERN
from .schemas import (
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    QAIssue,
    QAIssueType,
    VariableExtraction,
)


class QADetector:
    """Add report-level QA without reinterpreting explicit clinical categories."""

    def detect_issues(
        self,
        report_text: str,
        result: ExtractionResult,
    ) -> list[QAIssue]:
        if not isinstance(report_text, str):
            raise TypeError("report_text must be a string")

        issues = self._check_explicit_amendments(report_text)
        for variable in result.variables:
            issues.extend(self._check_variable_status(variable))
            if variable.variable_name == "pathological_n_category":
                issues.extend(self._check_nodal_consistency(variable, report_text))
        return issues

    @staticmethod
    def _span(report_text: str, start: int, end: int) -> EvidenceSpan:
        return EvidenceSpan(
            text=report_text[start:end],
            start_offset=start,
            end_offset=end,
        )

    def _check_explicit_amendments(self, report_text: str) -> list[QAIssue]:
        """Flag only headings/content that explicitly establish an update."""

        headings = list(SECTION_HEADING_PATTERN.finditer(report_text))
        if not headings:
            return []

        kinds = [
            next(name for name, value in heading.groupdict().items() if value is not None)
            for heading in headings
        ]
        evidence: list[EvidenceSpan] = []
        relationship_found = False

        preliminary_positions = [
            heading.start()
            for heading, kind in zip(headings, kinds)
            if kind == "preliminary"
        ]
        for heading, kind in zip(headings, kinds):
            if kind in {"revised", "amended"}:
                relationship_found = True
                evidence.append(self._span(report_text, heading.start(), heading.end()))
            elif kind == "final" and any(
                preliminary < heading.start() for preliminary in preliminary_positions
            ):
                relationship_found = True
                evidence.append(self._span(report_text, heading.start(), heading.end()))

        for index, (heading, kind) in enumerate(zip(headings, kinds)):
            if kind != "addendum":
                continue
            section_end = (
                headings[index + 1].start()
                if index + 1 < len(headings)
                else len(report_text)
            )
            update = EXPLICIT_UPDATE_PATTERN.search(
                report_text[heading.start():section_end]
            )
            if not update:
                continue
            relationship_found = True
            evidence.append(self._span(report_text, heading.start(), heading.end()))
            update_start = heading.start() + update.start()
            update_end = heading.start() + update.end()
            evidence.append(self._span(report_text, update_start, update_end))

        if not relationship_found:
            return []

        unique_evidence: list[EvidenceSpan] = []
        seen: set[tuple[int, int]] = set()
        for span in evidence:
            key = (span.start_offset, span.end_offset)
            if key not in seen:
                unique_evidence.append(span)
                seen.add(key)

        return [
            QAIssue(
                issue_type=QAIssueType.SUPERSEDED,
                description=(
                    "The report explicitly identifies preliminary, revised, amended, "
                    "or corrected documentation."
                ),
                severity="high",
                evidence=unique_evidence,
                suggestion="Confirm the selected current value and retain older evidence in the audit trail.",
            )
        ]

    def _check_variable_status(self, variable: VariableExtraction) -> list[QAIssue]:
        if variable.documentation_status == DocumentationStatus.CONFLICTING:
            return [
                QAIssue(
                    issue_type=QAIssueType.CONFLICTING,
                    variable_name=variable.variable_name,
                    description="The extractor abstained because explicit alternatives conflict.",
                    severity="high",
                    evidence=variable.evidence,
                    suggestion="Resolve the alternatives manually.",
                )
            ]
        if variable.documentation_status == DocumentationStatus.UNCERTAIN:
            return [
                QAIssue(
                    issue_type=QAIssueType.UNCERTAIN,
                    variable_name=variable.variable_name,
                    description="The retained candidate is qualified by uncertainty language.",
                    severity="medium",
                    evidence=variable.evidence,
                    suggestion="Review the candidate before acceptance.",
                )
            ]
        return []

    def _check_nodal_consistency(
        self,
        variable: VariableExtraction,
        report_text: str,
    ) -> list[QAIssue]:
        issues: list[QAIssue] = []
        no_nodes = re.search(
            r"\b(?:no\s+(?:regional\s+)?lymph\s+nodes?\s+(?:were\s+)?(?:submitted|identified|assessed|evaluated)"
            r"|regional\s+lymph\s+nodes?\s+cannot\s+be\s+assessed)\b[^\n.]*\.?",
            report_text,
            re.IGNORECASE,
        )
        explicit_pnx = re.search(r"\bpNX\b", report_text, re.IGNORECASE)

        if no_nodes and (
            variable.extracted_value == "pN0"
            or variable.documentation_status == DocumentationStatus.SUPPORTED
        ):
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.CONFLICTING,
                    variable_name="pathological_n_category",
                    description=(
                        "The extraction assigns a supported pN category although the report "
                        "states that lymph nodes were not assessed."
                    ),
                    severity="high",
                    evidence=[self._span(report_text, no_nodes.start(), no_nodes.end())],
                    suggestion="Use cannot_be_assigned and do not infer pN0 from absent assessment.",
                )
            )

        if explicit_pnx and variable.extracted_value not in {None, "pNX"}:
            issues.append(
                QAIssue(
                    issue_type=QAIssueType.CONFLICTING,
                    variable_name="pathological_n_category",
                    description=f"The report explicitly states pNX but the extraction returns {variable.extracted_value!r}.",
                    severity="high",
                    evidence=[
                        self._span(
                            report_text,
                            explicit_pnx.start(),
                            explicit_pnx.end(),
                        )
                    ],
                    suggestion="Keep pNX distinct from pN0 and from not_documented.",
                )
            )
        return issues

    def _priority_and_reason(self, result: ExtractionResult) -> tuple[str, str]:
        issues = result.qa_issues
        if not issues:
            return "low", "No documentation issues detected."
        if any(issue.severity == "critical" for issue in issues):
            return "critical", "Critical evidence validation failure requires manual review."

        conflicts = [issue for issue in issues if issue.issue_type == QAIssueType.CONFLICTING]
        if conflicts:
            fields = ", ".join(
                sorted({issue.variable_name or "report" for issue in conflicts})
            )
            return "high", f"Conflicting documentation requires review: {fields}."

        if any(issue.issue_type == QAIssueType.SUPERSEDED for issue in issues):
            return (
                "high",
                "An explicit amendment or final revision supersedes older documentation.",
            )

        high_impact = [issue for issue in issues if issue.severity == "high"]
        if high_impact:
            fields = ", ".join(
                sorted({issue.variable_name or "report" for issue in high_impact})
            )
            return "high", f"High-impact documentation issue requires review: {fields}."

        uncertain = [
            issue for issue in issues if issue.issue_type == QAIssueType.UNCERTAIN
        ]
        if uncertain:
            fields = ", ".join(
                sorted({issue.variable_name or "report" for issue in uncertain})
            )
            return "medium", f"Uncertain documentation requires review: {fields}."

        fields = ", ".join(
            sorted({issue.variable_name or "report" for issue in issues})
        )
        return "medium", f"Documentation requires review: {fields}."

    def prioritize_review(self, result: ExtractionResult) -> str:
        """Return the deterministic priority derived from the current QA issues."""

        priority, _ = self._priority_and_reason(result)
        return priority

    def run_full_qa(
        self,
        report_text: str,
        result: ExtractionResult,
    ) -> ExtractionResult:
        """Add non-duplicate QA issues and synchronize priority and explanation."""

        detected = self.detect_issues(report_text, result)
        existing = {(issue.issue_type, issue.variable_name) for issue in result.qa_issues}
        for issue in detected:
            key = (issue.issue_type, issue.variable_name)
            if key not in existing:
                result.qa_issues.append(issue)
                existing.add(key)

        priority, reason = self._priority_and_reason(result)
        if priority != "low":
            result.manual_review_required = True
            result.review_priority = priority
        else:
            result.review_priority = "low"
            result.manual_review_required = False
        result.review_priority_reason = reason
        return result
