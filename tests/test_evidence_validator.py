"""Fail-closed evidence validation regression tests."""

import unittest

from src.evidence_validator import EvidenceValidator
from src.schemas import (
    CORE_VARIABLES,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    QAIssueType,
    VariableExtraction,
)


def span(report, evidence_text, occurrence=0):
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = report.index(evidence_text, search_from)
        search_from = start + 1
    return EvidenceSpan(
        text=evidence_text,
        start_offset=start,
        end_offset=start + len(evidence_text),
    )


def result_with(target):
    variables = []
    for name in CORE_VARIABLES:
        if name == target.variable_name:
            variables.append(target)
        else:
            variables.append(
                VariableExtraction(
                    variable_name=name,
                    extracted_value=None,
                    documentation_status=DocumentationStatus.NOT_DOCUMENTED,
                )
            )
    return ExtractionResult(
        report_id="TEST-EVIDENCE",
        method="evidence_first",
        variables=variables,
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="Fixture contains intentionally incomplete fields.",
    )


class TestEvidenceValidator(unittest.TestCase):
    def setUp(self):
        self.validator = EvidenceValidator()
        self.report = "FINAL DIAGNOSIS: Tumor size: 3.2 cm. Pathological N category: pN0."

    def test_valid_exact_evidence_passes(self):
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="3.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(self.report, "Tumor size: 3.2 cm")],
        )
        issues = self.validator.validate(result_with(target), self.report)
        self.assertFalse(any(issue.severity == "critical" for issue in issues))

    def test_missing_evidence_is_detected_even_on_malformed_construct(self):
        valid_target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="3.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(self.report, "3.2 cm")],
        )
        result = result_with(valid_target)
        index = next(
            index
            for index, item in enumerate(result.variables)
            if item.variable_name == "tumor_size"
        )
        result.variables[index] = valid_target.model_copy(update={"evidence": None})

        issues = self.validator.validate(result, self.report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "tumor_size"
                for issue in issues
            )
        )

    def test_wrong_value_with_real_evidence_is_unsupported(self):
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="9.9 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(self.report, "3.2 cm")],
        )
        result = result_with(target)

        issues = self.validator.validate(result, self.report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.severity == "critical"
                for issue in issues
            )
        )

    def test_text_not_in_report_and_wrong_offsets_are_both_detected(self):
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="9.9 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[EvidenceSpan(text="9.9 cm", start_offset=0, end_offset=6)],
        )
        issues = self.validator.validate(result_with(target), self.report)
        issue_types = {issue.issue_type for issue in issues}
        self.assertIn(QAIssueType.UNSUPPORTED_EVIDENCE, issue_types)
        self.assertIn(QAIssueType.EVIDENCE_MISMATCH, issue_types)

    def test_out_of_range_offsets_are_detected(self):
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="3.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[EvidenceSpan(text="3.2 cm", start_offset=999, end_offset=1005)],
        )
        issues = self.validator.validate(result_with(target), self.report)
        self.assertTrue(
            any(issue.issue_type == QAIssueType.OFFSET_MISMATCH for issue in issues)
        )

    def test_abstained_evidence_is_still_validated(self):
        target = VariableExtraction(
            variable_name="histologic_diagnosis",
            extracted_value=None,
            documentation_status=DocumentationStatus.NEGATED,
            evidence=[EvidenceSpan(text="No carcinoma", start_offset=0, end_offset=12)],
        )
        issues = self.validator.validate(result_with(target), self.report)
        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                for issue in issues
            )
        )

    def test_validate_and_flag_removes_unsupported_candidate(self):
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="9.9 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(self.report, "3.2 cm")],
        )
        result = self.validator.validate_and_flag(result_with(target), self.report)
        tumor_size = next(
            item for item in result.variables if item.variable_name == "tumor_size"
        )

        self.assertIsNone(tumor_size.extracted_value)
        self.assertEqual(
            tumor_size.documentation_status,
            DocumentationStatus.UNSUPPORTED,
        )
        self.assertTrue(result.manual_review_required)
        self.assertEqual(result.review_priority, "critical")
        self.assertIn("evidence", result.review_priority_reason.lower())

    def test_explicit_pn0_is_not_misread_as_negation(self):
        report = (
            "Eight lymph nodes are negative for metastasis. "
            "Pathological N category: pN0."
        )
        target = VariableExtraction(
            variable_name="pathological_n_category",
            extracted_value="pN0",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, "Pathological N category: pN0")],
        )
        issues = self.validator.validate(result_with(target), report)
        self.assertFalse(any(issue.issue_type == QAIssueType.NEGATED for issue in issues))

    def test_supported_negated_candidate_is_converted_to_abstention(self):
        report = "No evidence of invasive adenocarcinoma is identified."
        target = VariableExtraction(
            variable_name="histologic_diagnosis",
            extracted_value="Invasive adenocarcinoma",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, "invasive adenocarcinoma")],
        )
        result = self.validator.validate_and_flag(result_with(target), report)
        diagnosis = next(
            item
            for item in result.variables
            if item.variable_name == "histologic_diagnosis"
        )
        self.assertIsNone(diagnosis.extracted_value)
        self.assertEqual(diagnosis.documentation_status, DocumentationStatus.NEGATED)
        self.assertEqual(
            report[
                diagnosis.evidence[0].start_offset:diagnosis.evidence[0].end_offset
            ],
            diagnosis.evidence[0].text,
        )

    def test_margin_measurement_is_rejected_as_tumor_size(self):
        report = "Closest surgical margin measures 0.2 cm."
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="0.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, "margin measures 0.2 cm")],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "tumor_size"
                and issue.severity == "critical"
                for issue in issues
            )
        )

    def test_historical_measurement_is_not_current_value_support(self):
        report = "Prior tumor measured 3.2 cm. Revised tumor size: 4.0 cm."
        target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="3.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, "Prior tumor measured 3.2 cm")],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "tumor_size"
                for issue in issues
            )
        )

    def test_status_cue_must_refer_to_the_target_variable(self):
        report = "No evidence of lymphovascular invasion. Invasive adenocarcinoma is present."
        target = VariableExtraction(
            variable_name="histologic_diagnosis",
            extracted_value=None,
            documentation_status=DocumentationStatus.NEGATED,
            evidence=[span(report, "No evidence of lymphovascular invasion")],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "histologic_diagnosis"
                for issue in issues
            )
        )

    def test_cannot_assign_cue_must_match_the_staging_axis(self):
        report = "No regional lymph nodes were submitted."
        target = VariableExtraction(
            variable_name="pathological_t_category",
            extracted_value=None,
            documentation_status=DocumentationStatus.CANNOT_BE_ASSIGNED,
            evidence=[span(report, report)],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "pathological_t_category"
                for issue in issues
            )
        )

    def test_negation_in_another_sentence_and_site_does_not_cancel_value(self):
        report = "Lung: Adenocarcinoma. Brain: no evidence of Adenocarcinoma."
        target = VariableExtraction(
            variable_name="histologic_diagnosis",
            extracted_value="Adenocarcinoma",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, "Adenocarcinoma", occurrence=0)],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertFalse(
            any(
                issue.issue_type == QAIssueType.NEGATED
                or issue.severity == "critical"
                for issue in issues
                if issue.variable_name == "histologic_diagnosis"
            )
        )

    def test_supported_status_rejects_target_tied_uncertainty(self):
        report = "Invasive adenocarcinoma is favored."
        target = VariableExtraction(
            variable_name="histologic_diagnosis",
            extracted_value="Invasive adenocarcinoma",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(report, report)],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                and issue.variable_name == "histologic_diagnosis"
                for issue in issues
            )
        )

    def test_explicitly_superseded_old_value_is_semantically_valid(self):
        report = "pT1c was superseded by pT2b."
        target = VariableExtraction(
            variable_name="pathological_t_category",
            extracted_value="pT1c",
            documentation_status=DocumentationStatus.SUPERSEDED,
            evidence=[span(report, report)],
        )

        issues = self.validator.validate(result_with(target), report)

        self.assertFalse(
            any(
                issue.severity == "critical"
                for issue in issues
                if issue.variable_name == "pathological_t_category"
            )
        )

    def test_process_status_does_not_accept_unrelated_evidence(self):
        report = "Unrelated administrative text."
        for status in (
            DocumentationStatus.UNSUPPORTED,
            DocumentationStatus.MANUAL_REVIEW_REQUIRED,
        ):
            with self.subTest(status=status):
                target = VariableExtraction(
                    variable_name="tumor_size",
                    extracted_value=None,
                    documentation_status=status,
                    evidence=[span(report, report)],
                )
                issues = self.validator.validate(result_with(target), report)
                self.assertTrue(
                    any(
                        issue.issue_type == QAIssueType.UNSUPPORTED_EVIDENCE
                        and issue.variable_name == "tumor_size"
                        for issue in issues
                    )
                )

    def test_schema_bypass_is_rejected_instead_of_returned_invalid(self):
        valid_target = VariableExtraction(
            variable_name="tumor_size",
            extracted_value="3.2 cm",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[span(self.report, "Tumor size: 3.2 cm")],
        )
        result = result_with(valid_target)
        invalid_target = valid_target.model_copy(
            update={"alternatives": ["bad", "also bad"]}
        )
        target_index = next(
            index
            for index, item in enumerate(result.variables)
            if item.variable_name == "tumor_size"
        )
        result.variables[target_index] = invalid_target

        with self.assertRaisesRegex(ValueError, "structured schema"):
            self.validator.validate_and_flag(result, self.report)


if __name__ == "__main__":
    unittest.main()
