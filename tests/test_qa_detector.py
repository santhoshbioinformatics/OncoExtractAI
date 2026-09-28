"""QA prioritization and amendment-boundary regression tests."""

import unittest

from src.extractor import evidence_first_extract
from src.qa_detector import QADetector
from src.schemas import (
    CORE_VARIABLES,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    QAIssue,
    QAIssueType,
    VariableExtraction,
)


def missing_result(issues=None):
    return ExtractionResult(
        report_id="TEST-QA",
        method="evidence_first",
        variables=[
            VariableExtraction(
                variable_name=name,
                extracted_value=None,
                documentation_status=DocumentationStatus.NOT_DOCUMENTED,
            )
            for name in CORE_VARIABLES
        ],
        qa_issues=issues or [],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="Fixture contains documentation issues.",
    )


class TestQADetector(unittest.TestCase):
    def setUp(self):
        self.detector = QADetector()

    def test_clean_case_has_no_false_size_conflict_or_pn_negation(self):
        report = (
            "SPECIMEN DESCRIPTION:\nThe specimen measures 12 x 8 x 6 cm.\n"
            "FINAL DIAGNOSIS:\n"
            "Invasive adenocarcinoma. Tumor size: 3.2 cm in greatest dimension. "
            "Pathological T category: pT2a. "
            "Eight nodes are negative for metastasis. "
            "Pathological N category: pN0."
        )
        result = evidence_first_extract(report, "CLEAN")
        updated = self.detector.run_full_qa(report, result)

        self.assertFalse(updated.qa_issues)
        self.assertEqual(updated.review_priority, "low")
        self.assertFalse(updated.manual_review_required)

    def test_explicit_addendum_has_exact_evidence_offsets(self):
        report = (
            "PRELIMINARY DIAGNOSIS:\nTumor size: 2.8 cm. Pathological T category: pT1c.\n"
            "ADDENDUM:\nThis addendum explicitly supersedes the preliminary values.\n"
            "REVISED FINAL DIAGNOSIS:\nTumor size: 4.1 cm. Pathological T category: pT2b."
        )
        result = evidence_first_extract(report, "AMENDED")
        issues = self.detector.detect_issues(report, result)
        superseded = [
            issue for issue in issues if issue.issue_type == QAIssueType.SUPERSEDED
        ]

        self.assertTrue(superseded)
        for issue in superseded:
            for evidence in issue.evidence or []:
                self.assertEqual(
                    report[evidence.start_offset:evidence.end_offset],
                    evidence.text,
                )

    def test_unrelated_addendum_is_not_called_supersession(self):
        report = (
            "FINAL DIAGNOSIS:\nInvasive adenocarcinoma.\n"
            "ADDENDUM:\nA stain confirms the diagnosis."
        )
        result = evidence_first_extract(report, "ADDENDUM")
        issues = self.detector.detect_issues(report, result)
        self.assertFalse(
            any(issue.issue_type == QAIssueType.SUPERSEDED for issue in issues)
        )

    def test_no_nodes_conflicts_with_supported_pn0_but_is_not_used_to_infer_it(self):
        report = (
            "Pathological N category: pN0. "
            "No regional lymph nodes were submitted for evaluation."
        )
        start = report.index("pN0")
        p_n = VariableExtraction(
            variable_name="pathological_n_category",
            extracted_value="pN0",
            documentation_status=DocumentationStatus.SUPPORTED,
            evidence=[
                EvidenceSpan(
                    text="pN0",
                    start_offset=start,
                    end_offset=start + 3,
                )
            ],
        )
        variables = [
            p_n
            if name == "pathological_n_category"
            else VariableExtraction(
                variable_name=name,
                extracted_value=None,
                documentation_status=DocumentationStatus.NOT_DOCUMENTED,
            )
            for name in CORE_VARIABLES
        ]
        result = ExtractionResult(
            report_id="NODES",
            method="evidence_first",
            variables=variables,
            manual_review_required=True,
            review_priority="medium",
            review_priority_reason="Fixture contains intentionally incomplete fields.",
        )

        issues = self.detector.detect_issues(report, result)

        self.assertTrue(
            any(issue.issue_type == QAIssueType.CONFLICTING for issue in issues)
        )
        self.assertEqual(p_n.extracted_value, "pN0")

    def test_priority_and_reason_follow_conflict(self):
        issues = [
            QAIssue(
                issue_type=QAIssueType.CONFLICTING,
                variable_name="tumor_size",
                description="Two explicit sizes remain unresolved.",
                severity="high",
            )
        ]
        result = missing_result(issues)
        result = self.detector.run_full_qa("", result)

        self.assertEqual(result.review_priority, "high")
        self.assertTrue(result.manual_review_required)
        self.assertIn("conflicting", result.review_priority_reason.lower())

    def test_single_uncertain_field_always_requires_review(self):
        issues = [
            QAIssue(
                issue_type=QAIssueType.UNCERTAIN,
                variable_name="tumor_size",
                description="Approximate measurement.",
                severity="medium",
            )
        ]
        result = missing_result(issues)
        result = self.detector.run_full_qa("", result)
        self.assertEqual(result.review_priority, "medium")
        self.assertTrue(result.manual_review_required)

    def test_full_qa_is_idempotent_for_issue_keys(self):
        report = (
            "PRELIMINARY DIAGNOSIS:\nPathological T category: pT1c.\n"
            "REVISED FINAL DIAGNOSIS:\nPathological T category: pT2b."
        )
        result = evidence_first_extract(report, "IDEMPOTENT")
        once = self.detector.run_full_qa(report, result)
        count = len(once.qa_issues)
        twice = self.detector.run_full_qa(report, once)
        self.assertEqual(len(twice.qa_issues), count)

    def test_synoptic_report_with_missing_stage_is_flagged_as_extraction_gap(self):
        report = (
            "SYNOPTIC REPORT - LUNG\n"
            "Tumor Size: 3.3 cm\n"
            "Lymph Node Involvement:\n"
            "Seven nodes negative for tumor (0/7)."
        )
        result = evidence_first_extract(report, "SYNOPTIC-GAP")

        updated = self.detector.run_full_qa(report, result)
        stage_gaps = [
            issue
            for issue in updated.qa_issues
            if issue.issue_type == QAIssueType.OCR_QUALITY
            and issue.variable_name
            in {"pathological_t_category", "pathological_n_category"}
        ]

        self.assertEqual(len(stage_gaps), 2)
        self.assertTrue(all(issue.severity == "high" for issue in stage_gaps))
        self.assertTrue(all("Do not derive" in issue.suggestion for issue in stage_gaps))


if __name__ == "__main__":
    unittest.main()
