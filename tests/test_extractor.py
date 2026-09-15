"""Clinical-rule regression tests for deterministic extraction."""

import unittest

from src.extractor import baseline_extract, evidence_first_extract
from src.schemas import CORE_VARIABLES, DocumentationStatus, QAIssueType
from src.workflow import ExtractionPipelineError, run_both_pipelines


def variable(result, name):
    return next(item for item in result.variables if item.variable_name == name)


def assert_exact_evidence(testcase, result, report):
    for item in result.variables:
        for span in item.evidence or []:
            testcase.assertEqual(
                report[span.start_offset:span.end_offset],
                span.text,
            )


class TestSafeExtraction(unittest.TestCase):
    def setUp(self):
        self.clean_report = (
            "FINAL DIAGNOSIS:\n"
            "- Invasive adenocarcinoma.\n"
            "- Tumor size: 3.2 cm in greatest dimension.\n"
            "- Pathological T category: pT2a.\n"
            "- Eight lymph nodes are negative for metastasis (0/8).\n"
            "- Pathological N category: pN0."
        )

    def test_clean_report_has_four_supported_exact_results(self):
        result = evidence_first_extract(self.clean_report, "TEST-001")

        self.assertEqual(len(result.variables), len(CORE_VARIABLES))
        self.assertTrue(
            all(
                item.documentation_status == DocumentationStatus.SUPPORTED
                for item in result.variables
            )
        )
        self.assertFalse(result.manual_review_required)
        self.assertEqual(result.review_priority, "low")
        self.assertTrue(result.review_priority_reason)
        assert_exact_evidence(self, result, self.clean_report)

    def test_baseline_never_infers_pn_from_negative_node_count(self):
        report = (
            "FINAL DIAGNOSIS:\n"
            "Invasive adenocarcinoma. Tumor size: 2.0 cm. "
            "Pathological T category: pT1b. "
            "Zero of eight lymph nodes show metastasis (0/8)."
        )

        result = baseline_extract(report, "TEST-002")
        pn = variable(result, "pathological_n_category")

        self.assertIsNone(pn.extracted_value)
        self.assertEqual(pn.documentation_status, DocumentationStatus.NOT_DOCUMENTED)

    def test_neither_method_infers_pt_from_measurement(self):
        report = (
            "FINAL DIAGNOSIS:\nInvasive adenocarcinoma. "
            "Tumor size: 2.0 cm. Pathological N category: pN0."
        )

        for extractor in (baseline_extract, evidence_first_extract):
            with self.subTest(extractor=extractor.__name__):
                pt = variable(extractor(report, "TEST-003"), "pathological_t_category")
                self.assertIsNone(pt.extracted_value)
                self.assertEqual(pt.documentation_status, DocumentationStatus.NOT_DOCUMENTED)

    def test_pn0_pnx_no_nodes_and_not_documented_are_distinct(self):
        cases = {
            "pn0": (
                "Pathological N category: pN0.",
                "pN0",
                DocumentationStatus.SUPPORTED,
            ),
            "pnx": (
                "Pathological N category: pNX.",
                "pNX",
                DocumentationStatus.CANNOT_BE_ASSIGNED,
            ),
            "no_nodes": (
                "No regional lymph nodes were submitted for evaluation.",
                None,
                DocumentationStatus.CANNOT_BE_ASSIGNED,
            ),
            "not_documented": (
                "Pathological N category: not documented.",
                None,
                DocumentationStatus.NOT_DOCUMENTED,
            ),
        }

        observed = set()
        for label, (report, expected_value, expected_status) in cases.items():
            with self.subTest(case=label):
                result = evidence_first_extract(report, label)
                pn = variable(result, "pathological_n_category")
                self.assertEqual(pn.extracted_value, expected_value)
                self.assertEqual(pn.documentation_status, expected_status)
                if label in {"pnx", "no_nodes", "not_documented"}:
                    self.assertTrue(pn.evidence)
                assert_exact_evidence(self, result, report)
                observed.add((pn.extracted_value, pn.documentation_status))
        self.assertEqual(len(observed), 4)

    def test_unresolved_conflicts_abstain_with_alternative_evidence(self):
        report = (
            "FINAL DIAGNOSIS:\n"
            "Invasive squamous cell carcinoma. "
            "The gross section records 3.8 cm and the microscopic section records 4.2 cm. "
            "One section records pT2a and another records pT2b. "
            "One section records pN0 and another records pN1."
        )

        result = evidence_first_extract(report, "TEST-004")
        for name in (
            "tumor_size",
            "pathological_t_category",
            "pathological_n_category",
        ):
            with self.subTest(variable=name):
                item = variable(result, name)
                self.assertIsNone(item.extracted_value)
                self.assertEqual(item.documentation_status, DocumentationStatus.CONFLICTING)
                self.assertGreaterEqual(len(item.evidence or []), 2)
                self.assertEqual(len(item.alternatives), len(item.evidence or []))
                self.assertEqual(len({value.casefold() for value in item.alternatives}), 2)
                for alternative, evidence_span in zip(item.alternatives, item.evidence or []):
                    self.assertIn(alternative.casefold(), evidence_span.text.casefold())
        self.assertTrue(result.manual_review_required)
        self.assertEqual(result.review_priority, "high")
        assert_exact_evidence(self, result, report)

    def test_explicit_revision_selects_current_value_but_baseline_abstains(self):
        report = (
            "PRELIMINARY DIAGNOSIS:\n"
            "- Invasive adenocarcinoma.\n"
            "- Tumor size: 2.8 cm.\n"
            "- Pathological T category: pT1c.\n"
            "- Pathological N category: pN0.\n\n"
            "ADDENDUM:\n"
            "This addendum explicitly supersedes the preliminary tumor size and pT.\n\n"
            "REVISED FINAL DIAGNOSIS:\n"
            "- Invasive adenocarcinoma.\n"
            "- Tumor size: 4.1 cm.\n"
            "- Pathological T category: pT2b (amended from pT1c).\n"
            "- Pathological N category: pN0."
        )

        baseline = baseline_extract(report, "TEST-005")
        evidence_first = evidence_first_extract(report, "TEST-005")

        self.assertIsNone(variable(baseline, "tumor_size").extracted_value)
        self.assertIsNone(variable(baseline, "pathological_t_category").extracted_value)
        self.assertEqual(variable(evidence_first, "tumor_size").extracted_value, "4.1 cm")
        self.assertEqual(
            variable(evidence_first, "pathological_t_category").extracted_value,
            "pT2b",
        )
        self.assertNotEqual(
            variable(evidence_first, "pathological_t_category").extracted_value,
            "pT1c",
        )
        self.assertTrue(
            any(
                issue.issue_type == QAIssueType.SUPERSEDED
                for issue in evidence_first.qa_issues
            )
        )
        assert_exact_evidence(self, evidence_first, report)

    def test_unrelated_addendum_does_not_supersede_values(self):
        report = (
            "FINAL DIAGNOSIS:\n"
            "Invasive adenocarcinoma. Tumor size: 2.1 cm. "
            "Pathological T category: pT1c. Pathological N category: pN0.\n\n"
            "ADDENDUM:\nA stain confirms the histologic diagnosis."
        )

        result = evidence_first_extract(report, "TEST-006")

        self.assertEqual(variable(result, "pathological_t_category").extracted_value, "pT1c")
        self.assertFalse(
            any(issue.issue_type == QAIssueType.SUPERSEDED for issue in result.qa_issues)
        )

    def test_uncertainty_retains_candidate_and_requires_review(self):
        report = (
            "FINAL DIAGNOSIS:\n"
            "Invasive adenocarcinoma is favored, but classification remains uncertain.\n"
            "Tumor size: approximately 1.8 cm.\n"
            "Pathological T category: pT1b (provisional).\n"
            "Pathological N category: pN0."
        )

        result = evidence_first_extract(report, "TEST-007")

        for name in ("histologic_diagnosis", "tumor_size", "pathological_t_category"):
            item = variable(result, name)
            self.assertIsNotNone(item.extracted_value)
            self.assertEqual(item.documentation_status, DocumentationStatus.UNCERTAIN)
        self.assertTrue(result.manual_review_required)
        self.assertEqual(result.review_priority, "medium")

    def test_explicit_negation_and_unassignable_pt_are_anchored(self):
        report = (
            "FINAL DIAGNOSIS:\n"
            "No evidence of invasive squamous cell carcinoma is identified.\n"
            "No invasive tumor is identified for measurement.\n"
            "Pathological T category cannot be assigned because no invasive tumor is identified.\n"
            "Pathological N category: not documented."
        )

        result = evidence_first_extract(report, "TEST-008")

        self.assertEqual(
            variable(result, "histologic_diagnosis").documentation_status,
            DocumentationStatus.NEGATED,
        )
        self.assertEqual(
            variable(result, "tumor_size").documentation_status,
            DocumentationStatus.NEGATED,
        )
        self.assertEqual(
            variable(result, "pathological_t_category").documentation_status,
            DocumentationStatus.CANNOT_BE_ASSIGNED,
        )
        self.assertEqual(
            variable(result, "pathological_n_category").documentation_status,
            DocumentationStatus.NOT_DOCUMENTED,
        )
        self.assertTrue(all(item.extracted_value is None for item in result.variables))
        assert_exact_evidence(self, result, report)

    def test_missing_histology_is_not_a_low_priority_clean_case(self):
        report = (
            "Tumor size: 2.0 cm. Pathological T category: pT1b. "
            "Pathological N category: pN0."
        )
        result = evidence_first_extract(report, "TEST-009")
        self.assertTrue(result.manual_review_required)
        self.assertEqual(result.review_priority, "high")

    def test_tumor_measurement_conflict_keeps_qualified_evidence(self):
        report = "Tumor measures 3.2 cm. Tumor measures 4.0 cm."

        item = variable(evidence_first_extract(report, "TEST-MEASURE-CONFLICT"), "tumor_size")

        self.assertEqual(item.documentation_status, DocumentationStatus.CONFLICTING)
        self.assertEqual(item.alternatives, ["3.2 cm", "4.0 cm"])
        self.assertEqual(
            [span.text for span in item.evidence or []],
            ["Tumor measures 3.2 cm", "Tumor measures 4.0 cm"],
        )

    def test_margin_measurement_is_not_extracted_as_tumor_size(self):
        report = "FINAL DIAGNOSIS:\nClosest surgical margin measures 0.2 cm."

        item = variable(evidence_first_extract(report, "TEST-MARGIN"), "tumor_size")

        self.assertIsNone(item.extracted_value)
        self.assertEqual(item.documentation_status, DocumentationStatus.NOT_DOCUMENTED)

    def test_both_workflow_pipelines_preserve_valid_cancer_type(self):
        results = run_both_pipelines(
            {
                "report_id": "TEST-010",
                "text": self.clean_report,
                "cancer_type": "LUAD",
            }
        )
        self.assertEqual(results["baseline"].cancer_type, "LUAD")
        self.assertEqual(results["evidence_first"].cancer_type, "LUAD")
        self.assertEqual(results["ml"].cancer_type, "LUAD")
        self.assertEqual(results["ml"].method, "ml")

    def test_workflow_rejects_invalid_cancer_type(self):
        with self.assertRaises(ExtractionPipelineError):
            run_both_pipelines(
                {
                    "report_id": "TEST-INVALID-CANCER",
                    "text": self.clean_report,
                    "cancer_type": "NOT-A-CANCER",
                }
            )


if __name__ == "__main__":
    unittest.main()
