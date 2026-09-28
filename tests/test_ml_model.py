"""Tests for the local sklearn ML extractor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.ml_model import (
    PathologyMLExtractor,
    ml_extract,
    reset_ml_extractor_cache,
)
from src.schemas import CORE_VARIABLES, DocumentationStatus
from src.workflow import run_extraction_pipeline


class TestPathologyMLExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        reset_ml_extractor_cache()
        cls.model = PathologyMLExtractor().train()

    def test_train_produces_all_variable_pipelines(self) -> None:
        self.assertTrue(self.model.is_ready)
        self.assertEqual(set(self.model.pipelines), set(CORE_VARIABLES))

    def test_save_and_load_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            self.model.save(path)
            loaded = PathologyMLExtractor.load(path)
            self.assertTrue(loaded.is_ready)
            report = (
                "FINAL DIAGNOSIS:\n"
                "- Invasive adenocarcinoma.\n"
                "- Tumor size: 3.2 cm in greatest dimension.\n"
                "- Pathological T category: pT2a.\n"
                "- Pathological N category: pN0."
            )
            original = self.model.predict_fields(report)
            restored = loaded.predict_fields(report)
            self.assertEqual(
                [item.label for item in original],
                [item.label for item in restored],
            )

    def test_ml_extract_anchors_supported_values(self) -> None:
        report = (
            "FINAL DIAGNOSIS:\n"
            "- Invasive adenocarcinoma.\n"
            "- Tumor size: 3.2 cm in greatest dimension.\n"
            "- Pathological T category: pT2a.\n"
            "- Pathological N category: pN0."
        )
        result = ml_extract(report, "ML-TEST-001", model=self.model)
        self.assertEqual(result.method, "ml")
        self.assertEqual(len(result.variables), len(CORE_VARIABLES))
        by_name = {item.variable_name: item for item in result.variables}
        self.assertEqual(by_name["histologic_diagnosis"].extracted_value, "Invasive adenocarcinoma")
        self.assertEqual(by_name["tumor_size"].extracted_value, "3.2 cm")
        self.assertEqual(by_name["pathological_t_category"].extracted_value, "pT2a")
        self.assertEqual(by_name["pathological_n_category"].extracted_value, "pN0")
        for item in result.variables:
            if item.extracted_value is not None:
                self.assertTrue(item.evidence)
                for span in item.evidence or []:
                    self.assertEqual(report[span.start_offset:span.end_offset], span.text)

    def test_ml_never_returns_unanchored_invented_value(self) -> None:
        report = "This note mentions no pathology categories at all."
        result = ml_extract(report, "ML-TEST-002", model=self.model)
        for item in result.variables:
            if item.extracted_value is not None:
                self.assertTrue(item.evidence)
                for span in item.evidence or []:
                    self.assertIn(span.text.casefold(), report.casefold())
            else:
                self.assertIn(
                    item.documentation_status,
                    {
                        DocumentationStatus.NOT_DOCUMENTED,
                        DocumentationStatus.CANNOT_BE_ASSIGNED,
                        DocumentationStatus.NEGATED,
                        DocumentationStatus.CONFLICTING,
                        DocumentationStatus.UNSUPPORTED,
                        DocumentationStatus.MANUAL_REVIEW_REQUIRED,
                    },
                )

    def test_ml_extracts_unseen_tcga_synoptic_values_from_exact_evidence(self) -> None:
        report = (
            "FINAL DIAGNOSIS\n"
            "Histologic type: Adenocarcinoma, acinar predominant\n"
            "Tumor size\nGreatest dimension of tumor: 6.7 cm\n"
            "Pathologic staging (pTNM)\n"
            "Primary tumor (pT): pT4\n"
            "Regional lymph nodes (pN): pN2\n"
        )

        result = ml_extract(report, "TCGA-SYNOPTIC-001", model=self.model)
        by_name = {item.variable_name: item for item in result.variables}

        self.assertEqual(
            by_name["histologic_diagnosis"].extracted_value,
            "Adenocarcinoma, acinar predominant",
        )
        self.assertEqual(by_name["tumor_size"].extracted_value, "6.7 cm")
        self.assertEqual(by_name["pathological_t_category"].extracted_value, "pT4")
        self.assertEqual(by_name["pathological_n_category"].extracted_value, "pN2")
        for item in result.variables:
            self.assertEqual(item.documentation_status, DocumentationStatus.SUPPORTED)
            self.assertTrue(item.evidence)
            for span in item.evidence or []:
                self.assertEqual(report[span.start_offset:span.end_offset], span.text)

        workflow_result = run_extraction_pipeline(
            {
                "report_id": "TCGA-SYNOPTIC-001",
                "text": report,
                "cancer_type": "LUAD",
            },
            "ml",
        )
        workflow_fields = {
            item.variable_name: item for item in workflow_result.variables
        }
        self.assertEqual(workflow_fields["tumor_size"].extracted_value, "6.7 cm")
        self.assertEqual(
            workflow_fields["tumor_size"].documentation_status,
            DocumentationStatus.SUPPORTED,
        )

    def test_workflow_ml_method(self) -> None:
        report = {
            "report_id": "ML-TEST-003",
            "text": (
                "FINAL DIAGNOSIS:\n"
                "Invasive squamous cell carcinoma. "
                "Tumor size: 5.5 cm in greatest dimension. "
                "Pathological T category: pT3. "
                "Pathological N category: pNX."
            ),
            "cancer_type": "LUSC",
        }
        result = run_extraction_pipeline(report, "ml")
        self.assertEqual(result.method, "ml")
        self.assertEqual(result.cancer_type, "LUSC")
        pn = next(item for item in result.variables if item.variable_name == "pathological_n_category")
        if pn.extracted_value == "pNX":
            self.assertEqual(pn.documentation_status, DocumentationStatus.CANNOT_BE_ASSIGNED)


if __name__ == "__main__":
    unittest.main()
