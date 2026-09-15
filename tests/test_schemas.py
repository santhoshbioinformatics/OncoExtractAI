"""Strict structured-output contract tests."""

import unittest

from pydantic import ValidationError

from src.schemas import (
    CORE_VARIABLES,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    GoldAnnotation,
    VariableExtraction,
)


def evidence(text="pN0", start=0):
    return EvidenceSpan(
        text=text,
        start_offset=start,
        end_offset=start + len(text),
    )


def missing_variables():
    return [
        VariableExtraction(
            variable_name=name,
            extracted_value=None,
            documentation_status=DocumentationStatus.NOT_DOCUMENTED,
        )
        for name in CORE_VARIABLES
    ]


class TestStrictSchemas(unittest.TestCase):
    def test_confidence_and_other_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN0",
                documentation_status="supported",
                evidence=[evidence()],
                confidence=0.91,
            )

    def test_non_null_candidate_requires_evidence(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN0",
                documentation_status="supported",
            )

    def test_manual_review_status_abstains_and_cannot_assign_requires_evidence(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="histologic_diagnosis",
                extracted_value="Invasive adenocarcinoma",
                documentation_status="manual_review_required",
                evidence=[evidence("Invasive adenocarcinoma")],
            )
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value=None,
                documentation_status="cannot_be_assigned",
            )

    def test_status_and_value_must_be_coherent(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN0",
                documentation_status="not_documented",
                evidence=[evidence()],
            )
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN1",
                documentation_status="conflicting",
                evidence=[evidence("pN0"), evidence("pN1", 10)],
            )

    def test_conflict_requires_two_evidence_alternatives(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value=None,
                documentation_status="conflicting",
                evidence=[evidence("pN0")],
            )

    def test_evidence_span_length_must_match_offsets(self):
        with self.assertRaises(ValidationError):
            EvidenceSpan(text="pN0", start_offset=4, end_offset=99)

    def test_categories_require_canonical_notation(self):
        with self.assertRaises(ValidationError):
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="N0",
                documentation_status="supported",
                evidence=[evidence("N0")],
            )

        for invalid_t in ("pTmi", "pT0a", "pT3c", "pT4b"):
            with self.subTest(invalid_t=invalid_t), self.assertRaises(ValidationError):
                VariableExtraction(
                    variable_name="pathological_t_category",
                    extracted_value=invalid_t,
                    documentation_status="supported",
                    evidence=[evidence(invalid_t)],
                )

        for invalid_n in ("pN0c", "pN1a", "pN3a"):
            with self.subTest(invalid_n=invalid_n), self.assertRaises(ValidationError):
                VariableExtraction(
                    variable_name="pathological_n_category",
                    extracted_value=invalid_n,
                    documentation_status="supported",
                    evidence=[evidence(invalid_n)],
                )

        accepted = VariableExtraction(
            variable_name="pathological_t_category",
            extracted_value="pT1mi",
            documentation_status="supported",
            evidence=[evidence("pT1mi")],
        )
        self.assertEqual(accepted.extracted_value, "pT1mi")

    def test_tumor_size_requires_canonical_centimeter_units(self):
        with self.assertRaisesRegex(ValidationError, "canonical"):
            VariableExtraction(
                variable_name="tumor_size",
                extracted_value="32 mm",
                documentation_status="supported",
                evidence=[evidence("32 mm")],
            )

    def test_result_requires_exactly_one_of_each_core_variable(self):
        variables = missing_variables()
        variables[-1] = variables[0]
        with self.assertRaises(ValidationError):
            ExtractionResult(
                report_id="TEST",
                method="evidence_first",
                variables=variables,
            )

    def test_high_priority_requires_manual_review(self):
        with self.assertRaises(ValidationError):
            ExtractionResult(
                report_id="TEST",
                method="evidence_first",
                variables=missing_variables(),
                review_priority="high",
                manual_review_required=False,
            )

    def test_gold_evidence_coordinates_are_all_or_nothing(self):
        with self.assertRaises(ValidationError):
            GoldAnnotation(
                report_id="TEST",
                variable_name="tumor_size",
                documentation_status="supported",
                evidence_text="3.2 cm",
                evidence_start_offset=5,
            )

    def test_gold_status_value_stage_and_review_contracts_are_coherent(self):
        with self.assertRaises(ValidationError):
            GoldAnnotation(
                report_id="TEST",
                variable_name="pathological_t_category",
                gold_value="pT9",
                evidence_text="pT9",
                evidence_start_offset=0,
                evidence_end_offset=3,
                documentation_status="supported",
            )
        with self.assertRaises(ValidationError):
            GoldAnnotation(
                report_id="TEST",
                variable_name="histologic_diagnosis",
                gold_value="Adenocarcinoma",
                evidence_text="No adenocarcinoma",
                evidence_start_offset=0,
                evidence_end_offset=17,
                documentation_status="negated",
                manual_review_required=True,
            )
        with self.assertRaises(ValidationError):
            GoldAnnotation(
                report_id="TEST",
                variable_name="pathological_n_category",
                gold_value=None,
                documentation_status="cannot_be_assigned",
                manual_review_required=True,
            )
        with self.assertRaises(ValidationError):
            GoldAnnotation(
                report_id="TEST",
                variable_name="pathological_n_category",
                gold_value=None,
                documentation_status="not_documented",
                manual_review_required=False,
            )
        with self.assertRaisesRegex(ValidationError, "does not state"):
            GoldAnnotation(
                report_id="TEST",
                variable_name="pathological_t_category",
                gold_value="pT2a",
                evidence_text="Tumor size: 3.2 cm",
                evidence_start_offset=0,
                evidence_end_offset=18,
                documentation_status="supported",
            )


if __name__ == "__main__":
    unittest.main()
