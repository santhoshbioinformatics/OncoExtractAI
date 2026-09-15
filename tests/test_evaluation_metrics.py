"""Tests for transparent synthetic evaluation denominators."""

from __future__ import annotations

import csv
import io

import pytest

from src.evaluation import Evaluator
from src.exporter import evaluation_comparison_csv
from src.schemas import (
    ComparisonResult,
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    GoldAnnotation,
    VariableExtraction,
)


REPORT = "Diagnosis: Invasive adenocarcinoma. Tumor size: 3.2 cm. pT2a. pN0."


def _span(text: str) -> EvidenceSpan:
    start = REPORT.index(text)
    return EvidenceSpan(text=text, start_offset=start, end_offset=start + len(text))


def _gold(name: str, value: str, evidence: str) -> GoldAnnotation:
    span = _span(evidence)
    return GoldAnnotation(
        report_id="SYN-EVAL-001",
        cancer_type="LUAD",
        variable_name=name,
        gold_value=value,
        evidence_text=evidence,
        evidence_start_offset=span.start_offset,
        evidence_end_offset=span.end_offset,
        documentation_status=DocumentationStatus.SUPPORTED,
    )


def test_wrong_value_counts_as_false_positive_and_false_negative() -> None:
    extraction = ExtractionResult(
        report_id="SYN-EVAL-001",
        cancer_type="LUAD",
        method="baseline",
        variables=[
            VariableExtraction(
                variable_name="histologic_diagnosis",
                extracted_value="Invasive adenocarcinoma",
                documentation_status="supported",
                evidence=[_span("Invasive adenocarcinoma")],
            ),
            VariableExtraction(
                variable_name="tumor_size",
                extracted_value="4.0 cm",
                documentation_status="supported",
                evidence=[_span("3.2 cm")],
            ),
            VariableExtraction(
                variable_name="pathological_t_category",
                extracted_value="pT2a",
                documentation_status="supported",
                evidence=[_span("pT2a")],
            ),
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value=None,
                documentation_status="not_documented",
            ),
        ],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="One required field is not documented.",
    )
    gold = [
        _gold("histologic_diagnosis", "Invasive adenocarcinoma", "Invasive adenocarcinoma"),
        _gold("tumor_size", "3.2 cm", "Tumor size: 3.2 cm"),
        _gold("pathological_t_category", "pT2a", "pT2a"),
        _gold("pathological_n_category", "pN0", "pN0"),
    ]

    metrics = Evaluator().compute_overall_metrics(
        [extraction], gold, {"SYN-EVAL-001": REPORT}
    )

    assert metrics.overall_precision == pytest.approx(2 / 3, abs=0.0001)
    assert metrics.overall_recall == pytest.approx(1 / 2, abs=0.0001)
    assert metrics.unsupported_extraction_rate == pytest.approx(1 / 3, abs=0.0001)
    assert metrics.coverage == pytest.approx(3 / 4, abs=0.0001)
    assert metrics.documentation_status_accuracy == pytest.approx(3 / 4, abs=0.0001)
    assert metrics.documentation_status_correct_count == 3
    assert metrics.exact_evidence_rate == pytest.approx(2 / 3, abs=0.0001)
    tumor = next(item for item in metrics.variable_metrics if item.variable_name == "tumor_size")
    assert tumor.false_positives == 1
    assert tumor.false_negatives == 1
    assert tumor.unsupported_count == 1


def test_appropriate_abstention_uses_only_fields_expected_to_abstain() -> None:
    extraction = ExtractionResult(
        report_id="SYN-EVAL-002",
        method="evidence_first",
        variables=[
            VariableExtraction(
                variable_name="histologic_diagnosis",
                extracted_value=None,
                documentation_status="not_documented",
            ),
            VariableExtraction(
                variable_name="tumor_size",
                extracted_value=None,
                documentation_status="not_documented",
            ),
            VariableExtraction(
                variable_name="pathological_t_category",
                extracted_value=None,
                documentation_status="not_documented",
            ),
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value=None,
                documentation_status="not_documented",
            ),
        ],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="All four fields are not documented.",
    )
    gold = [
        GoldAnnotation(
            report_id="SYN-EVAL-002",
            variable_name=name,
            gold_value=None,
            documentation_status=DocumentationStatus.NOT_DOCUMENTED,
            manual_review_required=True,
        )
        for name in (
            "histologic_diagnosis",
            "tumor_size",
            "pathological_t_category",
            "pathological_n_category",
        )
    ]

    metrics = Evaluator().compute_overall_metrics(
        [extraction], gold, {"SYN-EVAL-002": "No target variables are documented."}
    )
    assert metrics.appropriate_abstention_rate == 1.0
    assert metrics.false_positive_rate_undocumented == 0.0
    assert metrics.unsupported_extraction_rate is None
    assert metrics.overall_precision is None
    assert metrics.overall_recall is None
    assert metrics.overall_f1 is None
    assert metrics.exact_evidence_rate is None
    assert metrics.manual_review_referral_precision == 1.0
    assert metrics.manual_review_referral_recall == 1.0
    assert all(item.precision is None for item in metrics.variable_metrics)
    assert all(item.recall is None for item in metrics.variable_metrics)

    comparison = ComparisonResult(
        baseline_metrics=metrics.model_copy(update={"method": "baseline"}),
        evidence_first_metrics=metrics,
    )
    rows = list(csv.DictReader(io.StringIO(evaluation_comparison_csv(comparison))))
    assert {row["export_label"] for row in rows} == {"Illustrative synthetic results"}
    undefined = [row for row in rows if row["metric"] == "overall_precision"]
    assert len(undefined) == 2
    assert all(row["denominator"] == "0" and row["rate"] == "N/A" for row in undefined)


def test_evidence_value_matching_respects_numeric_and_stage_boundaries() -> None:
    report = "Tumor size: 13 cm. Pathologic category: pT2a."
    evaluator = Evaluator()

    tumor_text = "13 cm"
    tumor_start = report.index(tumor_text)
    tumor = VariableExtraction(
        variable_name="tumor_size",
        extracted_value="3 cm",
        documentation_status="supported",
        evidence=[
            EvidenceSpan(
                text=tumor_text,
                start_offset=tumor_start,
                end_offset=tumor_start + len(tumor_text),
            )
        ],
    )
    stage_text = "pT2a"
    stage_start = report.index(stage_text)
    stage = VariableExtraction(
        variable_name="pathological_t_category",
        extracted_value="pT2",
        documentation_status="supported",
        evidence=[
            EvidenceSpan(
                text=stage_text,
                start_offset=stage_start,
                end_offset=stage_start + len(stage_text),
            )
        ],
    )

    assert not evaluator._evidence_supports_value(tumor, report)
    assert not evaluator._evidence_supports_value(stage, report)


def test_every_supplied_evidence_span_must_be_exact() -> None:
    variable = VariableExtraction(
        variable_name="histologic_diagnosis",
        extracted_value="Invasive adenocarcinoma",
        documentation_status="supported",
        evidence=[
            _span("Invasive adenocarcinoma"),
            EvidenceSpan(text="Other", start_offset=0, end_offset=5),
        ],
    )

    assert not Evaluator()._evidence_supports_value(variable, REPORT)


def test_evaluation_rejects_empty_or_incomplete_cohorts() -> None:
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="at least one"):
        evaluator.compute_overall_metrics([], [], {})

    result = ExtractionResult(
        report_id="SYN-EVAL-001",
        cancer_type="LUAD",
        method="baseline",
        variables=[
            VariableExtraction(
                variable_name=name,
                extracted_value=None,
                documentation_status="not_documented",
            )
            for name in (
                "histologic_diagnosis",
                "tumor_size",
                "pathological_t_category",
                "pathological_n_category",
            )
        ],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="All four fields are not documented.",
    )
    incomplete_gold = [
        GoldAnnotation(
            report_id="SYN-EVAL-001",
            variable_name="histologic_diagnosis",
            gold_value=None,
            documentation_status="not_documented",
            manual_review_required=True,
        )
    ]
    with pytest.raises(ValueError, match="exactly four"):
        evaluator.compute_overall_metrics(
            [result], incomplete_gold, {"SYN-EVAL-001": REPORT}
        )


def test_single_report_evidence_requires_full_integrity_and_gold_containment() -> None:
    result = ExtractionResult(
        report_id="SYN-EVAL-001",
        cancer_type="LUAD",
        method="baseline",
        variables=[
            VariableExtraction(
                variable_name="histologic_diagnosis",
                extracted_value="Invasive adenocarcinoma",
                documentation_status="supported",
                evidence=[
                    _span("Invasive adenocarcinoma"),
                    EvidenceSpan(text="Other", start_offset=0, end_offset=5),
                ],
            ),
            VariableExtraction(
                variable_name="tumor_size",
                extracted_value="3.2 cm",
                documentation_status="supported",
                evidence=[_span("Tumor size: 3.2 cm")],
            ),
            VariableExtraction(
                variable_name="pathological_t_category",
                extracted_value="pT2a",
                documentation_status="supported",
                evidence=[_span("pT2a")],
            ),
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN0",
                documentation_status="supported",
                evidence=[_span("pN0")],
            ),
        ],
    )
    gold = [
        _gold("histologic_diagnosis", "Invasive adenocarcinoma", "Invasive adenocarcinoma"),
        _gold("tumor_size", "3.2 cm", "Tumor size: 3.2 cm"),
        _gold("pathological_t_category", "pT2a", "pT2a"),
        _gold("pathological_n_category", "pN0", "pN0"),
    ]

    single = Evaluator().evaluate_single_report(result, gold, REPORT)
    assert single["variables"]["histologic_diagnosis"]["evidence_valid"] is False
    assert single["variables"]["tumor_size"]["evidence_valid"] is True


def test_single_report_negated_evidence_is_scored_without_requiring_a_value() -> None:
    report = "No invasive adenocarcinoma is identified. Tumor size: 3.2 cm. pT2a. pN0."

    def local_span(text: str) -> EvidenceSpan:
        start = report.index(text)
        return EvidenceSpan(text=text, start_offset=start, end_offset=start + len(text))

    negation = "No invasive adenocarcinoma is identified."
    result = ExtractionResult(
        report_id="SYN-EVAL-NEG",
        method="evidence_first",
        variables=[
            VariableExtraction(
                variable_name="histologic_diagnosis",
                extracted_value=None,
                documentation_status="negated",
                evidence=[local_span(negation)],
            ),
            VariableExtraction(
                variable_name="tumor_size",
                extracted_value="3.2 cm",
                documentation_status="supported",
                evidence=[local_span("3.2 cm")],
            ),
            VariableExtraction(
                variable_name="pathological_t_category",
                extracted_value="pT2a",
                documentation_status="supported",
                evidence=[local_span("pT2a")],
            ),
            VariableExtraction(
                variable_name="pathological_n_category",
                extracted_value="pN0",
                documentation_status="supported",
                evidence=[local_span("pN0")],
            ),
        ],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="A diagnosis candidate is explicitly negated.",
    )
    gold = [
        GoldAnnotation(
            report_id="SYN-EVAL-NEG",
            variable_name="histologic_diagnosis",
            gold_value=None,
            evidence_text=negation,
            evidence_start_offset=local_span(negation).start_offset,
            evidence_end_offset=local_span(negation).end_offset,
            documentation_status="negated",
            manual_review_required=True,
        ),
        *[
            GoldAnnotation(
                report_id="SYN-EVAL-NEG",
                variable_name=name,
                gold_value=value,
                evidence_text=evidence_text,
                evidence_start_offset=local_span(evidence_text).start_offset,
                evidence_end_offset=local_span(evidence_text).end_offset,
                documentation_status="supported",
            )
            for name, value, evidence_text in (
                ("tumor_size", "3.2 cm", "Tumor size: 3.2 cm"),
                ("pathological_t_category", "pT2a", "pT2a"),
                ("pathological_n_category", "pN0", "pN0"),
            )
        ],
    ]

    single = Evaluator().evaluate_single_report(result, gold, report)
    assert single["variables"]["histologic_diagnosis"]["evidence_valid"] is True


def test_tumor_value_comparison_uses_exact_decimal_equality() -> None:
    evaluator = Evaluator()
    assert evaluator._values_match("3.20 cm", "3.2 cm")
    assert not evaluator._values_match("3.20 cm", "3.25 cm")


def test_margin_measurement_is_not_tumor_evidence() -> None:
    report = "Closest surgical margin measures 0.2 cm."
    evidence_text = "margin measures 0.2 cm"
    start = report.index(evidence_text)
    variable = VariableExtraction(
        variable_name="tumor_size",
        extracted_value="0.2 cm",
        documentation_status="supported",
        evidence=[
            EvidenceSpan(
                text=evidence_text,
                start_offset=start,
                end_offset=start + len(evidence_text),
            )
        ],
    )
    assert not Evaluator()._evidence_supports_value(variable, report)


def test_evaluation_rejects_gold_offset_and_cancer_type_mismatches() -> None:
    report_id = "SYN-EVAL-COHORT"
    result = ExtractionResult(
        report_id=report_id,
        cancer_type="LUAD",
        method="baseline",
        variables=[
            VariableExtraction(
                variable_name=name,
                extracted_value=None,
                documentation_status="not_documented",
            )
            for name in (
                "histologic_diagnosis",
                "tumor_size",
                "pathological_t_category",
                "pathological_n_category",
            )
        ],
        manual_review_required=True,
        review_priority="medium",
        review_priority_reason="Fixture intentionally abstains.",
    )
    gold = [
        GoldAnnotation(
            report_id=report_id,
            cancer_type="LUSC",
            variable_name=name,
            gold_value=None,
            documentation_status="not_documented",
            manual_review_required=True,
        )
        for name in (
            "histologic_diagnosis",
            "tumor_size",
            "pathological_t_category",
            "pathological_n_category",
        )
    ]
    with pytest.raises(ValueError, match="cancer_type"):
        Evaluator().compute_overall_metrics([result], gold, {report_id: REPORT})

    correct_span = _span("Invasive adenocarcinoma")
    supported = GoldAnnotation(
        report_id=report_id,
        cancer_type="LUAD",
        variable_name="histologic_diagnosis",
        gold_value="Invasive adenocarcinoma",
        evidence_text=correct_span.text,
        evidence_start_offset=correct_span.start_offset,
        evidence_end_offset=correct_span.end_offset,
        documentation_status="supported",
    ).model_copy(update={"evidence_start_offset": 0})
    offset_gold = [supported] + [
        item.model_copy(update={"cancer_type": "LUAD"})
        for item in gold
        if item.variable_name != "histologic_diagnosis"
    ]
    with pytest.raises(ValueError, match="Gold evidence offsets"):
        Evaluator().compute_overall_metrics(
            [result], offset_gold, {report_id: REPORT}
        )
