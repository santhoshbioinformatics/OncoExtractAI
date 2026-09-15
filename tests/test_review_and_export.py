"""Tests for review validation, audit history, and safe exports."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.exporter import audit_history_csv, reviewed_results_csv, reviewed_results_json
from src.review_workflow import (
    create_review_snapshot,
    initial_review_decisions,
    validate_review_decisions,
)
from src.schemas import (
    DocumentationStatus,
    EvidenceSpan,
    ExtractionResult,
    VariableExtraction,
)


REVIEW_REPORT = (
    "Invasive adenocarcinoma; Tumor size: 3.2 cm; pT2a; pN0. "
    "Tumor size is approximately 4.0 cm due to fragmented specimen. "
    "Gross section records 3.2 cm tumor. "
    "Microscopic section records 4.0 cm tumor. Correction token: =1+1."
)


def _supported_variable(name: str, value: str, evidence_text: str, report: str) -> VariableExtraction:
    start = report.index(evidence_text)
    return VariableExtraction(
        variable_name=name,
        extracted_value=value,
        documentation_status=DocumentationStatus.SUPPORTED,
        evidence=[
            EvidenceSpan(
                text=evidence_text,
                start_offset=start,
                end_offset=start + len(evidence_text),
            )
        ],
    )


def _result() -> ExtractionResult:
    return ExtractionResult(
        report_id="SYN-REVIEW-001",
        method="evidence_first",
        variables=[
            _supported_variable(
                "histologic_diagnosis",
                "Invasive adenocarcinoma",
                "Invasive adenocarcinoma",
                REVIEW_REPORT,
            ),
            _supported_variable(
                "tumor_size", "3.2 cm", "Tumor size: 3.2 cm", REVIEW_REPORT
            ),
            _supported_variable("pathological_t_category", "pT2a", "pT2a", REVIEW_REPORT),
            _supported_variable("pathological_n_category", "pN0", "pN0", REVIEW_REPORT),
        ],
        review_priority="low",
        review_priority_reason="All fields are explicitly supported.",
    )


def test_incomplete_review_and_unexplained_correction_are_rejected() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    assert len(validate_review_decisions(result, decisions)) == 4

    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="4.0 cm",
        corrected_status="supported",
        corrected_evidence_texts=["4.0 cm"],
        reason="",
    )
    errors = validate_review_decisions(result, decisions, REVIEW_REPORT)
    assert any("explain" in error for error in errors)


def test_reviewer_correction_creates_snapshot_and_append_only_audit_record() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="4.0 cm",
        corrected_status="uncertain",
        corrected_evidence_texts=[
            "Tumor size is approximately 4.0 cm due to fragmented specimen"
        ],
        reason="Fragmented specimen; reviewer selected the documented estimate.",
    )

    snapshot, audit = create_review_snapshot(
        result,
        decisions,
        report_text=REVIEW_REPORT,
        timestamp="2026-09-10T12:00:00+00:00",
    )
    corrected = next(field for field in snapshot.fields if field.variable_name == "tumor_size")
    corrected_event = next(item for item in audit if item.variable_name == "tumor_size")

    assert corrected.reviewed_value == "4.0 cm"
    assert corrected.reviewed_status == DocumentationStatus.UNCERTAIN
    assert corrected.evidence[0].text == (
        "Tumor size is approximately 4.0 cm due to fragmented specimen"
    )
    assert corrected_event.original_output.extracted_value == "3.2 cm"
    assert corrected_event.corrected_value == "4.0 cm"
    assert "4.0 cm" in corrected_event.resulting_evidence[0].text
    assert corrected_event.timestamp == "2026-09-10T12:00:00+00:00"

    decisions["tumor_size"]["corrected_value"] = "changed later"
    assert corrected_event.corrected_value == "4.0 cm"


def test_exports_include_original_output_and_neutralize_formulas() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["histologic_diagnosis"].update(
        action="corrected",
        corrected_value="=1+1",
        corrected_status="supported",
        corrected_evidence_texts=["Correction token: =1+1"],
        reason="@external formula-like reviewer text",
    )
    snapshot, audit = create_review_snapshot(
        result, decisions, report_text=REVIEW_REPORT
    )

    payload = json.loads(reviewed_results_json([snapshot], audit))
    assert payload["metadata"]["label"].startswith("Illustrative")
    assert payload["audit_history"][0]["original_output"]

    reviewed_rows = list(csv.DictReader(io.StringIO(reviewed_results_csv([snapshot]))))
    diagnosis_row = next(
        row for row in reviewed_rows if row["variable_name"] == "histologic_diagnosis"
    )
    assert diagnosis_row["reviewed_value"].startswith("'")
    assert diagnosis_row["review_reason"].startswith("'")

    audit_rows = list(csv.DictReader(io.StringIO(audit_history_csv(audit))))
    diagnosis_event = next(
        row for row in audit_rows if row["variable_name"] == "histologic_diagnosis"
    )
    assert diagnosis_event["original_value"] == "Invasive adenocarcinoma"
    assert diagnosis_event["original_evidence_json"]


def test_correction_requires_nonempty_value() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value=" ",
        corrected_status="supported",
        corrected_evidence_texts=["4.0 cm"],
        reason="Needs correction.",
    )
    with pytest.raises(ValueError, match="corrected value"):
        create_review_snapshot(result, decisions, report_text=REVIEW_REPORT)


def test_correction_evidence_must_state_value_and_identify_one_exact_span() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="4.0 cm",
        corrected_status="supported",
        corrected_evidence_texts=["Tumor size: 3.2 cm"],
        reason="Reviewer correction.",
    )
    errors = validate_review_decisions(result, decisions, REVIEW_REPORT)
    assert any("does not state" in error for error in errors)

    decisions["tumor_size"]["corrected_evidence_texts"] = ["cm"]
    errors = validate_review_decisions(result, decisions, REVIEW_REPORT)
    assert any("occurs" in error for error in errors)


def test_conflicting_correction_preserves_two_exact_alternative_spans() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="",
        corrected_status="conflicting",
        conflicting_alternatives=["3.2 cm", "4.0 cm"],
        corrected_evidence_texts=[
            "Gross section records 3.2 cm tumor",
            "Microscopic section records 4.0 cm tumor",
        ],
        reason="Two explicit measurements remain unresolved.",
    )

    snapshot, audit = create_review_snapshot(
        result, decisions, report_text=REVIEW_REPORT
    )
    field = next(item for item in snapshot.fields if item.variable_name == "tumor_size")
    event = next(item for item in audit if item.variable_name == "tumor_size")
    assert field.reviewed_value is None
    assert field.reviewed_status == DocumentationStatus.CONFLICTING
    assert field.reviewed_alternatives == ["3.2 cm", "4.0 cm"]
    assert [span.text for span in field.evidence] == [
        "Gross section records 3.2 cm tumor",
        "Microscopic section records 4.0 cm tumor",
    ]
    assert len(event.resulting_evidence) == 2
    assert event.resulting_alternatives == ["3.2 cm", "4.0 cm"]
    assert snapshot.original_review_priority == "low"
    assert snapshot.review_priority == "high"
    assert "unresolved" in snapshot.review_priority_reason

    reviewed_row = next(
        row for row in csv.DictReader(io.StringIO(reviewed_results_csv([snapshot])))
        if row["variable_name"] == "tumor_size"
    )
    audit_row = next(
        row for row in csv.DictReader(io.StringIO(audit_history_csv(audit)))
        if row["variable_name"] == "tumor_size"
    )
    assert json.loads(reviewed_row["reviewed_alternatives_json"]) == ["3.2 cm", "4.0 cm"]
    assert json.loads(audit_row["resulting_alternatives_json"]) == ["3.2 cm", "4.0 cm"]
    assert reviewed_row["original_review_priority"] == "low"
    assert reviewed_row["residual_review_priority"] == "high"


def test_conflicting_correction_rejects_wrong_or_duplicate_alternatives() -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="",
        corrected_status="conflicting",
        conflicting_alternatives=["3.2 cm", "4.0 cm"],
        corrected_evidence_texts=["3.2 cm", "pT2a"],
        reason="Testing mismatched alternatives.",
    )
    errors = validate_review_decisions(result, decisions, REVIEW_REPORT)
    assert any("conflict evidence 2 does not state" in error for error in errors)

    decisions["tumor_size"].update(
        conflicting_alternatives=["3.2 cm", "3.2 cm"],
        corrected_evidence_texts=["3.2 cm", "4.0 cm"],
    )
    errors = validate_review_decisions(result, decisions, REVIEW_REPORT)
    assert any("alternative values must be distinct" in error for error in errors)


@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        ("rejected", DocumentationStatus.UNSUPPORTED),
        ("flagged_for_review", DocumentationStatus.MANUAL_REVIEW_REQUIRED),
    ],
)
def test_reject_or_flag_recomputes_residual_priority(
    action: str, expected_status: DocumentationStatus
) -> None:
    result = _result()
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action=action,
        reason="Reviewer cannot retain this proposed value.",
    )

    snapshot, _ = create_review_snapshot(result, decisions, report_text=REVIEW_REPORT)
    field = next(item for item in snapshot.fields if item.variable_name == "tumor_size")
    assert field.reviewed_status == expected_status
    assert snapshot.original_review_priority == "low"
    assert snapshot.review_priority == "high"


def test_resolved_original_conflict_recomputes_priority_to_low() -> None:
    base = _result()
    variables = [item.model_copy(deep=True) for item in base.variables]
    variables[1] = VariableExtraction(
        variable_name="tumor_size",
        extracted_value=None,
        documentation_status="conflicting",
        evidence=[
            _supported_variable(
                "tumor_size",
                "3.2 cm",
                "Gross section records 3.2 cm tumor",
                REVIEW_REPORT,
            ).evidence[0],
            _supported_variable(
                "tumor_size",
                "4.0 cm",
                "Microscopic section records 4.0 cm tumor",
                REVIEW_REPORT,
            ).evidence[0],
        ],
        alternatives=["3.2 cm", "4.0 cm"],
    )
    result = ExtractionResult(
        report_id=base.report_id,
        method=base.method,
        variables=variables,
        manual_review_required=True,
        review_priority="high",
        review_priority_reason="Two explicit tumor sizes remain unresolved.",
    )
    decisions = initial_review_decisions(result)
    for decision in decisions.values():
        decision["action"] = "accepted"
    decisions["tumor_size"].update(
        action="corrected",
        corrected_value="4.0 cm",
        corrected_status="supported",
        corrected_evidence_texts=["Microscopic section records 4.0 cm tumor"],
        reason="Reviewer resolved the conflict to the amended measurement.",
    )

    snapshot, _ = create_review_snapshot(result, decisions, report_text=REVIEW_REPORT)
    assert snapshot.original_review_priority == "high"
    assert snapshot.review_priority == "low"
    assert "all four fields" in snapshot.review_priority_reason
