"""Tests for exact-offset evidence rendering and canonical labels."""

from __future__ import annotations

from src.display import evidence_anchor, render_highlighted_report, status_presentation
from src.schemas import DocumentationStatus, EvidenceSpan, VariableExtraction


def _variable(name: str, value: str, text: str, report: str) -> VariableExtraction:
    start = report.index(text)
    return VariableExtraction(
        variable_name=name,
        extracted_value=value,
        documentation_status=DocumentationStatus.SUPPORTED,
        evidence=[
            EvidenceSpan(text=text, start_offset=start, end_offset=start + len(text))
        ],
    )


def test_highlighter_uses_offsets_for_repeated_and_multiline_evidence() -> None:
    report = "Header <unsafe>\nTumor 3.2 cm\nTumor 3.2 cm"
    evidence_text = "Tumor 3.2 cm\nTumor 3.2 cm"
    variable = _variable("tumor_size", "3.2 cm", evidence_text, report)

    rendered = render_highlighted_report(report, [variable], "tumor_size")

    assert "&lt;unsafe&gt;" in rendered
    assert evidence_anchor("tumor_size") in rendered
    assert "evidence-mark--focused" in rendered
    assert rendered.count("evidence-mark") >= 1


def test_invalid_offset_span_is_not_highlighted() -> None:
    # Construct a valid model, then use model_copy to emulate malformed external data.
    report = "Final diagnosis: invasive adenocarcinoma"
    valid = _variable(
        "histologic_diagnosis",
        "Invasive adenocarcinoma",
        "invasive adenocarcinoma",
        report,
    )
    invalid_evidence = valid.evidence[0].model_copy(update={"start_offset": 0})
    malformed = valid.model_copy(update={"evidence": [invalid_evidence]})

    rendered = render_highlighted_report(report, [malformed])
    assert "<mark" not in rendered


def test_pn_status_labels_remain_distinct() -> None:
    labels = {
        status_presentation(DocumentationStatus.SUPPORTED).label,
        status_presentation(DocumentationStatus.CANNOT_BE_ASSIGNED).label,
        status_presentation(DocumentationStatus.NOT_DOCUMENTED).label,
    }
    assert labels == {"Supported", "Cannot be assigned", "Not documented"}

