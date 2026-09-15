"""Tests for safe report ingestion."""

from __future__ import annotations

import pytest

from src.report_input import (
    MAX_REPORT_BYTES,
    ReportInputError,
    decode_uploaded_text,
    find_direct_identifier_labels,
    prepare_report,
    sanitize_report_text,
)


def test_sanitizer_normalizes_line_endings_and_control_characters() -> None:
    text = sanitize_report_text("SYNTHETIC REPORT\r\nTumor\x00 text is documented.  \r\n")
    assert text == "SYNTHETIC REPORT\nTumor text is documented."


@pytest.mark.parametrize("text", ["", "tiny"])
def test_sanitizer_rejects_empty_or_too_short_text(text: str) -> None:
    with pytest.raises(ReportInputError):
        sanitize_report_text(text)


def test_upload_rejects_wrong_type_invalid_utf8_and_oversize() -> None:
    with pytest.raises(ReportInputError):
        decode_uploaded_text("report.pdf", b"plain text that is long enough")
    with pytest.raises(ReportInputError):
        decode_uploaded_text("report.txt", b"\xff\xfe\xfa")
    with pytest.raises(ReportInputError):
        decode_uploaded_text("report.txt", b"a" * (MAX_REPORT_BYTES + 1))


def test_direct_identifier_labels_are_blocked_but_redacted_values_are_allowed() -> None:
    unsafe = "SYNTHETIC REPORT\nPatient Name: Example Person\nFinding: adenocarcinoma"
    assert find_direct_identifier_labels(unsafe) == ["patient name"]
    with pytest.raises(ReportInputError, match="direct identifiers"):
        prepare_report(unsafe, source="pasted", approved=True)

    safe = "SYNTHETIC REPORT\nPatient Name: [REDACTED]\nFinding: adenocarcinoma"
    assert not find_direct_identifier_labels(safe)
    report = prepare_report(safe, source="pasted", approved=True)
    assert report["report_id"].startswith("LOCAL-")


def test_non_synthetic_input_requires_explicit_approval() -> None:
    with pytest.raises(ReportInputError, match="Confirm"):
        prepare_report(
            "APPROVED REPORT\nFinal diagnosis is explicitly documented.",
            source="uploaded",
            approved=False,
        )


@pytest.mark.parametrize("non_boolean_approval", ["false", "true", 1, None])
def test_approval_requires_literal_true(non_boolean_approval: object) -> None:
    with pytest.raises(ReportInputError, match="Confirm"):
        prepare_report(
            "APPROVED REPORT\nFinal diagnosis is explicitly documented.",
            source="pasted",
            approved=non_boolean_approval,  # type: ignore[arg-type]
        )


def test_unknown_runtime_source_is_rejected() -> None:
    with pytest.raises(ReportInputError, match="Unknown report source"):
        prepare_report(
            "APPROVED REPORT\nFinal diagnosis is explicitly documented.",
            source="trusted_magic",  # type: ignore[arg-type]
            approved=True,
        )


def test_report_id_is_stable_for_the_same_sanitized_text() -> None:
    first = prepare_report(
        "SYNTHETIC REPORT\nFinal diagnosis: invasive adenocarcinoma.",
        source="synthetic",
        approved=True,
    )
    second = prepare_report(
        "SYNTHETIC REPORT\r\nFinal diagnosis: invasive adenocarcinoma.\r\n",
        source="synthetic",
        approved=True,
    )
    assert first["report_id"] == second["report_id"]
