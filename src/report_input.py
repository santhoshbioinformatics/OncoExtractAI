"""Safe ingestion helpers for synthetic or approved pathology text.

The prototype never attempts to de-identify clinical text.  Instead, it rejects
obvious direct-identifier labels and requires the reviewer to confirm that
pasted or uploaded content is already synthetic or approved for local use.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Literal


MAX_REPORT_BYTES = 512_000
MAX_REPORT_CHARS = 100_000
MIN_REPORT_CHARS = 20
ALLOWED_UPLOAD_SUFFIXES = {".txt"}
REDACTED_VALUES = {
    "",
    "[redacted]",
    "redacted",
    "[removed]",
    "removed",
    "[synthetic]",
    "synthetic",
    "n/a",
    "none",
}


class ReportInputError(ValueError):
    """Raised when report text is unsafe or cannot be parsed."""


def sanitize_report_text(raw_text: str) -> str:
    """Normalize report text while preserving readable line structure.

    Extraction offsets are calculated only after this function runs, so the
    sanitized text shown to the reviewer is also the canonical source text.
    """

    if not isinstance(raw_text, str):
        raise ReportInputError("Report text must be plain text.")

    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        character
        for character in text
        if character in {"\n", "\t"} or unicodedata.category(character) != "Cc"
    )
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()

    if len(text) < MIN_REPORT_CHARS:
        raise ReportInputError(
            f"Report text must contain at least {MIN_REPORT_CHARS} characters."
        )
    if len(text) > MAX_REPORT_CHARS:
        raise ReportInputError(
            f"Report text exceeds the {MAX_REPORT_CHARS:,}-character local limit."
        )
    return text


def find_direct_identifier_labels(text: str) -> list[str]:
    """Return obvious direct-identifier labels with non-redacted values.

    This deliberately small safeguard is not a PHI detector and the UI says so.
    It prevents the most common accidental paste of names, record numbers, and
    birth dates into a research demonstration.
    """

    patterns = {
        "patient name": r"(?im)^\s*patient\s+name\s*:\s*([^\n]*)$",
        "medical record number": r"(?im)^\s*(?:medical\s+record\s+number|mrn)\s*:\s*([^\n]*)$",
        "date of birth": r"(?im)^\s*(?:date\s+of\s+birth|dob)\s*:\s*([^\n]*)$",
    }
    findings: list[str] = []
    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            value = match.group(1).strip().lower()
            if value not in REDACTED_VALUES:
                findings.append(label)
                break
    return findings


def decode_uploaded_text(filename: str, payload: bytes) -> str:
    """Decode and sanitize a UTF-8 ``.txt`` upload."""

    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ReportInputError("Upload a UTF-8 plain-text file with a .txt extension.")
    if not payload:
        raise ReportInputError("The uploaded file is empty.")
    if len(payload) > MAX_REPORT_BYTES:
        raise ReportInputError(
            f"The uploaded file exceeds the {MAX_REPORT_BYTES // 1024} KB local limit."
        )
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ReportInputError("The uploaded file must be valid UTF-8 text.") from error
    return sanitize_report_text(decoded)


def prepare_report(
    raw_text: str,
    *,
    source: Literal["synthetic", "pasted", "uploaded"],
    approved: bool,
    report_id: str | None = None,
    description: str | None = None,
    cancer_type: str | None = None,
) -> dict[str, Any]:
    """Validate content and return the app's canonical in-memory report shape."""

    if source not in {"synthetic", "pasted", "uploaded"}:
        raise ReportInputError("Unknown report source.")
    if source != "synthetic" and approved is not True:
        raise ReportInputError(
            "Confirm that the report is synthetic, de-identified, or approved for local research use."
        )

    text = sanitize_report_text(raw_text)
    identifiers = find_direct_identifier_labels(text)
    if identifiers:
        labels = ", ".join(sorted(identifiers))
        raise ReportInputError(
            f"Remove or redact direct identifiers before loading this report ({labels})."
        )

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10].upper()
    resolved_id = report_id or f"LOCAL-{digest}"
    return {
        "report_id": resolved_id,
        "cancer_type": cancer_type,
        "description": description or "Approved local report",
        "text": text,
        "source": source,
        "is_synthetic": source == "synthetic",
        "approved_for_local_use": source == "synthetic" or approved is True,
    }
