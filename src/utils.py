"""
Utility functions for OncoExtractAI-QA.
Handles file I/O, text processing, and common helpers.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ── File I/O ───────────────────────────────────────────────────────────────
def load_synthetic_reports(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load and validate the trusted bundled synthetic report boundary."""
    if path is None:
        path = str(DATA_DIR / "synthetic_reports.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reports = data.get("reports", data) if isinstance(data, dict) else data
    if not isinstance(reports, list):
        raise ValueError("Synthetic report file must contain a 'reports' list.")

    from .report_input import find_direct_identifier_labels

    seen: set[str] = set()
    validated: List[Dict[str, Any]] = []
    watermark = "SYNTHETIC DEMONSTRATION REPORT"
    for index, report in enumerate(reports, start=1):
        if not isinstance(report, dict):
            raise ValueError(f"Synthetic report {index} must be an object.")
        report_id = report.get("report_id")
        text = report.get("text")
        if not isinstance(report_id, str) or not report_id.startswith("SYN-"):
            raise ValueError(f"Synthetic report {index} has an invalid report_id.")
        if report_id in seen:
            raise ValueError(f"Duplicate synthetic report_id: {report_id}.")
        if report.get("is_synthetic") is not True:
            raise ValueError(f"Synthetic report {report_id} is missing is_synthetic=true.")
        if report.get("cancer_type") not in {"LUAD", "LUSC"}:
            raise ValueError(f"Synthetic report {report_id} has an unsupported cancer_type.")
        if not isinstance(text, str) or not text.startswith(watermark):
            raise ValueError(f"Synthetic report {report_id} is missing the visible watermark.")
        if find_direct_identifier_labels(text):
            raise ValueError(f"Synthetic report {report_id} contains direct identifier labels.")
        seen.add(report_id)
        validated.append(report)
    return validated


def load_csv(path: str) -> List[Dict[str, str]]:
    """Load CSV file into list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_iso_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ── Report helpers ──────────────────────────────────────────────────────────
def get_report_text(report: Dict[str, Any]) -> str:
    """Extract text from a report dict."""
    return report.get("text", report.get("report_text", ""))


def get_report_id(report: Dict[str, Any]) -> str:
    """Extract report ID from a report dict."""
    return report.get("report_id", report.get("id", "unknown"))


def load_gold_annotations(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load gold-standard annotations from CSV as raw dicts."""
    if path is None:
        path = str(DATA_DIR / "sample_gold_annotations.csv")
    return load_csv(path)


def parse_gold_annotations(
    rows: Optional[List[Dict[str, Any]]] = None,
    report_texts: Optional[Dict[str, str]] = None,
) -> List[Any]:
    """Parse and fail-fast validate the synthetic reference annotations.

    Invalid statuses, malformed CSV rows, duplicate fields, incomplete report
    sets, and incorrect exact offsets are errors.  Evaluation must never turn a
    broken reference file into apparently valid metrics.
    """
    from .schemas import CORE_VARIABLES, GoldAnnotation, DocumentationStatus

    loaded_default = rows is None
    if rows is None:
        rows = load_gold_annotations()
    if report_texts is None and loaded_default:
        report_texts = {
            get_report_id(report): get_report_text(report)
            for report in load_synthetic_reports()
        }

    required_columns = {
        "report_id",
        "cancer_type",
        "variable_name",
        "gold_value",
        "evidence_text",
        "evidence_start_offset",
        "evidence_end_offset",
        "documentation_status",
        "manual_review_required",
        "notes",
    }
    annotations = []
    seen: set[tuple[str, str]] = set()
    variables_by_report: Dict[str, set[str]] = {}

    for row_number, row in enumerate(rows, start=2):
        if None in row or set(row) != required_columns:
            raise ValueError(f"Malformed gold annotation CSV row {row_number}.")
        status_raw = (row.get("documentation_status") or "").strip().lower()
        try:
            status = DocumentationStatus(status_raw)
        except ValueError as error:
            raise ValueError(
                f"Invalid documentation status on gold annotation row {row_number}: {status_raw!r}."
            ) from error

        def _int_or_none(val):
            if val is None or str(val).strip() == "":
                return None
            normalized = str(val).strip()
            if not normalized.isdigit():
                raise ValueError(
                    f"Evidence offsets must be non-negative integers on row {row_number}."
                )
            return int(normalized)

        def _bool(val):
            normalized = str(val).strip().lower()
            if normalized not in {"true", "false"}:
                raise ValueError(
                    f"manual_review_required must be true or false on row {row_number}."
                )
            return normalized == "true"

        gold_value = row.get("gold_value")
        if gold_value is not None and str(gold_value).strip() == "":
            gold_value = None

        annotation = GoldAnnotation(
                report_id=row.get("report_id", "").strip(),
                cancer_type=(row.get("cancer_type") or None),
                variable_name=row.get("variable_name", "").strip(),
                gold_value=gold_value,
                evidence_text=(row.get("evidence_text") or None),
                evidence_start_offset=_int_or_none(row.get("evidence_start_offset")),
                evidence_end_offset=_int_or_none(row.get("evidence_end_offset")),
                documentation_status=status,
                manual_review_required=_bool(row.get("manual_review_required")),
                notes=(row.get("notes") or None),
            )

        key = (annotation.report_id, annotation.variable_name)
        if key in seen:
            raise ValueError(f"Duplicate gold annotation for {key[0]} / {key[1]}.")
        seen.add(key)
        variables_by_report.setdefault(annotation.report_id, set()).add(
            annotation.variable_name
        )

        if report_texts is not None and annotation.report_id not in report_texts:
            raise ValueError(f"Gold annotation references unknown report {annotation.report_id}.")

        if annotation.documentation_status == DocumentationStatus.SUPPORTED:
            if annotation.gold_value is None or annotation.evidence_text is None:
                raise ValueError(
                    f"Supported gold annotation {key[0]} / {key[1]} requires value and evidence."
                )
        if annotation.documentation_status == DocumentationStatus.NOT_DOCUMENTED:
            if annotation.gold_value is not None:
                raise ValueError(
                    f"Not-documented gold annotation {key[0]} / {key[1]} must abstain."
                )
        if (
            annotation.variable_name == "pathological_n_category"
            and annotation.gold_value is not None
            and annotation.gold_value.upper() == "PNX"
            and annotation.gold_value != "pNX"
        ):
            raise ValueError("Explicit pNX gold values must use canonical spelling 'pNX'.")

        if annotation.evidence_text is not None and report_texts is not None:
            report_text = report_texts.get(annotation.report_id)
            if report_text is None:
                raise ValueError(f"Gold annotation references unknown report {annotation.report_id}.")
            assert annotation.evidence_start_offset is not None
            assert annotation.evidence_end_offset is not None
            actual = report_text[
                annotation.evidence_start_offset:annotation.evidence_end_offset
            ]
            if actual != annotation.evidence_text:
                raise ValueError(
                    f"Gold evidence offsets do not match report text for {key[0]} / {key[1]}."
                )

        annotations.append(annotation)

    expected = set(CORE_VARIABLES)
    for report_id, variables in variables_by_report.items():
        if variables != expected:
            missing = sorted(expected - variables)
            extra = sorted(variables - expected)
            raise ValueError(
                f"Gold annotations for {report_id} must contain all four core fields "
                f"(missing={missing}, extra={extra})."
            )
    if report_texts is not None and set(variables_by_report) != set(report_texts):
        missing_reports = sorted(set(report_texts) - set(variables_by_report))
        extra_reports = sorted(set(variables_by_report) - set(report_texts))
        raise ValueError(
            "Gold annotation report coverage does not match the report set "
            f"(missing={missing_reports}, extra={extra_reports})."
        )
    return annotations
