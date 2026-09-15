#!/usr/bin/env python3
"""Validate the synthetic demonstration reports and their reference annotations."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_PATH = PROJECT_ROOT / "data" / "synthetic_reports.json"
ANNOTATIONS_PATH = PROJECT_ROOT / "data" / "sample_gold_annotations.csv"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "annotation_template.csv"

WATERMARK = "SYNTHETIC DEMONSTRATION REPORT — NO REAL PATIENT DATA"
CORE_VARIABLES = {
    "histologic_diagnosis",
    "tumor_size",
    "pathological_t_category",
    "pathological_n_category",
}
ALLOWED_STATUSES = {
    "supported",
    "not_documented",
    "cannot_be_assigned",
    "negated",
    "uncertain",
    "conflicting",
    "superseded",
    "unsupported",
    "manual_review_required",
}
EXPECTED_SCENARIOS = {
    "complete",
    "no_nodes_cannot_be_assigned",
    "pn_not_documented",
    "explicit_pnx",
    "amended",
    "conflicting",
    "uncertain",
    "negated",
}
CSV_COLUMNS = [
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
]
IDENTIFIER_LABEL = re.compile(
    r"\b(?:patient\s+name|medical\s+record(?:\s+number)?|mrn|date\s+of\s+birth|dob)\b",
    re.IGNORECASE,
)


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        add_error(errors, reader.fieldnames == CSV_COLUMNS, f"{path.name}: unexpected columns")
        rows = list(reader)
    for line_number, row in enumerate(rows, start=2):
        add_error(
            errors,
            None not in row,
            f"{path.name}:{line_number}: extra CSV fields (usually an unquoted comma)",
        )
        add_error(
            errors,
            all(value is not None for value in row.values()),
            f"{path.name}:{line_number}: missing CSV fields",
        )
    return rows


def main() -> int:
    errors: list[str] = []

    payload = json.loads(REPORTS_PATH.read_text(encoding="utf-8"))
    reports_list = payload.get("reports") if isinstance(payload, dict) else None
    add_error(errors, isinstance(reports_list, list), "synthetic_reports.json: 'reports' must be a list")
    if not isinstance(reports_list, list):
        reports_list = []

    add_error(errors, len(reports_list) == 8, "synthetic_reports.json: expected exactly 8 reports")
    reports: dict[str, dict[str, object]] = {}
    scenarios: set[str] = set()
    for index, report in enumerate(reports_list, start=1):
        if not isinstance(report, dict):
            errors.append(f"synthetic_reports.json: report {index} is not an object")
            continue
        report_id = report.get("report_id")
        text = report.get("text")
        scenario = report.get("scenario")
        description = report.get("description")
        add_error(errors, isinstance(report_id, str) and report_id.startswith("SYN-"), f"report {index}: invalid synthetic report_id")
        if not isinstance(report_id, str):
            continue
        add_error(errors, report_id not in reports, f"duplicate report_id: {report_id}")
        reports[report_id] = report
        add_error(errors, report.get("cancer_type") in {"LUAD", "LUSC"}, f"{report_id}: invalid cancer_type")
        add_error(errors, report.get("is_synthetic") is True, f"{report_id}: is_synthetic must be true")
        add_error(errors, isinstance(scenario, str), f"{report_id}: missing scenario")
        if isinstance(scenario, str):
            scenarios.add(scenario)
        add_error(errors, isinstance(description, str) and "synthetic" in description.lower(), f"{report_id}: description must say synthetic")
        add_error(errors, isinstance(text, str) and text.startswith(WATERMARK), f"{report_id}: missing visible synthetic watermark")
        if isinstance(text, str):
            add_error(errors, IDENTIFIER_LABEL.search(text) is None, f"{report_id}: identifier-like label found")

    add_error(errors, scenarios == EXPECTED_SCENARIOS, "synthetic_reports.json: scenario coverage does not match the required set")

    annotations = read_csv(ANNOTATIONS_PATH, errors)
    add_error(errors, len(annotations) == 32, "sample_gold_annotations.csv: expected 32 rows")
    seen_keys: set[tuple[str, str]] = set()
    variables_by_report: dict[str, set[str]] = defaultdict(set)
    annotations_by_scenario: dict[tuple[str, str], dict[str, str]] = {}

    for line_number, row in enumerate(annotations, start=2):
        report_id = row.get("report_id", "")
        variable = row.get("variable_name", "")
        key = (report_id, variable)
        add_error(errors, report_id in reports, f"sample_gold_annotations.csv:{line_number}: unknown report_id")
        add_error(errors, variable in CORE_VARIABLES, f"sample_gold_annotations.csv:{line_number}: invalid variable_name")
        add_error(errors, key not in seen_keys, f"sample_gold_annotations.csv:{line_number}: duplicate report/variable")
        seen_keys.add(key)
        variables_by_report[report_id].add(variable)
        add_error(errors, row.get("documentation_status") in ALLOWED_STATUSES, f"sample_gold_annotations.csv:{line_number}: invalid documentation_status")
        add_error(errors, row.get("manual_review_required") in {"True", "False"}, f"sample_gold_annotations.csv:{line_number}: invalid review boolean")

        report = reports.get(report_id)
        if not report:
            continue
        add_error(errors, row.get("cancer_type") == report.get("cancer_type"), f"sample_gold_annotations.csv:{line_number}: cancer_type mismatch")
        evidence = row.get("evidence_text", "")
        start_raw = row.get("evidence_start_offset", "")
        end_raw = row.get("evidence_end_offset", "")
        if evidence:
            try:
                start = int(start_raw)
                end = int(end_raw)
            except (TypeError, ValueError):
                errors.append(f"sample_gold_annotations.csv:{line_number}: offsets must be integers")
                continue
            report_text = str(report.get("text", ""))
            add_error(errors, 0 <= start < end <= len(report_text), f"sample_gold_annotations.csv:{line_number}: offsets out of bounds")
            if 0 <= start < end <= len(report_text):
                add_error(errors, report_text[start:end] == evidence, f"sample_gold_annotations.csv:{line_number}: evidence/offset mismatch")
        else:
            add_error(
                errors,
                not start_raw and not end_raw,
                f"sample_gold_annotations.csv:{line_number}: empty evidence requires empty offsets",
            )
            add_error(
                errors,
                row.get("documentation_status") == "not_documented",
                f"sample_gold_annotations.csv:{line_number}: only not_documented rows may omit evidence",
            )

        scenario = str(report.get("scenario", ""))
        annotations_by_scenario[(scenario, variable)] = row

    for report_id in reports:
        add_error(errors, variables_by_report[report_id] == CORE_VARIABLES, f"{report_id}: annotations must contain each core variable exactly once")

    def expect(scenario: str, variable: str, value: str, status: str) -> None:
        row = annotations_by_scenario.get((scenario, variable), {})
        add_error(errors, row.get("gold_value") == value, f"{scenario}/{variable}: unexpected gold_value")
        add_error(errors, row.get("documentation_status") == status, f"{scenario}/{variable}: unexpected status")

    expect("complete", "pathological_n_category", "pN0", "supported")
    expect("no_nodes_cannot_be_assigned", "pathological_n_category", "", "cannot_be_assigned")
    expect("pn_not_documented", "pathological_n_category", "", "not_documented")
    expect("explicit_pnx", "pathological_n_category", "pNX", "cannot_be_assigned")
    expect("amended", "tumor_size", "4.1 cm", "supported")
    expect("amended", "pathological_t_category", "pT2b", "supported")
    expect("negated", "histologic_diagnosis", "", "negated")

    template_rows = read_csv(TEMPLATE_PATH, errors)
    add_error(errors, len(template_rows) == 4, "annotation_template.csv: expected one row per core variable")
    add_error(
        errors,
        {row.get("variable_name") for row in template_rows} == CORE_VARIABLES,
        "annotation_template.csv: variable coverage is incomplete",
    )

    if errors:
        print("Synthetic demo data validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Validated 8 visibly synthetic reports, 8 required scenarios, "
        "32 unique gold annotations, exact evidence offsets, and the annotation template."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
