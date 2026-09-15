"""Acceptance checks for the synthetic demonstration set and gold labels."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json

import pytest

from src.schemas import CORE_VARIABLES, DocumentationStatus
from src.utils import load_gold_annotations, load_synthetic_reports, parse_gold_annotations


def test_every_demo_report_is_visibly_synthetic_and_has_no_direct_identifiers() -> None:
    reports = load_synthetic_reports()
    assert len(reports) == 8
    for report in reports:
        assert report.get("is_synthetic") is True
        text = report["text"]
        assert "SYNTHETIC" in text.upper()
        assert "Patient Name:" not in text
        assert "Medical Record Number:" not in text
        assert "MRN:" not in text


def test_gold_annotations_cover_each_core_field_once_with_exact_offsets() -> None:
    reports = {report["report_id"]: report["text"] for report in load_synthetic_reports()}
    annotations = parse_gold_annotations(report_texts=reports)
    assert len(annotations) == len(reports) * len(CORE_VARIABLES)

    counts = Counter((item.report_id, item.variable_name) for item in annotations)
    assert set(counts.values()) == {1}
    for annotation in annotations:
        if annotation.evidence_text is None:
            continue
        assert annotation.evidence_start_offset is not None
        assert annotation.evidence_end_offset is not None
        assert (
            reports[annotation.report_id][
                annotation.evidence_start_offset:annotation.evidence_end_offset
            ]
            == annotation.evidence_text
        )


def test_pn0_pnx_and_not_documented_are_distinct_reference_states() -> None:
    annotations = parse_gold_annotations()
    pn = {
        item.report_id: item
        for item in annotations
        if item.variable_name == "pathological_n_category"
    }

    pn0 = next(item for item in pn.values() if item.gold_value == "pN0")
    pnx = next(item for item in pn.values() if item.gold_value == "pNX")
    not_documented = next(
        item
        for item in pn.values()
        if item.documentation_status == DocumentationStatus.NOT_DOCUMENTED
    )

    assert pn0.documentation_status == DocumentationStatus.SUPPORTED
    assert pnx.documentation_status == DocumentationStatus.CANNOT_BE_ASSIGNED
    assert not_documented.gold_value is None
    assert not_documented.evidence_text is None
    assert len({pn0.report_id, pnx.report_id, not_documented.report_id}) == 3


def test_demo_set_contains_required_edge_case_statuses() -> None:
    statuses = {item.documentation_status for item in parse_gold_annotations()}
    assert DocumentationStatus.SUPPORTED in statuses
    assert DocumentationStatus.NOT_DOCUMENTED in statuses
    assert DocumentationStatus.CANNOT_BE_ASSIGNED in statuses
    assert DocumentationStatus.NEGATED in statuses
    assert DocumentationStatus.UNCERTAIN in statuses
    assert DocumentationStatus.CONFLICTING in statuses


def test_gold_parser_fails_when_a_report_is_omitted_or_an_offset_is_not_an_integer() -> None:
    reports = {report["report_id"]: report["text"] for report in load_synthetic_reports()}
    rows = load_gold_annotations()
    omitted_id = rows[0]["report_id"]
    incomplete = [row for row in rows if row["report_id"] != omitted_id]
    with pytest.raises(ValueError, match="coverage"):
        parse_gold_annotations(incomplete, reports)

    malformed = deepcopy(rows)
    malformed[0]["evidence_start_offset"] = "1.5"
    with pytest.raises(ValueError, match="non-negative integers"):
        parse_gold_annotations(malformed, reports)


def test_runtime_loader_rejects_a_fixture_that_is_not_marked_synthetic(tmp_path) -> None:
    report = deepcopy(load_synthetic_reports()[0])
    report["is_synthetic"] = False
    path = tmp_path / "reports.json"
    path.write_text(json.dumps({"reports": [report]}), encoding="utf-8")
    with pytest.raises(ValueError, match="is_synthetic"):
        load_synthetic_reports(str(path))
