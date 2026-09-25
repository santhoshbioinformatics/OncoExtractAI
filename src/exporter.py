"""Pure serialization helpers for reviewed abstractions and audit history."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Iterable

from .review_workflow import AuditRecord, ReviewSnapshot
from .schemas import ComparisonResult, ExtractionResult


EXPORT_LABEL = "Illustrative synthetic or approved local results"
EVALUATION_EXPORT_LABEL = "Illustrative synthetic results"
RESEARCH_DISCLAIMER = "Research and review prototype; not for diagnosis or clinical use."


def _spreadsheet_safe(value: Any) -> Any:
    """Prevent exported reviewer text from becoming a spreadsheet formula."""

    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    if stripped.startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _write_csv(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _spreadsheet_safe(row.get(key, "")) for key in fieldnames})
    return buffer.getvalue()


def reviewed_results_json(
    snapshots: Iterable[ReviewSnapshot],
    audit_records: Iterable[AuditRecord] = (),
) -> str:
    payload = {
        "metadata": {
            "label": EXPORT_LABEL,
            "disclaimer": RESEARCH_DISCLAIMER,
        },
        "reviewed_abstractions": [
            snapshot.model_dump(mode="json") for snapshot in snapshots
        ],
        "audit_history": [record.model_dump(mode="json") for record in audit_records],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def reviewed_results_csv(snapshots: Iterable[ReviewSnapshot]) -> str:
    fieldnames = [
        "export_label",
        "report_id",
        "method",
        "model_version",
        "source_report_digest",
        "extraction_timestamp",
        "document_provenance_json",
        "reviewer_identity",
        "reviewer_identity_verified",
        "review_id",
        "reviewed_at",
        "review_started_at",
        "review_duration_seconds",
        "returned_for_clarification",
        "original_review_priority",
        "original_review_priority_reason",
        "residual_review_priority",
        "residual_review_priority_reason",
        "variable_name",
        "original_value",
        "original_status",
        "review_action",
        "reviewed_value",
        "reviewed_status",
        "reviewed_alternatives_json",
        "review_reason",
        "evidence_json",
        "overall_note",
    ]
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for field in snapshot.fields:
            rows.append(
                {
                    "export_label": EXPORT_LABEL,
                    "report_id": snapshot.report_id,
                    "method": snapshot.method,
                    "model_version": snapshot.model_version,
                    "source_report_digest": snapshot.source_report_digest,
                    "extraction_timestamp": snapshot.extraction_timestamp or "",
                    "document_provenance_json": json.dumps(
                        snapshot.document_provenance.model_dump(mode="json")
                        if snapshot.document_provenance else {}, ensure_ascii=False
                    ),
                    "reviewer_identity": snapshot.reviewer_identity,
                    "reviewer_identity_verified": snapshot.reviewer_identity_verified,
                    "review_id": snapshot.review_id,
                    "reviewed_at": snapshot.reviewed_at,
                    "review_started_at": snapshot.review_started_at or "",
                    "review_duration_seconds": snapshot.review_duration_seconds,
                    "returned_for_clarification": snapshot.returned_for_clarification,
                    "original_review_priority": snapshot.original_review_priority,
                    "original_review_priority_reason": snapshot.original_review_priority_reason,
                    "residual_review_priority": snapshot.review_priority,
                    "residual_review_priority_reason": snapshot.review_priority_reason,
                    "variable_name": field.variable_name,
                    "original_value": field.original_output.extracted_value or "",
                    "original_status": field.original_output.documentation_status.value,
                    "review_action": field.review_action.value,
                    "reviewed_value": field.reviewed_value or "",
                    "reviewed_status": field.reviewed_status.value,
                    "reviewed_alternatives_json": json.dumps(
                        field.reviewed_alternatives, ensure_ascii=False
                    ),
                    "review_reason": field.review_reason,
                    "evidence_json": json.dumps(
                        [item.model_dump(mode="json") for item in field.evidence],
                        ensure_ascii=False,
                    ),
                    "overall_note": snapshot.overall_note,
                }
            )
    return _write_csv(rows, fieldnames)


def audit_history_csv(records: Iterable[AuditRecord]) -> str:
    fieldnames = [
        "export_label",
        "audit_id",
        "review_id",
        "timestamp",
        "report_id",
        "method",
        "model_version",
        "source_report_digest",
        "extraction_timestamp",
        "reviewer_identity",
        "reviewer_identity_verified",
        "variable_name",
        "original_value",
        "original_status",
        "reviewer_action",
        "corrected_value",
        "resulting_value",
        "resulting_status",
        "resulting_alternatives_json",
        "review_reason",
        "original_evidence_json",
        "resulting_evidence_json",
    ]
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "export_label": EXPORT_LABEL,
                "audit_id": record.audit_id,
                "review_id": record.review_id,
                "timestamp": record.timestamp,
                "report_id": record.report_id,
                "method": record.method,
                "model_version": record.model_version,
                "source_report_digest": record.source_report_digest,
                "extraction_timestamp": record.extraction_timestamp or "",
                "reviewer_identity": record.reviewer_identity,
                "reviewer_identity_verified": record.reviewer_identity_verified,
                "variable_name": record.variable_name,
                "original_value": record.original_output.extracted_value or "",
                "original_status": record.original_output.documentation_status.value,
                "reviewer_action": record.reviewer_action.value,
                "corrected_value": record.corrected_value or "",
                "resulting_value": record.resulting_value or "",
                "resulting_status": record.resulting_status.value,
                "resulting_alternatives_json": json.dumps(
                    record.resulting_alternatives, ensure_ascii=False
                ),
                "review_reason": record.review_reason,
                "original_evidence_json": json.dumps(
                    [
                        evidence.model_dump(mode="json")
                        for evidence in record.original_output.evidence or []
                    ],
                    ensure_ascii=False,
                ),
                "resulting_evidence_json": json.dumps(
                    [evidence.model_dump(mode="json") for evidence in record.resulting_evidence],
                    ensure_ascii=False,
                ),
            }
        )
    return _write_csv(rows, fieldnames)


def extraction_result_json(result: ExtractionResult) -> str:
    return json.dumps(
        {
            "metadata": {
                "label": EXPORT_LABEL,
                "disclaimer": RESEARCH_DISCLAIMER,
                "review_state": "unreviewed",
            },
            "extraction": result.model_dump(mode="json"),
        },
        indent=2,
        ensure_ascii=False,
    )


def evaluation_comparison_csv(comparison: ComparisonResult) -> str:
    fieldnames = [
        "export_label",
        "method",
        "metric",
        "numerator",
        "denominator",
        "rate",
        "definition",
    ]
    rows: list[dict[str, Any]] = []
    for metrics in (comparison.baseline_metrics, comparison.evidence_first_metrics):
        true_positives = sum(item.true_positives for item in metrics.variable_metrics)
        false_positives = sum(item.false_positives for item in metrics.variable_metrics)
        false_negatives = sum(item.false_negatives for item in metrics.variable_metrics)
        unsupported = sum(item.unsupported_count for item in metrics.variable_metrics)
        specs = [
            (
                "unsupported_extraction_rate",
                unsupported,
                metrics.returned_value_count,
                metrics.unsupported_extraction_rate,
                "Returned values without valid exact value-supporting evidence / returned values",
            ),
            (
                "overall_precision",
                true_positives,
                true_positives + false_positives,
                metrics.overall_precision,
                "Correct returned values / all returned values scored against gold",
            ),
            (
                "overall_recall",
                true_positives,
                true_positives + false_negatives,
                metrics.overall_recall,
                "Correct returned values / gold fields with a value",
            ),
            (
                "overall_f1",
                "",
                "",
                metrics.overall_f1,
                "Harmonic mean of overall precision and recall",
            ),
            (
                "coverage",
                metrics.returned_value_count,
                metrics.fields_evaluated,
                metrics.coverage,
                "Returned values / annotated fields",
            ),
            (
                "appropriate_abstention_rate",
                metrics.appropriate_abstention_count,
                metrics.expected_abstention_count,
                metrics.appropriate_abstention_rate,
                "Abstentions on gold no-value fields / gold no-value fields",
            ),
            (
                "false_positive_rate_undocumented",
                metrics.undocumented_false_positive_count,
                metrics.undocumented_gold_field_count,
                metrics.false_positive_rate_undocumented,
                "Returned values on not-documented gold fields / not-documented gold fields",
            ),
            (
                "documentation_status_accuracy",
                metrics.documentation_status_correct_count,
                metrics.fields_evaluated,
                metrics.documentation_status_accuracy,
                "Exact QA-status matches / annotated fields",
            ),
            (
                "exact_evidence_rate",
                metrics.exact_evidence_correct_count,
                metrics.exact_evidence_total_count,
                metrics.exact_evidence_rate,
                "Returned values with exact value-supporting evidence / returned values",
            ),
            (
                "manual_review_referral_rate",
                metrics.manual_review_count,
                metrics.report_count,
                metrics.manual_review_referral_rate,
                "Reports referred for manual review / reports",
            ),
            (
                "manual_review_referral_precision",
                metrics.review_referral_true_positive_count,
                (
                    metrics.review_referral_true_positive_count
                    + metrics.review_referral_false_positive_count
                ),
                metrics.manual_review_referral_precision,
                "Gold review-needed reports among referred reports / referred reports",
            ),
            (
                "manual_review_referral_recall",
                metrics.review_referral_true_positive_count,
                metrics.gold_manual_review_report_count,
                metrics.manual_review_referral_recall,
                "Referred gold review-needed reports / gold review-needed reports",
            ),
            (
                "schema_validity_rate",
                metrics.schema_valid_count,
                metrics.report_count,
                metrics.schema_validity_rate,
                "Contract-valid extraction results / reports",
            ),
        ]
        for metric, numerator, denominator, rate, definition in specs:
            rows.append(
                {
                    "export_label": EVALUATION_EXPORT_LABEL,
                    "method": metrics.method,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "rate": "N/A" if rate is None else rate,
                    "definition": definition,
                }
            )
    return _write_csv(rows, fieldnames)
