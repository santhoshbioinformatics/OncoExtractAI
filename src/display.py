"""Presentation metadata and safe evidence highlighting helpers."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable

from .schemas import DocumentationStatus, VariableExtraction


@dataclass(frozen=True)
class StatusPresentation:
    label: str
    tone: str
    description: str


STATUS_PRESENTATION: dict[DocumentationStatus, StatusPresentation] = {
    DocumentationStatus.SUPPORTED: StatusPresentation(
        "Supported", "positive", "An exact report span supports the proposed value."
    ),
    DocumentationStatus.NOT_DOCUMENTED: StatusPresentation(
        "Not documented", "neutral", "The report does not explicitly document this field."
    ),
    DocumentationStatus.CANNOT_BE_ASSIGNED: StatusPresentation(
        "Cannot be assigned", "neutral", "The report explicitly says the category cannot be assigned."
    ),
    DocumentationStatus.NEGATED: StatusPresentation(
        "Negated", "neutral", "The report explicitly denies the candidate finding."
    ),
    DocumentationStatus.UNCERTAIN: StatusPresentation(
        "Uncertain", "warning", "The report qualifies the candidate as uncertain or provisional."
    ),
    DocumentationStatus.CONFLICTING: StatusPresentation(
        "Conflicting", "danger", "Different report statements cannot be safely reconciled."
    ),
    DocumentationStatus.SUPERSEDED: StatusPresentation(
        "Superseded", "warning", "A later explicitly linked statement replaces this finding."
    ),
    DocumentationStatus.UNSUPPORTED: StatusPresentation(
        "Unsupported", "danger", "The proposed value does not have valid exact evidence."
    ),
    DocumentationStatus.MANUAL_REVIEW_REQUIRED: StatusPresentation(
        "Manual review required", "danger", "The field is intentionally withheld for reviewer resolution."
    ),
}


VARIABLE_LABELS = {
    "histologic_diagnosis": "Histologic diagnosis",
    "tumor_size": "Tumor size",
    "pathological_t_category": "Pathological T category",
    "pathological_n_category": "Pathological N category",
}


REVIEW_ACTION_LABELS = {
    "pending": "Pending",
    "accepted": "Accept",
    "corrected": "Correct",
    "rejected": "Reject",
    "flagged_for_review": "Flag",
}


def status_presentation(status: DocumentationStatus | str) -> StatusPresentation:
    """Return the canonical label, tone, and plain-language definition."""

    resolved = status if isinstance(status, DocumentationStatus) else DocumentationStatus(status)
    return STATUS_PRESENTATION[resolved]


def variable_label(variable_name: str) -> str:
    return VARIABLE_LABELS.get(variable_name, variable_name.replace("_", " ").title())


def evidence_anchor(variable_name: str) -> str:
    return f"evidence-{variable_name.replace('_', '-')}"


def render_highlighted_report(
    report_text: str,
    variables: Iterable[VariableExtraction],
    focused_variable: str | None = None,
) -> str:
    """Return escaped report HTML with exact-offset evidence highlights.

    Highlight boundaries are built from offsets rather than string replacement,
    so repeated, multiline, and overlapping evidence remains aligned to the
    canonical report.  A zero-width anchor is inserted for each field's first
    valid evidence span so field links can scroll to the source.
    """

    spans: list[tuple[int, int, str]] = []
    anchors_by_position: dict[int, list[str]] = {}

    for variable in variables:
        first_valid_start: int | None = None
        for evidence in variable.evidence or []:
            start, end = evidence.start_offset, evidence.end_offset
            if not (0 <= start < end <= len(report_text)):
                continue
            if report_text[start:end] != evidence.text:
                continue
            spans.append((start, end, variable.variable_name))
            if first_valid_start is None or start < first_valid_start:
                first_valid_start = start
        if first_valid_start is not None:
            anchors_by_position.setdefault(first_valid_start, []).append(variable.variable_name)

    if not spans:
        return html.escape(report_text)

    boundaries = {0, len(report_text)}
    for start, end, _ in spans:
        boundaries.update((start, end))
    ordered = sorted(boundaries)
    rendered: list[str] = []

    for index, start in enumerate(ordered[:-1]):
        end = ordered[index + 1]
        for variable_name in anchors_by_position.get(start, []):
            rendered.append(
                f'<span class="evidence-anchor" id="{evidence_anchor(variable_name)}"></span>'
            )

        escaped_segment = html.escape(report_text[start:end])
        covering = {
            variable_name
            for span_start, span_end, variable_name in spans
            if span_start <= start and end <= span_end
        }
        if not covering:
            rendered.append(escaped_segment)
            continue

        classes = ["evidence-mark"]
        if focused_variable and focused_variable in covering:
            classes.append("evidence-mark--focused")
        label = ", ".join(variable_label(name) for name in sorted(covering))
        rendered.append(
            f'<mark class="{" ".join(classes)}" title="Evidence for {html.escape(label)}">'
            f"{escaped_segment}</mark>"
        )

    for variable_name in anchors_by_position.get(len(report_text), []):
        rendered.append(
            f'<span class="evidence-anchor" id="{evidence_anchor(variable_name)}"></span>'
        )
    return "".join(rendered)

