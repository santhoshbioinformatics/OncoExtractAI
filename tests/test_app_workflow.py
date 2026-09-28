"""End-to-end regression tests for safety-critical Streamlit workflow state."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
PASTE_APPROVAL_LABEL = (
    "I confirm this text is synthetic, de-identified, or approved for local research use."
)
CONTENT_A = (
    "LUNG RESECTION\n"
    "Invasive adenocarcinoma measuring 3.2 cm.\n"
    "Pathologic stage: pT2a pN0."
)
CONTENT_B = (
    "LUNG RESECTION\n"
    "Invasive squamous cell carcinoma measuring 4.1 cm.\n"
    "Pathologic stage: pT2b pN0."
)


def _single_labeled(elements: Iterable[Any], label: str) -> Any:
    matches = [element for element in elements if element.label == label]
    assert len(matches) == 1, f"Expected one {label!r} element, found {len(matches)}"
    return matches[0]


def _assert_no_app_exceptions(app: AppTest) -> None:
    assert not app.exception, [exception.value for exception in app.exception]


def test_paste_approval_is_content_bound_and_loader_is_cleared() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    _assert_no_app_exceptions(app)

    _single_labeled(app.text_area, "Pathology report text").set_value(CONTENT_A).run()
    _single_labeled(app.checkbox, PASTE_APPROVAL_LABEL).check().run()

    # Editing approved content revokes approval, and the changed text cannot load.
    _single_labeled(app.text_area, "Pathology report text").set_value(CONTENT_B).run()
    assert _single_labeled(app.checkbox, PASTE_APPROVAL_LABEL).value is False
    _single_labeled(app.button, "Load pasted report").click().run()
    _assert_no_app_exceptions(app)
    assert app.session_state["active_report_id"] is None
    assert any("approved" in message.value.lower() for message in app.error)

    # Re-approving the current content permits the load and retires loader state.
    _single_labeled(app.checkbox, PASTE_APPROVAL_LABEL).check().run()
    _single_labeled(app.button, "Load pasted report").click().run()
    _assert_no_app_exceptions(app)
    report_id = app.session_state["active_report_id"]
    assert app.session_state["loaded_reports"][report_id]["text"] == CONTENT_B
    assert _single_labeled(app.text_area, "Pathology report text").value == ""
    assert _single_labeled(app.checkbox, PASTE_APPROVAL_LABEL).value is False

    # Clearing the report also discards any new sensitive loader draft.
    _single_labeled(app.text_area, "Pathology report text").set_value(CONTENT_A).run()
    _single_labeled(app.checkbox, PASTE_APPROVAL_LABEL).check().run()
    _single_labeled(app.button, "Clear report").click().run()
    confirm_clear = next(
        button for button in app.button
        if button.label == "Clear report" and button.key == "confirm_clear_report"
    )
    confirm_clear.click().run()

    _assert_no_app_exceptions(app)
    assert app.session_state["active_report_id"] is None
    loader_version = app.session_state["loader_version"]
    current_text = next(
        item for item in app.text_area
        if item.key == f"pasted_report_text__v{loader_version}"
    )
    current_approval = next(
        item for item in app.checkbox
        if item.key == f"paste_approved__v{loader_version}"
    )
    assert current_text.value == ""
    assert current_approval.value is False


def test_sample_extraction_review_comparison_evaluation_and_audit_flow() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    _single_labeled(app.button, "Load sample report").click().run()
    _single_labeled(app.button, "Run extraction").click().run()
    _assert_no_app_exceptions(app)
    _single_labeled(app.text_input, "Assigned reviewer").set_value("Test Reviewer").run()

    workspace_area = _single_labeled(app.radio, "Workspace area")
    workspace_area.set_value("Evidence Viewer").run()
    _assert_no_app_exceptions(app)
    _single_labeled(app.radio, "Workspace area").set_value("Documentation Issues").run()
    _assert_no_app_exceptions(app)
    _single_labeled(app.radio, "Workspace area").set_value("Human Review").run()

    while True:
        pending = [
            element
            for element in app.selectbox
            if element.label == "Reviewer decision" and element.value != "accepted"
        ]
        if not pending:
            break
        pending[0].select("accepted").run()
    _single_labeled(app.button, "Complete review").click().run()
    _assert_no_app_exceptions(app)
    assert len(app.session_state["review_snapshots"]) == 1
    assert len(app.session_state["audit_records"]) == 4
    completed_snapshot = app.session_state["review_snapshots"][0]
    assert completed_snapshot.review_started_at is not None
    assert completed_snapshot.review_duration_seconds is not None
    assert completed_snapshot.review_duration_seconds >= 0
    assert completed_snapshot.reviewer_identity == "Test Reviewer"
    assert completed_snapshot.reviewer_identity_verified is False
    assert completed_snapshot.source_report_digest.startswith("sha256:")

    _single_labeled(app.radio, "Navigation").set_value("Comparison").run()
    _single_labeled(app.button, "Run all methods").click().run()
    _assert_no_app_exceptions(app)
    comparison_frames = [
        frame.value for frame in app.dataframe
        if "Agreement" in frame.value.columns
    ]
    assert len(comparison_frames) == 1
    assert list(comparison_frames[0].columns) == [
        "Field", "Baseline", "Evidence-first", "ML", "Agreement"
    ]
    assert len(comparison_frames[0]) == 4

    _single_labeled(app.button, "Evaluation").click().run()
    _single_labeled(app.button, "Run synthetic evaluation").click().run()
    _assert_no_app_exceptions(app)
    comparison = app.session_state["evaluation_comparison"]
    assert comparison.baseline_metrics.report_count == 8
    assert comparison.evidence_first_metrics.fields_evaluated == 32
    assert app.dataframe
    metric_labels = {metric.label for metric in app.metric}
    assert "Median review time / report" in metric_labels
    assert "Field acceptance rate" in metric_labels
    assert "Returned for clarification" in metric_labels
    assert "Completed without editing" in metric_labels

    _single_labeled(app.radio, "Navigation").set_value("Audit & Export").run()
    _assert_no_app_exceptions(app)
    assert len(app.session_state["audit_records"]) == 4
