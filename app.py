"""Production-quality Streamlit UI for the OncoExtractAI-QA research prototype.

The UI owns presentation and in-session workflow state only. Validated domain
modules handle report preparation, extraction, review snapshots, and exports.
"""

from __future__ import annotations

import hashlib
import html
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd
import streamlit as st

from src.display import (
    REVIEW_ACTION_LABELS,
    STATUS_PRESENTATION,
    evidence_anchor,
    render_highlighted_report,
    status_presentation,
    variable_label,
)
from src.evaluation import Evaluator
from src.exporter import (
    audit_history_csv,
    evaluation_comparison_csv,
    extraction_result_json,
    reviewed_results_csv,
    reviewed_results_json,
)
from src.report_input import (
    ReportInputError,
    decode_uploaded_text,
    find_direct_identifier_labels,
    prepare_report,
)
from src.review_workflow import (
    AuditRecord,
    ReviewSnapshot,
    copy_decisions,
    create_review_snapshot,
    initial_review_decisions,
    review_progress,
    validate_review_decisions,
)
from src.schemas import (
    CORE_VARIABLES,
    ComparisonResult,
    DocumentationStatus,
    ExtractionResult,
    ReviewAction,
    VariableExtraction,
)
from src.utils import load_synthetic_reports, parse_gold_annotations
from src.workflow import (
    ExtractionPipelineError,
    run_both_pipelines,
    run_extraction_pipeline,
)


# --- Authentication configuration ---
# Override with environment variables for production use.
DEFAULT_USERNAME = os.environ.get("ONCOEXTRACT_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("ONCOEXTRACT_PASSWORD", "admin123")

st.set_page_config(
    page_title="OncoExtract",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #0b1f33; --teal: #0f6f78; --canvas: #f6f8fa;
        --paper: #fff; --text: #172b3a; --muted: #556b7c;
        --line: #dbe3ea; --soft: #edf2f6; --radius: 10px;
        --green: #246b45; --green-bg: #e5f4ea;
        --amber: #7a4b00; --amber-bg: #fff1d6;
        --red: #8f2929; --red-bg: #fbe7e7;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--canvas); }
    .block-container { max-width: 1440px; padding-top: 1.5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #f2f7fa; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        min-height: 40px; border-radius: 8px; padding: .35rem .45rem;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stCaption p { color: inherit; }
    [data-testid="stSidebar"] .sidebar-profile {
        background: rgba(255,255,255,.14) !important;
        border: 1px solid rgba(255,255,255,.28) !important;
        border-radius: 10px !important;
        margin: .35rem 0 .75rem !important;
        padding: .85rem .9rem !important;
    }
    [data-testid="stSidebar"] .sidebar-profile,
    [data-testid="stSidebar"] .sidebar-profile * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] .sidebar-profile-name {
        font-size: 1rem !important;
        font-weight: 750 !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }
    [data-testid="stSidebar"] .sidebar-profile-status {
        color: #c5e4ec !important;
        font-size: .8rem !important;
        margin: .25rem 0 0 !important;
        opacity: 1 !important;
    }
    /* Keep button labels dark even though sidebar forces light text on p/span */
    [data-testid="stSidebar"] div[data-testid="stButton"] button,
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"],
    [data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background: #ffffff !important;
        border: 1px solid #ffffff !important;
        color: #0b1f33 !important;
        font-weight: 750 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #d9eef2 !important;
        border-color: #d9eef2 !important;
        color: #0b1f33 !important;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button *,
    [data-testid="stSidebar"] div[data-testid="stButton"] button p,
    [data-testid="stSidebar"] div[data-testid="stButton"] button span {
        color: #0b1f33 !important;
    }
    h1, h2, h3 { color: var(--navy); letter-spacing: -.02em; }
    h1 { font-size: clamp(1.75rem, 3vw, 2.25rem); }
    h2 { font-size: clamp(1.3rem, 2vw, 1.6rem); }
    p, label, li, button, input, textarea, select { font-size: .94rem; }
    small, .stCaption p { font-size: .78rem !important; }
    :focus-visible { outline: 3px solid #18a3ad !important; outline-offset: 2px !important; }
    a { color: #0b6570; } a:hover { color: #084b53; }
    .page-heading { margin-bottom: 1rem; }
    .page-heading h1 { margin: 0 0 .35rem; line-height: 1.15; }
    .page-heading p { color: var(--muted); margin: 0; max-width: 880px; line-height: 1.55; }
    .section-heading { color: var(--navy); font-size: 1rem; font-weight: 750; margin: 0 0 .55rem; }
    .research-banner {
        background: #fff8e8; border: 1px solid #ead29b; border-left: 4px solid #a56b05;
        border-radius: var(--radius); color: #624308; font-size: .86rem;
        line-height: 1.45; margin: 0 0 1rem; padding: .7rem .85rem;
    }
    .case-strip, .priority-strip {
        background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius);
        margin: .75rem 0 1rem; padding: .8rem .95rem;
    }
    .case-strip { align-items: center; display: flex; flex-wrap: wrap; gap: .55rem 1rem; }
    .case-id { color: var(--navy); font-weight: 750; overflow-wrap: anywhere; }
    .case-meta { color: var(--muted); font-size: .8rem; overflow-wrap: anywhere; }
    .priority-strip p { color: #40586b; font-size: .84rem; line-height: 1.5; margin: .35rem 0 0; }
    .status-pill, .priority-pill {
        align-items: center; border: 1px solid transparent; border-radius: 999px;
        display: inline-flex; font-size: .75rem; font-weight: 750; line-height: 1.2;
        max-width: 100%; min-height: 26px; padding: .28rem .58rem; white-space: normal;
    }
    .tone-positive { background: var(--green-bg); color: var(--green); border-color: #bfdfca; }
    .tone-neutral { background: var(--soft); color: #405465; border-color: #d7e0e6; }
    .tone-warning { background: var(--amber-bg); color: var(--amber); border-color: #ecd69e; }
    .tone-danger { background: var(--red-bg); color: var(--red); border-color: #edc4c4; }
    .field-card, .review-card, .method-panel {
        background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius);
        margin-bottom: .75rem; padding: .9rem 1rem;
    }
    .field-top { align-items: flex-start; display: flex; flex-wrap: wrap;
        gap: .6rem 1rem; justify-content: space-between; }
    .field-label, .comparison-label { color: var(--muted); font-size: .74rem;
        font-weight: 750; letter-spacing: .055em; text-transform: uppercase; }
    .field-value { color: var(--navy); font-size: 1.02rem; font-weight: 720;
        line-height: 1.4; margin-top: .25rem; overflow-wrap: anywhere; }
    .field-note, .distinction-note { color: var(--muted); font-size: .8rem;
        line-height: 1.45; margin-top: .5rem; overflow-wrap: anywhere; }
    .distinction-note { background: #eaf3fb; border-radius: 7px; color: #254f70; padding: .5rem .6rem; }
    .anchor-link { display: inline-block; font-size: .8rem; font-weight: 650; margin-top: .55rem; }
    .evidence-nav { align-items: center; display: flex; flex-wrap: wrap;
        gap: .45rem; margin: .4rem 0 .75rem; }
    .evidence-nav a { background: var(--paper); border: 1px solid var(--line);
        border-radius: 7px; font-size: .78rem; font-weight: 650;
        padding: .35rem .55rem; text-decoration: none; }
    .report-reader { background: var(--paper); border: 1px solid var(--line);
        border-radius: var(--radius); color: #203746;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .84rem; line-height: 1.7; max-height: 68vh; min-height: 360px;
        overflow: auto; overflow-wrap: anywhere; padding: 1rem 1.05rem; white-space: pre-wrap; }
    .evidence-anchor { scroll-margin-top: 1rem; }
    .evidence-mark { background: #d9f1ea; border-bottom: 2px solid #3c9b7c;
        border-radius: 2px; color: #183e32; padding: 1px 0; }
    .evidence-mark--focused { background: #ffe7a8; border-bottom-color: #af7200;
        box-shadow: 0 0 0 2px rgba(175,114,0,.18); }
    .evidence-excerpt { background: #f8fafb; border: 1px solid var(--line);
        border-left: 3px solid #55959b; border-radius: 7px; color: #294150;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: .82rem; line-height: 1.55; margin: .4rem 0 .75rem;
        overflow-wrap: anywhere; padding: .65rem .75rem; white-space: pre-wrap; }
    .offset-label { color: var(--muted); font-size: .76rem; font-weight: 650; }
    .empty-panel { background: var(--paper); border: 1px dashed #b8c7d2;
        border-radius: var(--radius); color: var(--muted); line-height: 1.5; padding: 1.25rem; }
    .review-card h3 { font-size: 1rem; margin: 0 0 .15rem; }
    .method-panel { min-height: 150px; }
    .method-panel--evidence { border-top: 3px solid var(--teal); }
    .method-name { color: var(--navy); font-weight: 750; }
    .method-value { font-size: .98rem; font-weight: 680; margin: .55rem 0; overflow-wrap: anywhere; }
    .status-legend { display: grid; gap: .55rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
    .legend-item { background: var(--paper); border: 1px solid var(--line);
        border-radius: var(--radius); padding: .75rem; }
    .legend-item p { color: var(--muted); font-size: .8rem; line-height: 1.4; margin: .4rem 0 0; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
    [data-testid="stExpander"] { background: var(--paper); border-color: var(--line); border-radius: var(--radius); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
    [data-testid="stTabs"] [data-baseweb="tab"] { min-height: 44px; white-space: nowrap; }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        border-radius: 8px; min-height: 40px; font-weight: 650;
    }
    #MainMenu, footer { visibility: hidden; }
    @media (max-width: 820px) {
        .block-container { padding: 1rem .75rem 2rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="column"] { flex: 1 1 280px !important;
            min-width: min(100%,280px) !important; width: 100% !important; }
        .report-reader { max-height: 55vh; min-height: 300px; }
        .status-legend { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


NAVIGATION = ("Workspace", "Comparison", "Evaluation", "Audit & Export")
METHOD_LABELS = {
    "evidence_first": "Evidence-first QA",
    "baseline": "Conventional baseline",
    "ml": "ML model (TF-IDF + logistic regression)",
}
ACTION_OPTIONS = (
    ReviewAction.PENDING.value,
    ReviewAction.ACCEPTED.value,
    ReviewAction.CORRECTED.value,
    ReviewAction.REJECTED.value,
    ReviewAction.FLAGGED.value,
)
STATUS_OPTIONS = tuple(status.value for status in STATUS_PRESENTATION)
PRIORITY_TONES = {"low": "positive", "medium": "warning", "high": "danger", "critical": "danger"}
SENSITIVE_LOADER_WIDGETS = (
    "pasted_report_text",
    "paste_cancer_type",
    "paste_approved",
    "uploaded_report_file",
    "upload_cancer_type",
    "upload_approved",
)


def _loader_widget_key(base: str, version: int | None = None) -> str:
    """Return a versioned key so sensitive loader values can be retired safely."""
    resolved_version = (
        int(st.session_state.get("loader_version", 0))
        if version is None else version
    )
    return f"{base}__v{resolved_version}"


def _content_digest(content: str | bytes) -> str:
    """Hash the exact text or file bytes covered by a user's approval."""
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.sha256(payload).hexdigest()


def _purge_retired_loader_state() -> None:
    """Remove values belonging to loader widgets that are no longer rendered."""
    current_keys = {_loader_widget_key(base) for base in SENSITIVE_LOADER_WIDGETS}
    for key in list(st.session_state):
        is_legacy_key = key in SENSITIVE_LOADER_WIDGETS
        is_retired_version = (
            any(key.startswith(f"{base}__v") for base in SENSITIVE_LOADER_WIDGETS)
            and key not in current_keys
        )
        if is_legacy_key or is_retired_version:
            del st.session_state[key]


def _rotate_sensitive_loader_state() -> None:
    """Retire pasted/uploaded content after a completed load or clear action."""
    st.session_state.loader_version = int(st.session_state.loader_version) + 1
    st.session_state.paste_approval_digest = None
    st.session_state.upload_approval_digest = None


def _revoke_paste_approval(version: int) -> None:
    st.session_state[_loader_widget_key("paste_approved", version)] = False
    st.session_state.paste_approval_digest = None


def _capture_paste_approval(version: int) -> None:
    approval_key = _loader_widget_key("paste_approved", version)
    text_key = _loader_widget_key("pasted_report_text", version)
    st.session_state.paste_approval_digest = (
        _content_digest(str(st.session_state.get(text_key, "")))
        if st.session_state.get(approval_key) is True else None
    )


def _revoke_upload_approval(version: int) -> None:
    st.session_state[_loader_widget_key("upload_approved", version)] = False
    st.session_state.upload_approval_digest = None


def _capture_upload_approval(version: int) -> None:
    approval_key = _loader_widget_key("upload_approved", version)
    upload_key = _loader_widget_key("uploaded_report_file", version)
    uploaded = st.session_state.get(upload_key)
    st.session_state.upload_approval_digest = (
        _content_digest(uploaded.getvalue())
        if st.session_state.get(approval_key) is True and uploaded is not None
        else None
    )


def _initialize_state() -> None:
    if "sample_reports" not in st.session_state:
        try:
            st.session_state.sample_reports = load_synthetic_reports()
            st.session_state.sample_load_error = None
        except (OSError, ValueError) as error:
            st.session_state.sample_reports = []
            st.session_state.sample_load_error = str(error)
    defaults: dict[str, Any] = {
        "nav_page": "Workspace", "loaded_reports": {}, "active_report_id": None,
        "extraction_results": {}, "review_decisions": {}, "review_baselines": {},
        "review_notes": {}, "review_note_baselines": {}, "review_draft_versions": {},
        "review_snapshots": [], "audit_records": [], "evaluation_comparison": None,
        "evaluation_run_at": None, "focused_variable": "all",
        "pending_report_change": None, "pending_rerun_method": None,
        "show_clear_dialog": False,
        "report_selector_version": 0, "flash_message": None,
        "workspace_method": "evidence_first",
        "workspace_area": "AI Abstraction",
        "loader_version": 0,
        "paste_approval_digest": None,
        "upload_approval_digest": None,
        "authenticated": False,
        "current_user": None,
        "login_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    _purge_retired_loader_state()


def _page_heading(title: str, description: str) -> None:
    st.markdown(
        f'<div class="page-heading"><h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(description)}</p></div>', unsafe_allow_html=True,
    )


def _research_banner() -> None:
    st.markdown(
        '<div class="research-banner" role="note"><strong>Research and review prototype.</strong> '
        'Use synthetic, de-identified, or institutionally approved text only. '
        'This tool is not for diagnosis, prognosis, or treatment decisions.</div>',
        unsafe_allow_html=True,
    )


def _show_flash_message() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not flash:
        return
    renderer = {"success": st.success, "warning": st.warning,
                "error": st.error, "info": st.info}.get(flash.get("kind"), st.info)
    renderer(str(flash.get("message", "")))


def _set_flash(kind: str, message: str) -> None:
    st.session_state.flash_message = {"kind": kind, "message": message}


def _status_badge(status: DocumentationStatus | str) -> str:
    presentation = status_presentation(status)
    return (f'<span class="status-pill tone-{html.escape(presentation.tone)}" '
            f'title="{html.escape(presentation.description, quote=True)}">'
            f'{html.escape(presentation.label)}</span>')


def _priority_badge(priority: str) -> str:
    tone = PRIORITY_TONES.get(priority, "neutral")
    return f'<span class="priority-pill tone-{tone}">{html.escape(priority.title())} priority</span>'


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _action_label(action: str) -> str:
    return REVIEW_ACTION_LABELS.get(action, action.replace("_", " ").title())


def _status_label(status: str) -> str:
    return status_presentation(status).label


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return (cleaned or "report")[:80]


def _draft_key(report_id: str, method: str) -> str:
    return f"{report_id}::{method}"


def _widget_scope(report_id: str, method: str) -> str:
    digest = hashlib.sha256(f"{report_id}::{method}".encode()).hexdigest()[:12]
    version = st.session_state.review_draft_versions.get(_draft_key(report_id, method), 0)
    return f"{digest}-{version}"


def _active_report() -> dict[str, Any] | None:
    report_id = st.session_state.active_report_id
    return st.session_state.loaded_reports.get(report_id) if report_id else None


def _result_for(report_id: str, method: str) -> ExtractionResult | None:
    return st.session_state.extraction_results.get(report_id, {}).get(method)


def _ensure_review_draft(result: ExtractionResult) -> str:
    key = _draft_key(result.report_id, result.method)
    if key not in st.session_state.review_decisions:
        decisions = initial_review_decisions(result)
        st.session_state.review_decisions[key] = decisions
        st.session_state.review_baselines[key] = copy_decisions(decisions)
        st.session_state.review_notes[key] = ""
        st.session_state.review_note_baselines[key] = ""
        st.session_state.review_draft_versions[key] = 0
    return key


def _reset_review_draft(result: ExtractionResult) -> None:
    key = _draft_key(result.report_id, result.method)
    decisions = initial_review_decisions(result)
    st.session_state.review_decisions[key] = decisions
    st.session_state.review_baselines[key] = copy_decisions(decisions)
    st.session_state.review_notes[key] = ""
    st.session_state.review_note_baselines[key] = ""
    st.session_state.review_draft_versions[key] = st.session_state.review_draft_versions.get(key, 0) + 1


def _review_is_dirty(key: str) -> bool:
    decisions = st.session_state.review_decisions.get(key)
    return decisions is not None and (
        decisions != st.session_state.review_baselines.get(key)
        or st.session_state.review_notes.get(key, "") != st.session_state.review_note_baselines.get(key, "")
    )


def _report_has_dirty_review(report_id: str | None) -> bool:
    prefix = f"{report_id}::" if report_id else ""
    return bool(prefix) and any(key.startswith(prefix) and _review_is_dirty(key)
                                for key in st.session_state.review_decisions)


def _purge_report_drafts(report_id: str) -> None:
    prefix = f"{report_id}::"
    for name in ("review_decisions", "review_baselines", "review_notes",
                 "review_note_baselines", "review_draft_versions"):
        collection = st.session_state[name]
        for key in [item for item in collection if item.startswith(prefix)]:
            collection.pop(key, None)


def _commit_report(report: dict[str, Any]) -> None:
    report_id = str(report["report_id"])
    existing = st.session_state.loaded_reports.get(report_id)
    if existing and existing.get("text") != report.get("text"):
        st.session_state.extraction_results.pop(report_id, None)
        _purge_report_drafts(report_id)
    st.session_state.loaded_reports[report_id] = report
    st.session_state.active_report_id = report_id
    st.session_state.focused_variable = "all"
    st.session_state.report_selector_version += 1
    _rotate_sensitive_loader_state()
    _set_flash("success", f"Loaded report {report_id} into the local workspace.")


def _activate_report(report_id: str) -> None:
    if report_id not in st.session_state.loaded_reports:
        _set_flash("error", "That report is no longer available in this session.")
        return
    st.session_state.active_report_id = report_id
    st.session_state.focused_variable = "all"
    st.session_state.report_selector_version += 1
    _set_flash("success", f"Opened report {report_id}.")


def _request_report_change(*, report: dict[str, Any] | None = None,
                           report_id: str | None = None) -> None:
    current = _active_report()
    current_id = str(current["report_id"]) if current else None
    target_id = str(report["report_id"]) if report else report_id
    same_text = not report or (current and current.get("text") == report.get("text"))
    if target_id == current_id and same_text:
        if report is not None:
            _rotate_sensitive_loader_state()
        _set_flash("info", f"Report {target_id} is already open.")
        st.rerun()
    replacing_current_content = (
        report is not None and target_id == current_id and not same_text
    )
    if replacing_current_content and _report_has_dirty_review(current_id):
        st.session_state.pending_report_change = {"report": report, "report_id": report_id}
        st.rerun()
    _commit_report(report) if report is not None else _activate_report(str(report_id))
    st.rerun()


def _clear_current_report() -> None:
    report = _active_report()
    if not report:
        return
    report_id = str(report["report_id"])
    st.session_state.loaded_reports.pop(report_id, None)
    st.session_state.extraction_results.pop(report_id, None)
    _purge_report_drafts(report_id)
    remaining = list(st.session_state.loaded_reports)
    st.session_state.active_report_id = remaining[0] if remaining else None
    st.session_state.focused_variable = "all"
    st.session_state.report_selector_version += 1
    _rotate_sensitive_loader_state()
    _set_flash("success", f"Cleared report {report_id}. Saved audit records remain available.")


def _store_results(report_id: str, results: dict[str, ExtractionResult]) -> None:
    st.session_state.extraction_results.setdefault(report_id, {}).update(results)
    for result in results.values():
        _reset_review_draft(result)


def _perform_extraction(method: Literal["baseline", "evidence_first", "ml", "both"]) -> None:
    report = _active_report()
    if not report:
        raise ExtractionPipelineError("Load a report before running extraction.")
    if method == "both":
        _store_results(str(report["report_id"]), run_both_pipelines(report))
    else:
        result = run_extraction_pipeline(report, method)
        _store_results(str(report["report_id"]), {method: result})


def _run_and_report(method: Literal["baseline", "evidence_first", "ml", "both"]) -> None:
    label = "all methods" if method == "both" else _method_label(method)
    try:
        with st.spinner(f"Running {label}…"):
            _perform_extraction(method)
    except ExtractionPipelineError as error:
        _set_flash("error", str(error))
    except Exception:
        _set_flash("error", "Extraction stopped safely because an unexpected error occurred.")
    else:
        _set_flash("success", f"Completed {label} with validated structured output.")


def _request_extraction(method: Literal["baseline", "evidence_first", "ml", "both"]) -> None:
    report = _active_report()
    if not report:
        _set_flash("error", "Load a report before running extraction.")
        st.rerun()
    methods = ("baseline", "evidence_first", "ml") if method == "both" else (method,)
    if any(_review_is_dirty(_draft_key(str(report["report_id"]), item)) for item in methods):
        st.session_state.pending_rerun_method = method
        st.rerun()
    _run_and_report(method)
    st.rerun()


@st.dialog("Discard unsaved review changes?")
def _confirm_report_change_dialog() -> None:
    pending = st.session_state.pending_report_change
    if not pending:
        st.info("There is no pending report change.")
        return
    st.warning(
        "This content would replace the current report under the same report ID. "
        "Its unsaved review draft will be discarded; saved reviews and audit "
        "records remain."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Discard and continue", type="primary", width="stretch",
                     key="confirm_report_change"):
            pending = st.session_state.pending_report_change
            st.session_state.pending_report_change = None
            if pending.get("report") is not None:
                _commit_report(pending["report"])
            else:
                _activate_report(str(pending["report_id"]))
            st.rerun()
    with cancel_col:
        if st.button("Keep current content", width="stretch",
                     key="cancel_report_change"):
            st.session_state.pending_report_change = None
            st.rerun()


@st.dialog("Clear current report?")
def _confirm_clear_dialog() -> None:
    report = _active_report()
    if not report:
        st.info("No report is loaded.")
        st.session_state.show_clear_dialog = False
        return
    dirty_message = (" Unsaved reviewer decisions will be discarded."
                     if _report_has_dirty_review(str(report["report_id"])) else "")
    st.warning(
        "This removes the report, extraction output, and current review draft "
        f"from the workspace.{dirty_message} Saved audit records are retained."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Clear report", type="primary", width="stretch",
                     key="confirm_clear_report"):
            st.session_state.show_clear_dialog = False
            _clear_current_report()
            st.rerun()
    with cancel_col:
        if st.button("Cancel", width="stretch", key="cancel_clear_report"):
            st.session_state.show_clear_dialog = False
            st.rerun()


@st.dialog("Rerun extraction and discard draft?")
def _confirm_rerun_dialog() -> None:
    method = st.session_state.pending_rerun_method
    if method not in {"baseline", "evidence_first", "ml", "both"}:
        st.info("There is no pending extraction.")
        return
    st.warning(
        "Rerunning will replace the selected extraction output and its unsaved "
        "review draft. Saved reviews and audit records remain available."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Discard draft and rerun", type="primary", width="stretch",
                     key="confirm_rerun"):
            method = st.session_state.pending_rerun_method
            st.session_state.pending_rerun_method = None
            _run_and_report(method)
            st.rerun()
    with cancel_col:
        if st.button("Keep draft", width="stretch", key="cancel_rerun"):
            st.session_state.pending_rerun_method = None
            st.rerun()


def _render_sidebar() -> str:
    st.sidebar.title("OncoExtract")
    st.sidebar.caption("Evidence-grounded lung pathology abstraction")

    # Profile + sign out near the top so they stay visible
    username = st.session_state.get("current_user") or DEFAULT_USERNAME
    st.sidebar.markdown(
        f'<div class="sidebar-profile">'
        f'<p class="sidebar-profile-name">👤 {html.escape(str(username))}</p>'
        f'<p class="sidebar-profile-status">Signed in</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Sign out", type="primary", width="stretch", key="sign_out_button"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.login_error = None
        st.rerun()

    st.sidebar.divider()
    page = st.sidebar.radio("Navigation", NAVIGATION, key="nav_page")
    st.sidebar.divider()
    reports = st.session_state.loaded_reports
    if reports:
        st.sidebar.subheader("Open reports")
        report_ids = list(reports)
        active_id = st.session_state.active_report_id
        active_index = report_ids.index(active_id) if active_id in report_ids else 0
        selected_id = st.sidebar.selectbox(
            "Choose report", report_ids, index=active_index,
            key=f"report_choice_{st.session_state.report_selector_version}",
        )
        if st.sidebar.button("Open selected report", width="stretch",
                             disabled=selected_id == active_id,
                             key="open_selected_report"):
            _request_report_change(report_id=selected_id)
        result_count = len(st.session_state.extraction_results.get(active_id, {}))
        st.sidebar.caption(
            f"{len(reports)} report(s) in session · {result_count}/2 methods run for current report"
        )
    else:
        st.sidebar.info("No report loaded. Start in Workspace.")
    st.sidebar.divider()
    st.sidebar.warning("Research prototype · Not for clinical use")
    st.sidebar.caption("Data remains in this local Streamlit session.")
    return page


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    try:
        import PyPDF2
        from io import BytesIO
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts) if text_parts else ""
    except ImportError:
        raise ReportInputError(
            "PyPDF2 is not installed. Install it with: pip install PyPDF2"
        )
    except Exception as e:
        raise ReportInputError(f"Failed to extract text from PDF: {e}")


def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a Word document (.docx)."""
    try:
        import docx
        from io import BytesIO
        doc = docx.Document(BytesIO(file_bytes))
        text_parts = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        return "\n\n".join(text_parts) if text_parts else ""
    except ImportError:
        raise ReportInputError(
            "python-docx is not installed. Install it with: pip install python-docx"
        )
    except Exception as e:
        raise ReportInputError(f"Failed to extract text from document: {e}")


def _extract_text_from_image(file_bytes: bytes) -> str:
    """Extract text from an image file using OCR (Tesseract)."""
    try:
        import pytesseract
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else ""
    except ImportError:
        raise ReportInputError(
            "OCR dependencies are not installed. Install with: "
            "pip install Pillow pytesseract\n"
            "You also need Tesseract OCR installed on your system:\n"
            "  macOS: brew install tesseract\n"
            "  Ubuntu: sudo apt install tesseract-ocr"
        )
    except Exception as e:
        raise ReportInputError(f"Failed to extract text from image (OCR): {e}")


def _extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    """Extract text from a file based on its extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return _extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        return _extract_text_from_docx(file_bytes)
    elif ext in ("jpg", "jpeg", "png"):
        return _extract_text_from_image(file_bytes)
    else:
        # Fallback: try to decode as plain text
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            raise ReportInputError(
                f"Unsupported file format: .{ext}. "
                "Supported formats: .txt, .pdf, .docx, .jpg, .jpeg, .png"
            )


def _load_prepared_report(prepared: dict[str, Any]) -> None:
    _request_report_change(report=prepared)


def _render_source_loader() -> None:
    loader_version = int(st.session_state.loader_version)
    with st.expander("Load a synthetic or approved report", expanded=_active_report() is None):
        st.caption(
            "The identifier check is a limited safeguard, not a de-identification "
            "service. Review text before loading it."
        )
        sample_tab, paste_tab, upload_tab = st.tabs(
            ["Sample report", "Paste text", "Upload file"]
        )
        with sample_tab:
            samples = st.session_state.sample_reports
            if st.session_state.sample_load_error:
                st.error("Synthetic reports could not be loaded: "
                         f"{st.session_state.sample_load_error}")
            elif not samples:
                st.info("No synthetic demonstration reports are available.")
            else:
                sample_ids = [str(item.get("report_id")) for item in samples]
                selected_id = st.selectbox(
                    "Synthetic scenario", sample_ids,
                    format_func=lambda report_id: next(
                        (f"{report_id} — {item.get('description', 'Synthetic scenario')}"
                         for item in samples if str(item.get("report_id")) == report_id),
                        report_id,
                    ),
                    key="sample_report_choice",
                )
                selected = next(item for item in samples
                                if str(item.get("report_id")) == selected_id)
                st.caption(str(selected.get("description", "")))
                if st.button("Load sample report", type="primary", width="stretch",
                             key="load_sample_report"):
                    try:
                        prepared = prepare_report(
                            str(selected.get("text", "")), source="synthetic", approved=True,
                            report_id=selected_id,
                            description=str(selected.get("description", "Synthetic demonstration report")),
                            cancer_type=selected.get("cancer_type"),
                        )
                    except ReportInputError as error:
                        st.error(str(error))
                    else:
                        _load_prepared_report(prepared)

        with paste_tab:
            paste_text_key = _loader_widget_key("pasted_report_text", loader_version)
            paste_approval_key = _loader_widget_key("paste_approved", loader_version)
            pasted_text = st.text_area(
                "Pathology report text", height=220,
                placeholder="Paste a synthetic, de-identified, or approved lung pathology report.",
                key=paste_text_key,
                on_change=_revoke_paste_approval,
                args=(loader_version,),
            )
            identifiers = find_direct_identifier_labels(pasted_text) if pasted_text.strip() else []
            if identifiers:
                st.error("Direct identifier labels detected: "
                         + ", ".join(sorted(identifiers))
                         + ". Remove or redact them before loading.")
            cancer_type = st.selectbox("Cancer type", ["Not specified", "LUAD", "LUSC"],
                                       key=_loader_widget_key("paste_cancer_type", loader_version))
            approved = st.checkbox(
                "I confirm this text is synthetic, de-identified, or approved for local research use.",
                key=paste_approval_key,
                on_change=_capture_paste_approval,
                args=(loader_version,),
            )
            if st.button("Load pasted report", type="primary", width="stretch",
                         key="load_pasted_report"):
                approval_matches_content = (
                    approved is True
                    and st.session_state.paste_approval_digest
                    == _content_digest(pasted_text)
                )
                try:
                    prepared = prepare_report(
                        pasted_text, source="pasted", approved=approval_matches_content,
                        description="Approved pasted report",
                        cancer_type=None if cancer_type == "Not specified" else cancer_type,
                    )
                except ReportInputError as error:
                    st.error(str(error))
                else:
                    _load_prepared_report(prepared)

        with upload_tab:
            upload_file_key = _loader_widget_key("uploaded_report_file", loader_version)
            upload_approval_key = _loader_widget_key("upload_approved", loader_version)
            uploaded = st.file_uploader(
                "Upload pathology report file",
                type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
                accept_multiple_files=False, key=upload_file_key,
                on_change=_revoke_upload_approval,
                args=(loader_version,),
                help="Supported formats: .txt, .pdf, .docx, .jpg, .jpeg, .png\n"
                     "Images use OCR (requires Tesseract). PDFs and .docx files are extracted automatically.",
            )
            if uploaded is not None:
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                if ext in ("jpg", "jpeg", "png"):
                    st.caption("📷 Image file detected — text will be extracted via OCR.")
                elif ext == "pdf":
                    st.caption("📄 PDF file detected — text will be extracted from all pages.")
                elif ext == "docx":
                    st.caption("📝 Word document detected — text will be extracted from paragraphs.")
                else:
                    st.caption("📃 Plain text file detected.")
            cancer_type = st.selectbox("Cancer type", ["Not specified", "LUAD", "LUSC"],
                                       key=_loader_widget_key("upload_cancer_type", loader_version))
            approved = st.checkbox(
                "I confirm this file is synthetic, de-identified, or approved for local research use.",
                key=upload_approval_key,
                on_change=_capture_upload_approval,
                args=(loader_version,),
            )
            if st.button("Load uploaded report", type="primary", width="stretch",
                         key="load_uploaded_report"):
                if uploaded is None:
                    st.error("Choose a file before loading.")
                else:
                    try:
                        uploaded_bytes = uploaded.getvalue()
                        approval_matches_content = (
                            approved is True
                            and st.session_state.upload_approval_digest
                            == _content_digest(uploaded_bytes)
                        )
                        extracted_text = _extract_text_from_file(uploaded.name, uploaded_bytes)
                        if not extracted_text.strip():
                            st.error(
                                "No text could be extracted from this file. "
                                "For images, ensure Tesseract OCR is installed and the image contains clear text."
                            )
                        else:
                            prepared = prepare_report(
                                extracted_text, source="uploaded", approved=approval_matches_content,
                                description=f"Approved upload: {uploaded.name}",
                                cancer_type=None if cancer_type == "Not specified" else cancer_type,
                            )
                    except ReportInputError as error:
                        st.error(str(error))
                    else:
                        _load_prepared_report(prepared)


def _render_case_strip(report: dict[str, Any]) -> None:
    source = str(report.get("source", "local")).replace("_", " ").title()
    cancer_type = report.get("cancer_type") or "Cancer type not specified"
    description = report.get("description") or "Local report"
    st.markdown(
        f'<div class="case-strip" aria-label="Current report">'
        f'<span class="case-id">{html.escape(str(report["report_id"]))}</span>'
        '<span class="status-pill tone-positive">Approved for local use</span>'
        f'<span class="case-meta">{html.escape(source)} · {html.escape(str(cancer_type))}</span>'
        f'<span class="case-meta">{html.escape(str(description))}</span></div>',
        unsafe_allow_html=True,
    )


def _nodal_distinction(variable: VariableExtraction) -> str | None:
    if variable.variable_name != "pathological_n_category":
        return None
    if variable.extracted_value == "pN0":
        return "Explicit pN0: the report documents pathological N0. This is not pNX or Not documented."
    if (variable.extracted_value == "pNX"
            or variable.documentation_status == DocumentationStatus.CANNOT_BE_ASSIGNED):
        return "pNX / Cannot be assigned: this is not pN0 or Not documented."
    if variable.documentation_status == DocumentationStatus.NOT_DOCUMENTED:
        return "Not documented: no explicit pathological N category is present. This is not pN0 or pNX."
    return None


def _focus_evidence(report_id: str, method: str, variable_name: str) -> None:
    """Open the evidence area with one field selected for visual review."""
    st.session_state.workspace_area = "Evidence Viewer"
    st.session_state.focused_variable = variable_name
    st.session_state[f"evidence_focus_{report_id}_{method}"] = variable_name


def _render_variable_card(variable: VariableExtraction) -> None:
    presentation = status_presentation(variable.documentation_status)
    value = variable.extracted_value or "No value returned"
    notes = (f'<div class="field-note">{html.escape(variable.notes)}</div>'
             if variable.notes else "")
    distinction = _nodal_distinction(variable)
    distinction_html = (f'<div class="distinction-note">{html.escape(distinction)}</div>'
                        if distinction else "")
    st.markdown(
        f'<article class="field-card"><div class="field-top"><div>'
        f'<div class="field-label">{html.escape(variable_label(variable.variable_name))}</div>'
        f'<div class="field-value">{html.escape(value)}</div></div>'
        f'{_status_badge(variable.documentation_status)}</div>'
        f'<div class="field-note">{html.escape(presentation.description)}</div>'
        f'{notes}{distinction_html}</article>',
        unsafe_allow_html=True,
    )


def _render_priority(result: ExtractionResult) -> None:
    review_text = ("Manual review required" if result.manual_review_required
                   else "Routine reviewer verification")
    st.markdown(
        f'<div class="priority-strip" role="status"><div class="field-top">'
        f'<div><strong>Overall review priority</strong><br>'
        f'<span class="case-meta">{html.escape(review_text)}</span></div>'
        f'{_priority_badge(result.review_priority)}</div>'
        f'<p>{html.escape(result.review_priority_reason)}</p></div>',
        unsafe_allow_html=True,
    )


def _render_empty_result(method: str) -> None:
    st.markdown(
        f'<div class="empty-panel">No {html.escape(_method_label(method))} result is '
        'available. Run extraction to populate this area.</div>',
        unsafe_allow_html=True,
    )


def _render_evidence_excerpt(text: str, start: int, end: int) -> None:
    st.markdown(
        f'<div class="offset-label">Characters {start}–{end} (end exclusive)</div>'
        f'<div class="evidence-excerpt">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _render_ai_abstraction(result: ExtractionResult | None, method: str) -> None:
    st.markdown('<div class="section-heading">AI Abstraction</div>', unsafe_allow_html=True)
    if result is None:
        _render_empty_result(method)
        return
    st.caption(
        "Four schema-validated fields. A missing value is an explicit abstention, "
        "not a zero or inferred category."
    )
    for variable in result.variables:
        _render_variable_card(variable)
        if variable.evidence:
            st.button(
                f"Open exact evidence for {variable_label(variable.variable_name)}",
                key=(f"open_evidence_abstraction_{result.report_id}_{result.method}_"
                     f"{variable.variable_name}"),
                on_click=_focus_evidence,
                args=(result.report_id, result.method, variable.variable_name),
            )


def _render_evidence_viewer(report: dict[str, Any],
                            result: ExtractionResult | None) -> None:
    st.markdown('<div class="section-heading">Evidence Viewer</div>', unsafe_allow_html=True)
    variables = result.variables if result else []
    focus_options = ["all"] + [v.variable_name for v in variables if v.evidence]
    current_focus = st.session_state.focused_variable
    if current_focus not in focus_options:
        current_focus = "all"
    focused = st.selectbox(
        "Highlight field", focus_options, index=focus_options.index(current_focus),
        format_func=lambda item: "All evidence" if item == "all" else variable_label(item),
        key=f"evidence_focus_{report['report_id']}_{result.method if result else 'none'}",
    )
    st.session_state.focused_variable = focused
    evidence_variables = [v for v in variables if v.evidence]
    if evidence_variables:
        links = "".join(
            f'<a href="#{evidence_anchor(v.variable_name)}">'
            f'{html.escape(variable_label(v.variable_name))}</a>'
            for v in evidence_variables
        )
        st.markdown(
            f'<nav class="evidence-nav" aria-label="Evidence anchors">{links}</nav>',
            unsafe_allow_html=True,
        )
    elif result:
        st.info("This extraction returned no valid evidence spans.")
    else:
        st.info("Run extraction to add exact-offset evidence highlighting.")
    highlighted = render_highlighted_report(
        str(report["text"]), variables, None if focused == "all" else focused,
    )
    st.markdown(
        f'<div class="report-reader" tabindex="0" '
        f'aria-label="Pathology report with evidence highlights">{highlighted}</div>',
        unsafe_allow_html=True,
    )
    if not result:
        return
    st.markdown("#### Exact evidence spans")
    for variable in result.variables:
        spans = variable.evidence or []
        with st.expander(
            f"{variable_label(variable.variable_name)} · {len(spans)} span(s)",
            expanded=focused == variable.variable_name,
        ):
            if not spans:
                st.caption("No evidence span accompanies this abstention.")
            for evidence in spans:
                _render_evidence_excerpt(
                    evidence.text, evidence.start_offset, evidence.end_offset
                )


def _render_documentation_issues(result: ExtractionResult | None, method: str) -> None:
    st.markdown(
        '<div class="section-heading">Documentation Issues</div>',
        unsafe_allow_html=True,
    )
    if result is None:
        _render_empty_result(method)
        return
    issues = result.qa_issues
    if not issues:
        st.success("No documentation issues are recorded for this extraction output.")
        return
    severity_order = ["critical", "high", "medium", "low"]
    available_severities = [
        severity for severity in severity_order
        if any(issue.severity == severity for issue in issues)
    ]
    available_types = sorted({issue.issue_type.value for issue in issues})
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        severity_filter = st.multiselect(
            "Severity", available_severities, format_func=str.title,
            key=f"issue_severity_{result.report_id}_{result.method}",
        )
    with filter_col2:
        type_filter = st.multiselect(
            "Issue type", available_types,
            format_func=lambda item: item.replace("_", " ").title(),
            key=f"issue_type_{result.report_id}_{result.method}",
        )
    filtered = [
        issue for issue in issues
        if (not severity_filter or issue.severity in severity_filter)
        and (not type_filter or issue.issue_type.value in type_filter)
    ]
    st.caption(f"Showing {len(filtered)} of {len(issues)} issue(s).")
    if not filtered:
        st.info("No issues match the selected filters.")
        return
    for issue in filtered:
        affected = variable_label(issue.variable_name) if issue.variable_name else "Report-level"
        title = (
            f"{issue.issue_type.value.replace('_', ' ').title()} · "
            f"{issue.severity.title()} · {affected}"
        )
        with st.expander(title):
            st.write(issue.description)
            if issue.suggestion:
                st.caption(f"Reviewer guidance: {issue.suggestion}")
            for evidence in issue.evidence or []:
                _render_evidence_excerpt(
                    evidence.text, evidence.start_offset, evidence.end_offset
                )


def _render_review_field(
    variable: VariableExtraction,
    decision: dict[str, Any],
    scope: str,
    report_id: str,
    method: str,
) -> None:
    action = str(decision.get("action", ReviewAction.PENDING.value))
    if action not in ACTION_OPTIONS:
        action = ReviewAction.PENDING.value
    with st.container(border=True):
        st.markdown(f"### {variable_label(variable.variable_name)}")
        value_col, status_col = st.columns([1.25, 1])
        with value_col:
            st.caption("Proposed value")
            st.write(variable.extracted_value or "No value returned")
        with status_col:
            st.caption("Proposed QA status")
            st.markdown(_status_badge(variable.documentation_status), unsafe_allow_html=True)
        action = st.selectbox(
            "Reviewer decision", ACTION_OPTIONS, index=ACTION_OPTIONS.index(action),
            format_func=_action_label,
            key=f"review_action_{scope}_{variable.variable_name}",
        )
        decision["action"] = action
        if action == ReviewAction.CORRECTED.value:
            corrected_status = str(
                decision.get("corrected_status") or variable.documentation_status.value
            )
            if corrected_status not in STATUS_OPTIONS:
                corrected_status = DocumentationStatus.SUPPORTED.value
            corrected_status = st.selectbox(
                "Corrected QA status", STATUS_OPTIONS,
                index=STATUS_OPTIONS.index(corrected_status), format_func=_status_label,
                key=f"corrected_status_{scope}_{variable.variable_name}",
            )
            conflicting_alternatives: list[str] = []
            if corrected_status == DocumentationStatus.CONFLICTING.value:
                corrected_value = ""
                saved_alternatives = decision.get("conflicting_alternatives")
                if not isinstance(saved_alternatives, list):
                    saved_alternatives = []
                st.caption(
                    "No single value is returned for a conflict. Enter two distinct "
                    "alternatives and pair each with its exact evidence below."
                )
                for alternative_index in range(2):
                    conflicting_alternatives.append(
                        st.text_input(
                            f"Conflicting alternative {alternative_index + 1}",
                            value=(
                                str(saved_alternatives[alternative_index])
                                if alternative_index < len(saved_alternatives) else ""
                            ),
                            key=(f"conflicting_alternative_{scope}_{variable.variable_name}_"
                                 f"{alternative_index}"),
                        )
                    )
            else:
                corrected_value = st.text_input(
                    "Corrected value (leave empty for an abstention status)",
                    value=str(decision.get("corrected_value") or ""),
                    key=f"corrected_value_{scope}_{variable.variable_name}",
                )
            saved_evidence = decision.get("corrected_evidence_texts")
            if not isinstance(saved_evidence, list):
                saved_evidence = []
            evidence_slot_count = (
                2 if corrected_status == DocumentationStatus.CONFLICTING.value else 1
            )
            corrected_evidence: list[str] = []
            if evidence_slot_count == 2:
                st.caption(
                    "Conflicting status requires two distinct exact spans showing the alternatives."
                )
            for evidence_index in range(evidence_slot_count):
                evidence_label = (
                    f"Corrected exact evidence {evidence_index + 1}"
                    if evidence_slot_count > 1 else "Corrected exact evidence"
                )
                corrected_evidence.append(
                    st.text_area(
                        evidence_label,
                        value=(
                            str(saved_evidence[evidence_index])
                            if evidence_index < len(saved_evidence) else ""
                        ),
                        height=100,
                        help=("Paste an exact, uniquely occurring substring from the loaded report. "
                              "Evidence is required for evidence-bearing statuses."),
                        key=(f"corrected_evidence_{scope}_{variable.variable_name}_"
                             f"{evidence_index}"),
                    )
                )
            reason = st.text_area(
                "Reason for correction (required)",
                value=str(decision.get("reason") or ""), height=90,
                key=f"review_reason_{scope}_{variable.variable_name}",
            )
            decision.update(corrected_value=corrected_value,
                            corrected_status=corrected_status,
                            conflicting_alternatives=conflicting_alternatives,
                            corrected_evidence_texts=corrected_evidence,
                            reason=reason)
        elif action in {ReviewAction.REJECTED.value, ReviewAction.FLAGGED.value}:
            behavior = (
                "Reject withholds the value and records Unsupported."
                if action == ReviewAction.REJECTED.value
                else "Flag withholds the value and records Manual review required."
            )
            st.caption(behavior)
            reason = st.text_area(
                f"Reason for {_action_label(action).lower()} (required)",
                value=str(decision.get("reason") or ""), height=90,
                key=f"review_reason_{scope}_{variable.variable_name}",
            )
            decision["reason"] = reason
        elif action == ReviewAction.ACCEPTED.value:
            st.caption("Accept preserves the proposed value, QA status, and evidence.")
        else:
            st.caption("Choose Accept, Correct, Reject, or Flag to review this field.")
        if variable.evidence:
            first = variable.evidence[0]
            st.button(
                f"Open exact evidence · characters {first.start_offset}–{first.end_offset}",
                key=f"open_evidence_review_{scope}_{variable.variable_name}",
                on_click=_focus_evidence,
                args=(report_id, method, variable.variable_name),
            )


def _render_human_review(
    report: dict[str, Any], result: ExtractionResult | None, method: str
) -> None:
    st.markdown('<div class="section-heading">Human Review</div>', unsafe_allow_html=True)
    if result is None:
        _render_empty_result(method)
        return
    key = _ensure_review_draft(result)
    decisions = st.session_state.review_decisions[key]
    scope = _widget_scope(result.report_id, result.method)
    progress_placeholder = st.empty()
    progress_caption = st.empty()
    for variable in result.variables:
        _render_review_field(
            variable, decisions[variable.variable_name], scope,
            result.report_id, result.method,
        )
    reviewed, total = review_progress(result, decisions)
    progress_placeholder.progress(reviewed / total if total else 0.0)
    progress_caption.caption(
        f"{reviewed} of {total} fields have a reviewer decision. "
        "Corrections, rejections, and flags require a reason."
    )
    overall_note = st.text_area(
        "Overall review note (optional)",
        value=st.session_state.review_notes.get(key, ""), height=100,
        key=f"overall_note_{scope}",
    )
    st.session_state.review_notes[key] = overall_note
    dirty = _review_is_dirty(key)
    st.caption("Unsaved review changes" if dirty else "No unsaved review changes")
    if st.button(
        "Save reviewed abstraction", type="primary", width="stretch",
        disabled=not dirty, key=f"save_review_{scope}",
    ):
        errors = validate_review_decisions(result, decisions, str(report["text"]))
        if errors:
            st.error("Complete the review before saving:")
            for error in errors:
                st.write(f"• {error}")
        else:
            try:
                snapshot, records = create_review_snapshot(
                    result, copy_decisions(decisions), report_text=str(report["text"]),
                    overall_note=overall_note,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state.review_snapshots.append(snapshot)
                st.session_state.audit_records.extend(records)
                st.session_state.review_baselines[key] = copy_decisions(decisions)
                st.session_state.review_note_baselines[key] = overall_note
                _set_flash("success", f"Saved review {snapshot.review_id} with "
                           f"{len(records)} audit records.")
                st.rerun()


def page_workspace() -> None:
    _page_heading(
        "Report Workspace",
        "Load approved text, run one extraction method, inspect exact evidence, and record reviewer decisions.",
    )
    _research_banner()
    _render_source_loader()
    report = _active_report()
    if report is None:
        st.markdown(
            '<div class="empty-panel"><strong>No report loaded.</strong><br>'
            'Open the loader above to choose a synthetic scenario, paste approved text, '
            'or upload a UTF-8 plain-text report.</div>', unsafe_allow_html=True,
        )
        return
    _render_case_strip(report)
    method_col, run_col, clear_col = st.columns([1.4, .8, .65])
    with method_col:
        method = st.selectbox(
            "Extraction method", ["evidence_first", "baseline", "ml"],
            format_func=_method_label, key="workspace_method",
            help=("Evidence-first validates exact support and may abstain. "
                  "Baseline is rule-based comparison. "
                  "ML uses a local TF-IDF + logistic regression model with evidence anchoring."),
        )
    with run_col:
        st.write("")
        if st.button("Run extraction", type="primary", width="stretch",
                     key="run_workspace_extraction"):
            _request_extraction(method)
    with clear_col:
        st.write("")
        if st.button("Clear report", width="stretch", key="request_clear_report"):
            st.session_state.show_clear_dialog = True
            st.rerun()
    result = _result_for(str(report["report_id"]), method)
    if result:
        _render_priority(result)
        st.caption("The result passed the structured pipeline contract. Reviewer verification is still required.")
    workspace_areas = (
        "AI Abstraction", "Evidence Viewer", "Documentation Issues", "Human Review",
    )
    if st.session_state.workspace_area not in workspace_areas:
        st.session_state.workspace_area = workspace_areas[0]
    workspace_area = st.radio(
        "Workspace area", workspace_areas, horizontal=True, key="workspace_area",
        help="Choose one report task area. Evidence buttons open the viewer with the field focused.",
    )
    if workspace_area == "AI Abstraction":
        _render_ai_abstraction(result, method)
    elif workspace_area == "Evidence Viewer":
        _render_evidence_viewer(report, result)
    elif workspace_area == "Documentation Issues":
        _render_documentation_issues(result, method)
    else:
        _render_human_review(report, result, method)


def _render_comparison_panel(method: str, variable: VariableExtraction) -> None:
    extra_class = (
        " method-panel--evidence" if method in {"evidence_first", "ml"} else ""
    )
    st.markdown(
        f'<div class="method-panel{extra_class}">'
        f'<div class="method-name">{html.escape(_method_label(method))}</div>'
        f'<div class="method-value">{html.escape(variable.extracted_value or "No value returned")}</div>'
        f'{_status_badge(variable.documentation_status)}</div>',
        unsafe_allow_html=True,
    )
    spans = variable.evidence or []
    if spans:
        with st.expander(f"Exact evidence · {len(spans)} span(s)"):
            for evidence in spans:
                _render_evidence_excerpt(evidence.text, evidence.start_offset,
                                         evidence.end_offset)
    else:
        st.caption("No evidence span returned.")


def page_comparison() -> None:
    _page_heading(
        "Method Comparison",
        "Inspect each value, QA status, and exact evidence side by side for the current report.",
    )
    report = _active_report()
    if report is None:
        st.info("Load a report in Workspace before comparing methods.")
        return
    _render_case_strip(report)
    report_id = str(report["report_id"])
    methods = ("baseline", "evidence_first", "ml")
    results = {method: _result_for(report_id, method) for method in methods}
    if any(result is None for result in results.values()):
        missing = [
            _method_label(method)
            for method, result in results.items()
            if result is None
        ]
        st.info("Missing comparison output: " + " and ".join(missing) + ".")
        if st.button("Run all methods", type="primary", width="stretch",
                     key="run_both_comparison"):
            _request_extraction("both")
        return
    differences = 0
    for name in CORE_VARIABLES:
        signatures = {
            (
                next(v for v in results[method].variables if v.variable_name == name).extracted_value,
                next(v for v in results[method].variables if v.variable_name == name).documentation_status,
            )
            for method in methods
        }
        if len(signatures) > 1:
            differences += 1
    st.caption(
        f"{differences} of {len(CORE_VARIABLES)} fields differ in value or QA status across methods. "
        "A difference is not itself a correctness judgment."
    )
    for name in CORE_VARIABLES:
        st.markdown(f"### {variable_label(name)}")
        columns = st.columns(3)
        for column, method in zip(columns, methods):
            variable = next(v for v in results[method].variables if v.variable_name == name)
            with column:
                _render_comparison_panel(method, variable)
        st.divider()
    priority_cols = st.columns(3)
    for column, method in zip(priority_cols, methods):
        with column:
            st.markdown(f"#### {_method_label(method)} review priority")
            _render_priority(results[method])


def _prepared_synthetic_reports() -> list[dict[str, Any]]:
    return [
        prepare_report(
            str(report.get("text", "")), source="synthetic", approved=True,
            report_id=str(report.get("report_id")),
            description=str(report.get("description", "Synthetic demonstration report")),
            cancer_type=report.get("cancer_type"),
        )
        for report in st.session_state.sample_reports
    ]


def _run_evaluation() -> tuple[ComparisonResult, int]:
    reports = _prepared_synthetic_reports()
    if not reports:
        raise ReportInputError("No synthetic demonstration reports are available.")
    baseline_results: list[ExtractionResult] = []
    evidence_results: list[ExtractionResult] = []
    report_texts: dict[str, str] = {}
    for report in reports:
        results = run_both_pipelines(report)
        baseline_results.append(results["baseline"])
        evidence_results.append(results["evidence_first"])
        report_texts[str(report["report_id"])] = str(report["text"])
    comparison = Evaluator().compare_methods(
        baseline_results, evidence_results, parse_gold_annotations(), report_texts,
    )
    return comparison, len(reports)


def page_evaluation() -> None:
    _page_heading(
        "Illustrative synthetic results",
        "Metrics are computed only from labeled synthetic reports; they are not clinical performance claims.",
    )
    _research_banner()
    if st.button("Run synthetic evaluation", type="primary", width="stretch",
                 key="run_synthetic_evaluation"):
        try:
            with st.spinner("Running both methods on the synthetic evaluation set…"):
                comparison, report_count = _run_evaluation()
        except (ReportInputError, ExtractionPipelineError, ValueError) as error:
            st.error(f"Evaluation could not be completed: {error}")
        except Exception:
            st.error("Evaluation stopped safely because an unexpected error occurred.")
        else:
            st.session_state.evaluation_comparison = comparison
            st.session_state.evaluation_run_at = datetime.now(timezone.utc).isoformat()
            _set_flash("success", f"Computed illustrative metrics on {report_count} synthetic reports.")
            st.rerun()
    comparison: ComparisonResult | None = st.session_state.evaluation_comparison
    if comparison is None:
        st.markdown(
            '<div class="empty-panel">No evaluation has been run in this session. '
            'Run the synthetic evaluation to calculate metrics from the bundled gold annotations.</div>',
            unsafe_allow_html=True,
        )
        return
    baseline = comparison.baseline_metrics
    evidence = comparison.evidence_first_metrics
    st.caption(f"Computed {st.session_state.evaluation_run_at} · "
               f"{evidence.report_count} synthetic report(s).")
    metric_specs = [
        ("Unsupported extraction rate", "unsupported_extraction_rate", "Non-abstained fields"),
        ("Precision", "overall_precision", "Returned values"),
        ("Recall", "overall_recall", "Gold values"),
        ("F1 score", "overall_f1", "Gold and returned values"),
        ("Coverage", "coverage", "Annotated fields"),
        ("Appropriate abstention rate", "appropriate_abstention_rate", "Expected abstentions"),
        ("False-positive rate · not documented", "false_positive_rate_undocumented", "Not-documented gold fields"),
        ("Documentation status accuracy", "documentation_status_accuracy", "Annotated fields"),
        ("Exact value-supporting evidence rate", "exact_evidence_rate", "Returned values"),
        ("Manual review referral rate", "manual_review_referral_rate", "Reports"),
        ("Review referral precision", "manual_review_referral_precision", "Referred reports"),
        ("Review referral sensitivity", "manual_review_referral_recall", "Gold review-needed reports"),
        ("Schema validity rate", "schema_validity_rate", "Reports"),
    ]
    def metric_basis(metrics: Any, attribute: str) -> str:
        true_positives = sum(item.true_positives for item in metrics.variable_metrics)
        false_positives = sum(item.false_positives for item in metrics.variable_metrics)
        false_negatives = sum(item.false_negatives for item in metrics.variable_metrics)
        unsupported = sum(item.unsupported_count for item in metrics.variable_metrics)
        bases = {
            "unsupported_extraction_rate": (unsupported, metrics.returned_value_count),
            "overall_precision": (true_positives, true_positives + false_positives),
            "overall_recall": (true_positives, true_positives + false_negatives),
            "coverage": (metrics.returned_value_count, metrics.fields_evaluated),
            "appropriate_abstention_rate": (
                metrics.appropriate_abstention_count, metrics.expected_abstention_count
            ),
            "false_positive_rate_undocumented": (
                metrics.undocumented_false_positive_count,
                metrics.undocumented_gold_field_count,
            ),
            "documentation_status_accuracy": (
                metrics.documentation_status_correct_count, metrics.fields_evaluated
            ),
            "exact_evidence_rate": (
                metrics.exact_evidence_correct_count, metrics.exact_evidence_total_count
            ),
            "manual_review_referral_rate": (metrics.manual_review_count, metrics.report_count),
            "manual_review_referral_precision": (
                metrics.review_referral_true_positive_count,
                metrics.review_referral_true_positive_count
                + metrics.review_referral_false_positive_count,
            ),
            "manual_review_referral_recall": (
                metrics.review_referral_true_positive_count,
                metrics.gold_manual_review_report_count,
            ),
            "schema_validity_rate": (metrics.schema_valid_count, metrics.report_count),
        }
        numerator_denominator = bases.get(attribute)
        return (f"{numerator_denominator[0]}/{numerator_denominator[1]}"
                if numerator_denominator else "Derived")
    rows = []
    for label, attribute, denominator in metric_specs:
        baseline_value = getattr(baseline, attribute)
        evidence_value = getattr(evidence, attribute)
        baseline_display = "N/A" if baseline_value is None else f"{baseline_value:.1%}"
        evidence_display = "N/A" if evidence_value is None else f"{evidence_value:.1%}"
        difference_display = (
            "N/A" if baseline_value is None or evidence_value is None
            else f"{evidence_value - baseline_value:+.1%}"
        )
        rows.append({"Metric": label, "Baseline": baseline_display,
                     "Baseline count": metric_basis(baseline, attribute),
                     "Evidence-first": evidence_display,
                     "Evidence-first count": metric_basis(evidence, attribute),
                     "Difference": difference_display,
                     "Denominator": denominator})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Differences are descriptive for this small synthetic set. Lower is better "
        "for unsupported extraction rate; coverage and review rate require context."
    )
    st.download_button(
        "Download evaluation CSV", data=evaluation_comparison_csv(comparison),
        file_name="illustrative_synthetic_evaluation.csv", mime="text/csv",
        width="stretch", key="download_evaluation_csv",
    )


def _audit_rows(records: list[AuditRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Timestamp": record.timestamp, "Report": record.report_id,
         "Method": _method_label(record.method),
         "Field": variable_label(record.variable_name),
         "Action": _action_label(record.reviewer_action.value),
         "Original": record.original_output.extracted_value or "—",
         "Result": record.resulting_value or "—",
         "Status": status_presentation(record.resulting_status).label}
        for record in records
    ])


def _render_audit_detail(record: AuditRecord) -> None:
    title = (f"{record.timestamp} · {record.report_id} · "
             f"{variable_label(record.variable_name)} · "
             f"{_action_label(record.reviewer_action.value)}")
    with st.expander(title):
        original_col, reviewed_col = st.columns(2)
        with original_col:
            st.caption("Original output")
            st.write(record.original_output.extracted_value or "No value returned")
            st.markdown(_status_badge(record.original_output.documentation_status),
                        unsafe_allow_html=True)
        with reviewed_col:
            st.caption("Reviewed result")
            st.write(record.resulting_value or "No reviewed value")
            st.markdown(_status_badge(record.resulting_status), unsafe_allow_html=True)
        st.write(f"Reason: {record.review_reason}")
        if record.resulting_alternatives:
            st.markdown("**Reviewed conflicting alternatives**")
            for alternative in record.resulting_alternatives:
                st.write(f"• {alternative}")
        st.caption(f"Audit ID {record.audit_id} · Review ID {record.review_id} · "
                   f"{_method_label(record.method)}")
        if record.resulting_evidence:
            st.markdown("**Resulting exact evidence**")
            for evidence in record.resulting_evidence:
                _render_evidence_excerpt(evidence.text, evidence.start_offset,
                                         evidence.end_offset)
        if record.original_output.evidence:
            st.markdown("**Original exact evidence**")
            for evidence in record.original_output.evidence:
                _render_evidence_excerpt(evidence.text, evidence.start_offset,
                                         evidence.end_offset)


def page_audit_export() -> None:
    _page_heading(
        "Audit History & Export",
        "Inspect append-only session history and download reviewed abstractions with evidence and provenance.",
    )
    snapshots: list[ReviewSnapshot] = st.session_state.review_snapshots
    records: list[AuditRecord] = st.session_state.audit_records
    history_tab, export_tab = st.tabs(["Audit history", "Export"])
    with history_tab:
        st.markdown('<div class="section-heading">Audit History</div>', unsafe_allow_html=True)
        if not records:
            st.info("No audit records yet. Save a complete review in Workspace to create them.")
        else:
            report_options = sorted({record.report_id for record in records})
            method_options = sorted({record.method for record in records})
            action_options = sorted({record.reviewer_action.value for record in records})
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                report_filter = st.multiselect("Report", report_options,
                                               key="audit_report_filter")
            with filter_col2:
                method_filter = st.multiselect("Method", method_options,
                                               format_func=_method_label,
                                               key="audit_method_filter")
            with filter_col3:
                action_filter = st.multiselect("Reviewer action", action_options,
                                               format_func=_action_label,
                                               key="audit_action_filter")
            filtered = [record for record in records
                        if (not report_filter or record.report_id in report_filter)
                        and (not method_filter or record.method in method_filter)
                        and (not action_filter or record.reviewer_action.value in action_filter)]
            filtered.sort(key=lambda item: item.timestamp, reverse=True)
            st.caption(f"Showing {len(filtered)} of {len(records)} audit record(s).")
            if filtered:
                st.dataframe(_audit_rows(filtered), hide_index=True, width="stretch")
                st.markdown("#### Record details")
                for record in filtered:
                    _render_audit_detail(record)
            else:
                st.info("No audit records match the selected filters.")
    with export_tab:
        st.markdown('<div class="section-heading">Reviewed results</div>', unsafe_allow_html=True)
        if snapshots:
            st.caption(f"{len(snapshots)} saved review snapshot(s) · "
                       f"{len(records)} audit record(s).")
            json_col, csv_col, audit_col = st.columns(3)
            with json_col:
                st.download_button(
                    "Reviewed JSON", data=reviewed_results_json(snapshots, records),
                    file_name="reviewed_abstractions.json", mime="application/json",
                    width="stretch", key="download_reviewed_json",
                )
            with csv_col:
                st.download_button(
                    "Reviewed CSV", data=reviewed_results_csv(snapshots),
                    file_name="reviewed_abstractions.csv", mime="text/csv",
                    width="stretch", key="download_reviewed_csv",
                )
            with audit_col:
                st.download_button(
                    "Audit CSV", data=audit_history_csv(records),
                    file_name="audit_history.csv", mime="text/csv",
                    width="stretch", key="download_audit_csv",
                )
        else:
            st.info("No reviewed abstractions are available for export.")
        st.divider()
        st.markdown('<div class="section-heading">Unreviewed extraction</div>',
                    unsafe_allow_html=True)
        report = _active_report()
        if report is None:
            st.caption("Load and extract a report to download an unreviewed result.")
        else:
            report_id = str(report["report_id"])
            available = st.session_state.extraction_results.get(report_id, {})
            if not available:
                st.caption("Run extraction in Workspace before downloading an unreviewed result.")
            else:
                method = st.selectbox(
                    "Extraction to download", list(available), format_func=_method_label,
                    key="unreviewed_export_method",
                )
                st.download_button(
                    "Download unreviewed JSON", data=extraction_result_json(available[method]),
                    file_name=(f"{_safe_filename(report_id)}-"
                               f"{_safe_filename(method)}-unreviewed.json"),
                    mime="application/json", width="stretch",
                    key="download_unreviewed_json",
                )


def _is_authenticated() -> bool:
    """Check if the user is authenticated."""
    return bool(st.session_state["authenticated"]) if "authenticated" in st.session_state else False


def _render_login_page() -> None:
    """Render the login page with authentication form."""
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 420px;
            margin: 80px auto;
            padding: 2.5rem 2rem;
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            box-shadow: 0 4px 24px rgba(0,0,0,.06);
        }
        .login-heading {
            text-align: center;
            margin-bottom: 2rem;
        }
        .login-heading h1 {
            font-size: 2.4rem;
            margin-bottom: .5rem;
        }
        .login-heading p {
            color: var(--muted);
            font-size: .88rem;
        }
        .login-form label {
            font-weight: 650;
            color: var(--navy);
            margin-bottom: .35rem;
        }
        .login-form input {
            border-radius: 8px;
            border: 1px solid var(--line);
            padding: .6rem .75rem;
            font-size: .92rem;
            width: 100%;
        }
        .login-error {
            background: var(--red-bg);
            border: 1px solid #edc4c4;
            border-radius: 8px;
            color: var(--red);
            padding: .6rem .75rem;
            font-size: .84rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        .login-footer {
            text-align: center;
            margin-top: 1.5rem;
            font-size: .78rem;
            color: var(--muted);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="login-container">'
        '<div class="login-heading">'
        '<h1>OncoExtract</h1>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Show error message if login failed
    if st.session_state.get("login_error"):
        st.markdown(
            f'<div class="login-error">{html.escape(str(st.session_state.login_error))}</div>',
            unsafe_allow_html=True,
        )

    with st.form("login_form", clear_on_submit=True):
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

        if submitted:
            if (
                username == DEFAULT_USERNAME
                and hashlib.sha256(password.encode()).hexdigest()
                == hashlib.sha256(DEFAULT_PASSWORD.encode()).hexdigest()
            ):
                st.session_state.authenticated = True
                st.session_state.login_error = None
                st.session_state.current_user = username
                st.rerun()
            else:
                st.session_state.login_error = "Invalid username or password. Please try again."
                st.rerun()

    st.markdown(
        '<div class="login-footer">'
        'Research prototype · Not for clinical use<br>'
        'Default credentials: admin / admin123'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    _initialize_state()
    if not _is_authenticated():
        _render_login_page()
        return
    page = _render_sidebar()
    _show_flash_message()
    if page == "Workspace":
        page_workspace()
    elif page == "Comparison":
        page_comparison()
    elif page == "Evaluation":
        page_evaluation()
    elif page == "Audit & Export":
        page_audit_export()
    if st.session_state.pending_report_change:
        _confirm_report_change_dialog()
    elif st.session_state.pending_rerun_method:
        _confirm_rerun_dialog()
    elif st.session_state.show_clear_dialog:
        _confirm_clear_dialog()


if __name__ == "__main__":
    main()
