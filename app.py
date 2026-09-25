"""Production-quality Streamlit UI for the OncoExtractAI-QA research prototype.

The UI owns presentation and in-session workflow state only. Validated domain
modules handle report preparation, extraction, review snapshots, and exports.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from datetime import datetime, timezone
from statistics import median
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
    sanitize_report_text,
)
from src.pdf_processor import (
    PDFProcessingError,
    assemble_accepted_text,
    process_pdf,
    render_pdf_page,
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
    ProcessedDocument,
    ReviewAction,
    VariableExtraction,
)
from src.utils import load_synthetic_reports, parse_gold_annotations
from src.workflow import (
    ExtractionPipelineError,
    run_both_pipelines,
    run_extraction_pipeline,
)


st.set_page_config(
    page_title="OncoExtract | Pathology review",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #16324f; --teal: #087e8b; --canvas: #f7f9fb;
        --paper: #fff; --text: #1d2939; --muted: #667085;
        --line: #e2e8f0; --soft: #f1f5f9; --radius: 10px;
        --green: #246b45; --green-bg: #e5f4ea;
        --amber: #7a4b00; --amber-bg: #fff1d6;
        --red: #8f2929; --red-bg: #fbe7e7;
        --accent: #087e8b; --accent-light: #e7f5f6;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--canvas); }
    .block-container { max-width: 1320px; padding-top: 2.25rem; padding-bottom: 4rem; }
    
    /* Quiet application rail: navigation remains distinct without competing with work. */
    [data-testid="stSidebar"] { background: #102a43; border-right: 0; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #f2f7fa; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        min-height: 44px; border-radius: 10px; padding: .4rem .6rem;
        margin: 2px 0;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: rgba(255,255,255,.08);
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] .stCaption p { color: inherit; }
    
    /* Profile card in sidebar */
    [data-testid="stSidebar"] .sidebar-profile {
        background: rgba(255,255,255,.14) !important;
        border: 1px solid rgba(255,255,255,.28) !important;
        border-radius: 12px !important;
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
    
    /* Sidebar buttons */
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
    
    /* Typography */
    h1, h2, h3 { color: var(--navy); letter-spacing: -.02em; }
    h1 { font-size: clamp(1.75rem, 3vw, 2.25rem); }
    h2 { font-size: clamp(1.3rem, 2vw, 1.6rem); }
    p, label, li, button, input, textarea, select { font-size: .94rem; }
    small, .stCaption p { font-size: .78rem !important; }
    :focus-visible { outline: 3px solid #18a3ad !important; outline-offset: 2px !important; }
    a { color: #0b6570; } a:hover { color: #084b53; }
    
    /* Page headings */
    .page-heading { margin-bottom: 1.35rem; }
    .page-heading:before { color: var(--teal); content: "ONCOEXTRACT WORKSPACE";
        display: block; font-size: .7rem; font-weight: 750; letter-spacing: .12em; margin-bottom: .45rem; }
    .page-heading h1 { font-size: 2rem; margin: 0 0 .35rem; line-height: 1.15; }
    .page-heading p { color: var(--muted); margin: 0; max-width: 880px; line-height: 1.55; }
    .section-heading { color: var(--navy); font-size: 1rem; font-weight: 750; margin: 0 0 .55rem; }
    
    /* Research banner */
    .research-banner {
        background: #fff8e8; border: 1px solid #ead29b; border-left: 4px solid #a56b05;
        border-radius: var(--radius); color: #624308; font-size: .86rem;
        line-height: 1.45; margin: 0 0 1rem; padding: .7rem .85rem;
    }
    
    /* Compact operational summary */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        background: var(--paper);
        border: 1px solid var(--line);
        border-radius: var(--radius);
        padding: 1rem 1.1rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        box-shadow: 0 1px 2px rgba(16,42,67,.04);
    }
    .stat-card:hover { border-color: #b9c8d5; }
    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--teal); font-size: 1.1rem;
        flex-shrink: 0;
    }
    .stat-icon-blue { background: var(--accent-light); }
    .stat-icon-green { background: var(--green-bg); }
    .stat-icon-amber { background: var(--amber-bg); }
    .stat-icon-red { background: var(--red-bg); }
    .stat-content { flex: 1; }
    .stat-label { color: var(--muted); font-size: .82rem; font-weight: 600; margin: 0 0 .25rem; }
    .stat-value { color: var(--navy); font-size: 1.45rem; font-weight: 750; margin: 0; line-height: 1.1; }
    .stat-trend { font-size: .78rem; font-weight: 600; margin: .25rem 0 0; }
    .stat-trend-up { color: var(--green); }
    .stat-trend-down { color: var(--red); }
    
    /* Case strip */
    .case-strip, .priority-strip {
        background: var(--paper); border: 1px solid var(--line); border-radius: var(--radius);
        margin: .75rem 0 1rem; padding: .8rem .95rem;
    }
    .case-strip { align-items: center; display: flex; flex-wrap: wrap; gap: .55rem 1rem; }
    .case-id { color: var(--navy); font-weight: 750; overflow-wrap: anywhere; }
    .case-meta { color: var(--muted); font-size: .8rem; overflow-wrap: anywhere; }
    .priority-strip p { color: #40586b; font-size: .84rem; line-height: 1.5; margin: .35rem 0 0; }
    
    /* Status pills */
    .status-pill, .priority-pill {
        align-items: center; border: 1px solid transparent; border-radius: 999px;
        display: inline-flex; font-size: .78rem; font-weight: 750; gap: .3rem; line-height: 1.3;
        max-width: 100%; min-height: 26px; padding: .28rem .58rem; white-space: normal;
    }
    .tone-positive { background: var(--green-bg); color: var(--green); border-color: #bfdfca; }
    .tone-neutral { background: var(--soft); color: #405465; border-color: #d7e0e6; }
    .tone-warning { background: var(--amber-bg); color: var(--amber); border-color: #ecd69e; }
    .tone-danger { background: var(--red-bg); color: var(--red); border-color: #edc4c4; }
    
    /* Field cards */
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
    .triage-label { align-items: center; border-radius: 7px; display: flex; font-size: .8rem;
        font-weight: 700; gap: .4rem; line-height: 1.4; margin: 0 0 .75rem; padding: .5rem .65rem; }
    .triage-label--exception { background: var(--amber-bg); border: 1px solid #dfc27e; color: #684000; }
    .triage-label--routine { background: var(--green-bg); border: 1px solid #acd2b9; color: #205f3e; }
    .triage-reasons { color: var(--muted); font-size: .8rem; margin: -.35rem 0 .75rem; }
    
    /* Evidence navigation */
    .anchor-link { display: inline-block; font-size: .8rem; font-weight: 650; margin-top: .55rem; }
    .review-field-link { color: var(--navy); text-decoration: none; }
    .review-field-link:hover { color: var(--teal); text-decoration: underline; }
    .review-field-link:focus-visible { border-radius: 4px; outline: 3px solid #18a3ad; outline-offset: 3px; }
    .evidence-nav { align-items: center; display: flex; flex-wrap: wrap;
        gap: .45rem; margin: .4rem 0 .75rem; }
    .evidence-nav a { background: var(--paper); border: 1px solid var(--line);
        border-radius: 7px; font-size: .78rem; font-weight: 650;
        padding: .35rem .55rem; text-decoration: none; }
    
    /* Report reader */
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
    .technical-details { color: var(--muted); font-size: .78rem; margin: -.4rem 0 .75rem; }
    .technical-details summary { cursor: pointer; font-weight: 650; padding: .2rem 0; }
    .technical-details summary:focus-visible { outline: 3px solid #18a3ad; outline-offset: 2px; }
    
    /* Empty panels */
    .empty-panel { background: var(--paper); border: 1px dashed #b8c7d2;
        border-radius: var(--radius); color: var(--muted); line-height: 1.5; padding: 1.25rem; }
    .review-card h3 { font-size: 1rem; margin: 0 0 .15rem; }
    .method-panel { min-height: 150px; }
    .method-panel--evidence { border-top: 3px solid var(--teal); }
    .method-name { color: var(--navy); font-weight: 750; }
    .method-value { font-size: .98rem; font-weight: 680; margin: .55rem 0; overflow-wrap: anywhere; }
    
    /* Status legend */
    .status-legend { display: grid; gap: .55rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
    .legend-item { background: var(--paper); border: 1px solid var(--line);
        border-radius: var(--radius); padding: .75rem; }
    .legend-item p { color: var(--muted); font-size: .8rem; line-height: 1.4; margin: .4rem 0 0; }
    
    /* Streamlit component overrides */
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
    [data-testid="stExpander"] { background: var(--paper); border-color: var(--line); border-radius: var(--radius); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .25rem; overflow-x: auto; }
    [data-testid="stTabs"] [data-baseweb="tab"] { min-height: 44px; white-space: nowrap; }
    div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
        border-radius: 8px; min-height: 40px; font-weight: 650;
    }
    #MainMenu, footer { visibility: hidden; }
    
    /* Welcome greeting */
    .welcome-greeting {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 .25rem;
    }
    .welcome-subtitle {
        color: var(--muted);
        font-size: .95rem;
        margin: 0 0 1rem;
    }

    .workflow-steps { align-items: center; display: flex; flex-wrap: wrap; gap: .45rem;
        margin: -.25rem 0 1.25rem; }
    .workflow-step { align-items: center; background: #fff; border: 1px solid var(--line);
        border-radius: 999px; color: var(--muted); display: inline-flex; font-size: .78rem;
        font-weight: 650; gap: .35rem; padding: .38rem .7rem; }
    .workflow-step--active { background: var(--accent-light); border-color: #9fd2d6; color: #075f68; }
    .workflow-step--done { background: var(--green-bg); border-color: #bfdfca; color: var(--green); }
    .workflow-arrow { color: #98a2b3; font-size: .8rem; }
    
    @media (max-width: 820px) {
        .block-container { padding: 1rem .75rem 2rem; }
        [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
        [data-testid="column"] { flex: 1 1 280px !important;
            min-width: min(100%,280px) !important; width: 100% !important; }
        .report-reader { max-height: 55vh; min-height: 300px; }
        .status-legend { grid-template-columns: 1fr; }
        .stats-grid { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DAILY_NAVIGATION = ("Workspace", "Comparison", "Audit & Export")
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


_REVIEW_SHORTCUTS = st.components.v2.component(
    "oncoextract_review_shortcuts",
    html="""
<div class="shortcut-list" role="note" aria-label="Review keyboard shortcuts">
  <span><kbd>A</kbd> Accept</span><span><kbd>C</kbd> Correct</span>
  <span><kbd>F</kbd> Flag</span><span><kbd>J</kbd>/<kbd>K</kbd> Field</span>
  <span><kbd>E</kbd> Evidence</span><span><kbd>⌘/Ctrl</kbd>+<kbd>Enter</kbd> Complete</span>
</div>
""",
    css="""
.shortcut-list { color: var(--st-text-color); display: flex; flex-wrap: wrap;
  font: 0.78rem var(--st-font); gap: .45rem .8rem; padding: .2rem 0 .55rem; }
.shortcut-list span { align-items: center; display: inline-flex; gap: .25rem; }
kbd { background: var(--st-secondary-background-color); border: 1px solid
  color-mix(in srgb, var(--st-text-color) 25%, transparent); border-radius: .3rem;
  box-shadow: 0 1px 0 color-mix(in srgb, var(--st-text-color) 18%, transparent);
  font: inherit; font-weight: 700; min-width: 1.45rem; padding: .08rem .3rem;
  text-align: center; }
""",
    js="""
export default function (component) {
  const { setTriggerValue } = component
  const handler = (event) => {
    const target = event.target
    const isEditing = target && (target.matches?.("input, textarea, select") || target.isContentEditable)
    const complete = event.key === "Enter" && (event.metaKey || event.ctrlKey)
    if (event.defaultPrevented || event.repeat || (isEditing && !complete)) return
    let action = null
    if (complete) action = "complete"
    else if (!event.metaKey && !event.ctrlKey && !event.altKey) {
      action = ({a: "accept", c: "correct", f: "flag", j: "next",
                 k: "previous", e: "evidence"})[event.key.toLowerCase()] ?? null
    }
    if (!action) return
    event.preventDefault()
    setTriggerValue("shortcut", {action, nonce: Date.now()})
  }
  window.addEventListener("keydown", handler)
  return () => window.removeEventListener("keydown", handler)
}
""",
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


def _load_pasted_from_state(version: int) -> None:
    """Validate and load pasted text in a pre-render widget callback."""
    text = str(st.session_state.get(_loader_widget_key("pasted_report_text", version), ""))
    approved = st.session_state.get(_loader_widget_key("paste_approved", version)) is True
    cancer_type = st.session_state.get(
        _loader_widget_key("paste_cancer_type", version), "Not specified"
    )
    approval_matches = approved and st.session_state.paste_approval_digest == _content_digest(text)
    try:
        prepared = prepare_report(
            text, source="pasted", approved=approval_matches,
            description="Approved pasted report",
            cancer_type=None if cancer_type == "Not specified" else cancer_type,
        )
    except ReportInputError as error:
        _set_flash("error", str(error))
    else:
        _commit_report(prepared)


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
        "nav_page": "Workspace", "daily_navigation": "Workspace",
        "loaded_reports": {}, "active_report_id": None,
        "extraction_results": {}, "review_decisions": {}, "review_baselines": {},
        "review_notes": {}, "review_note_baselines": {}, "review_draft_versions": {},
        "review_draft_saved_at": {}, "review_started_at": {},
        "review_returned_for_clarification": {},
        "case_assignments": {}, "queue_methods": {},
        "review_snapshots": [], "audit_records": [], "evaluation_comparison": None,
        "evaluation_run_at": None, "focused_variable": "all",
        "pending_report_change": None, "pending_rerun_method": None,
        "show_clear_dialog": False,
        "report_selector_version": 0, "flash_message": None,
        "workspace_method": "evidence_first",
        "workspace_area": "Extraction results",
        "queue_filter": "All", "reader_mode": False,
        "keyboard_active_fields": {}, "keyboard_save_requests": {},
        "loader_version": 0,
        "paste_approval_digest": None,
        "upload_approval_digest": None,
        "pending_pdf_documents": {}, "pending_pdf_payloads": {},
        "pdf_active_page": {}, "pdf_editor_versions": {},
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


def _workflow_steps(*, has_report: bool, has_result: bool, has_review: bool,
                    has_pending_pdf: bool = False) -> None:
    """Show the user's current place in the review workflow."""
    states = (
        ("1", "Upload", has_report or has_pending_pdf),
        ("2", "OCR & text review", has_report),
        ("3", "Abstraction", has_result),
        ("4", "QA & human review", has_review),
        ("5", "Export", has_review),
    )
    first_incomplete = next(
        (index for index, (_, _, done) in enumerate(states) if not done), len(states) - 1
    )
    parts: list[str] = []
    for index, (number, label, done) in enumerate(states):
        state_class = "workflow-step--done" if done else (
            "workflow-step--active" if index == first_incomplete else ""
        )
        marker = "✓" if done else number
        parts.append(
            f'<span class="workflow-step {state_class}"><strong>{marker}</strong> '
            f'{html.escape(label)}</span>'
        )
        if index < len(states) - 1:
            parts.append('<span class="workflow-arrow">›</span>')
    st.markdown(f'<div class="workflow-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


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
    icons = {"positive": "✓", "neutral": "—", "warning": "!", "danger": "⚠"}
    icon = icons.get(presentation.tone, "•")
    return (f'<span class="status-pill tone-{html.escape(presentation.tone)}" '
            f'role="status" aria-label="{html.escape(presentation.label, quote=True)}" '
            f'title="{html.escape(presentation.description, quote=True)}">'
            f'<span aria-hidden="true">{icon}</span> {html.escape(presentation.label)}</span>')


def _priority_badge(priority: str) -> str:
    tone = PRIORITY_TONES.get(priority, "neutral")
    icon = {"positive": "✓", "neutral": "—", "warning": "!", "danger": "⚠"}.get(
        tone, "•"
    )
    label = f"{priority.title()} priority"
    return (f'<span class="priority-pill tone-{tone}" role="status" '
            f'aria-label="{html.escape(label, quote=True)}">'
            f'<span aria-hidden="true">{icon}</span> {html.escape(label)}</span>')


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


def _queue_method(report_id: str) -> str:
    available = st.session_state.extraction_results.get(report_id, {})
    preferred = st.session_state.queue_methods.get(report_id)
    if preferred in available:
        return str(preferred)
    if "evidence_first" in available:
        return "evidence_first"
    if available:
        return next(iter(available))
    return str(preferred or "evidence_first")


def _case_status(report_id: str, method: str) -> str:
    if any(
        snapshot.report_id == report_id and snapshot.method == method
        for snapshot in st.session_state.review_snapshots
    ):
        return "Completed"
    result = _result_for(report_id, method)
    if result is None:
        return "Needs review"
    key = _draft_key(report_id, method)
    decisions = st.session_state.review_decisions.get(key)
    if decisions:
        reviewed, _ = review_progress(result, decisions)
        if reviewed or _review_is_dirty(key):
            return "In progress"
    return "Needs review"


def _queue_report_ids(status_filter: str = "All") -> list[str]:
    report_ids = list(st.session_state.loaded_reports)
    if status_filter == "All":
        return report_ids
    return [
        report_id for report_id in report_ids
        if _case_status(report_id, _queue_method(report_id)) == status_filter
    ]


def _render_case_queue() -> None:
    reports = st.session_state.loaded_reports
    if not reports:
        return
    st.subheader(":material/format_list_bulleted: Review queue")
    status_filter = st.radio(
        "Case queue filter",
        ["All", "Needs review", "In progress", "Completed"],
        horizontal=True,
        key="queue_filter",
    )
    filtered_ids = _queue_report_ids(status_filter)
    rows: list[dict[str, str]] = []
    for report_id in filtered_ids:
        method = _queue_method(report_id)
        result = _result_for(report_id, method)
        rows.append({
            "Case": report_id,
            "Status": _case_status(report_id, method),
            "Priority": result.review_priority.title() if result else "Not assessed",
            "Method": _method_label(method),
            "Assigned reviewer": st.session_state.case_assignments.get(
                report_id, "Unassigned"
            ),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info(
            f"No cases match the {status_filter.lower()} filter. Choose another status "
            "filter or load a report below."
        )

    active_id = st.session_state.active_report_id
    navigation_ids = filtered_ids or list(reports)
    active_index = navigation_ids.index(active_id) if active_id in navigation_ids else 0
    previous_col, chooser_col, next_col = st.columns(
        [.65, 2, .65], vertical_alignment="bottom"
    )
    with previous_col:
        if st.button(
            "Previous", icon=":material/arrow_back:", width="stretch",
            disabled=not navigation_ids or active_index == 0, key="queue_previous",
        ):
            _activate_report(navigation_ids[active_index - 1])
            st.rerun()
    with chooser_col:
        chosen = st.selectbox(
            "Open case", navigation_ids,
            index=active_index if navigation_ids else None,
            placeholder="No matching cases",
            key="queue_case_choice",
        )
    with next_col:
        if st.button(
            "Next", icon=":material/arrow_forward:", width="stretch",
            disabled=not navigation_ids or active_index >= len(navigation_ids) - 1,
            key="queue_next",
        ):
            _activate_report(navigation_ids[active_index + 1])
            st.rerun()
    if chosen and chosen != active_id:
        _activate_report(chosen)
        st.rerun()


def _purge_report_drafts(report_id: str) -> None:
    prefix = f"{report_id}::"
    for name in ("review_decisions", "review_baselines", "review_notes",
                 "review_note_baselines", "review_draft_versions",
                 "review_draft_saved_at", "review_started_at",
                 "review_returned_for_clarification", "keyboard_active_fields",
                 "keyboard_save_requests"):
        collection = st.session_state[name]
        for key in [item for item in collection if item.startswith(prefix)]:
            collection.pop(key, None)


def _clear_pending_pdf(document_id: str) -> None:
    """Drop sensitive PDF bytes and page text from session state."""
    st.session_state.pending_pdf_documents.pop(document_id, None)
    st.session_state.pending_pdf_payloads.pop(document_id, None)
    st.session_state.pdf_active_page.pop(document_id, None)
    st.session_state.pdf_editor_versions.pop(document_id, None)


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


def _clear_current_report() -> None:
    report = _active_report()
    if not report:
        return
    report_id = str(report["report_id"])
    st.session_state.loaded_reports.pop(report_id, None)
    st.session_state.extraction_results.pop(report_id, None)
    st.session_state.case_assignments.pop(report_id, None)
    st.session_state.queue_methods.pop(report_id, None)
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
        st.session_state.queue_methods[report_id] = result.method


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
        st.session_state.workspace_area = "Human Review"
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
    st.sidebar.title(":material/biotech: OncoExtract")
    st.sidebar.caption("Evidence-grounded pathology review")
    st.sidebar.caption("DAILY REVIEW")
    st.sidebar.radio(
        "Navigation",
        DAILY_NAVIGATION,
        key="daily_navigation",
        on_change=lambda: st.session_state.update(
            nav_page=st.session_state.daily_navigation
        ),
    )
    with st.sidebar.expander(
        "Research tools",
        icon=":material/science:",
        expanded=st.session_state.nav_page == "Evaluation",
    ):
        st.caption(
            "Synthetic evaluation and workflow metrics are separated from daily case review."
        )
        if st.button(
            "Evaluation",
            icon=":material/analytics:",
            width="stretch",
            type="primary" if st.session_state.nav_page == "Evaluation" else "secondary",
            key="open_research_evaluation",
        ):
            st.session_state.nav_page = "Evaluation"
            st.rerun()
        st.caption(
            "Visible in this local prototype. Restrict this section by authenticated "
            "role before broader deployment."
        )
    page = str(st.session_state.nav_page)
    reports = st.session_state.loaded_reports
    if reports:
        st.sidebar.subheader("Current case")
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
        st.sidebar.caption(f"{len(reports)} in session · {result_count}/3 methods run")
    else:
        st.sidebar.info("Start in Workspace to load a report.", icon=":material/info:")
    st.sidebar.warning("Research prototype · Not for clinical use")
    st.sidebar.caption(":material/computer: Local Streamlit session")
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


def _try_synthetic_example() -> None:
    """Load the first bundled scenario from the onboarding call to action."""

    samples = st.session_state.sample_reports
    if not samples:
        _set_flash("error", "No synthetic examples are available in this installation.")
        return
    selected = samples[0]
    try:
        prepared = prepare_report(
            str(selected.get("text", "")),
            source="synthetic",
            approved=True,
            report_id=str(selected.get("report_id")),
            description=str(selected.get("description", "Synthetic demonstration report")),
            cancer_type=selected.get("cancer_type"),
        )
    except ReportInputError as error:
        _set_flash("error", f"The synthetic example could not be loaded: {error}")
    else:
        _load_prepared_report(prepared)


def _render_pdf_text_review(document_id: str, cancer_type: str, approved: bool) -> None:
    """Render the gated page-by-page OCR and text acceptance stage."""

    document = ProcessedDocument.model_validate(
        st.session_state.pending_pdf_documents[document_id]
    )
    payload = st.session_state.pending_pdf_payloads[document_id]
    active_page = int(st.session_state.pdf_active_page.get(document_id, 1))
    active_page = min(max(active_page, 1), len(document.pages))
    page = document.pages[active_page - 1]

    st.markdown("#### OCR & text review")
    accepted_count = sum(item.accepted for item in document.pages)
    st.caption(
        f"{document.document_id} · {accepted_count}/{len(document.pages)} pages accepted · "
        f"{document.provenance.native_text_pages} native / {document.provenance.ocr_pages} OCR"
    )
    page_number = st.selectbox(
        "Page", list(range(1, len(document.pages) + 1)), index=active_page - 1,
        format_func=lambda number: f"Page {number}", key=f"pdf_page_{document_id}",
    )
    if page_number != active_page:
        st.session_state.pdf_active_page[document_id] = page_number
        st.rerun()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(render_pdf_page(payload, active_page), caption=f"Source page {active_page}", width="stretch")
    with right:
        quality_icon = {"Good": "✓", "Review recommended": "!", "Poor": "×"}[page.quality.label]
        st.markdown(f"**{quality_icon} {page.quality.label}** · {page.extraction_method.replace('_', ' ')}")
        st.caption(f"{page.character_count:,} characters · confidence score {page.quality.score:.0%}")
        for warning in page.warning_flags:
            st.warning(warning, icon=":material/warning:")
        version = int(st.session_state.pdf_editor_versions.get(document_id, 0))
        editor_key = f"pdf_text_{document_id}_{active_page}_{version}"
        if editor_key not in st.session_state:
            st.session_state[editor_key] = page.authoritative_text
        edited_text = st.text_area(
            "Authoritative page text", height=430, key=editor_key,
            help="Edit against the source image. This accepted text becomes the evidence source.",
        )
        if edited_text != page.authoritative_text:
            page.accepted = False

        with st.container(horizontal=True):
            if st.button("Accept page", type="primary", key=f"accept_pdf_{document_id}_{active_page}"):
                try:
                    page.corrected_text = sanitize_report_text(edited_text)
                except ReportInputError as error:
                    st.error(str(error))
                else:
                    page.accepted = True
                    st.session_state.pending_pdf_documents[document_id] = document.model_dump(mode="json")
                    st.rerun()
            if st.button("Reset text", key=f"reset_pdf_{document_id}_{active_page}"):
                page.corrected_text = None
                page.accepted = False
                st.session_state.pending_pdf_documents[document_id] = document.model_dump(mode="json")
                st.session_state.pdf_editor_versions[document_id] = version + 1
                st.rerun()
            if st.button("Rerun OCR", key=f"ocr_pdf_{document_id}_{active_page}"):
                with st.spinner(f"Running local OCR on page {active_page}…"):
                    refreshed = process_pdf("document.pdf", payload, force_ocr_pages=[active_page])
                replacement = refreshed.pages[active_page - 1]
                document.pages[active_page - 1] = replacement
                document.provenance.ocr_pages = sum(p.extraction_method == "ocr" for p in document.pages)
                document.provenance.native_text_pages = len(document.pages) - document.provenance.ocr_pages
                st.session_state.pending_pdf_documents[document_id] = document.model_dump(mode="json")
                st.session_state.pdf_editor_versions[document_id] = version + 1
                st.rerun()

    with st.container(horizontal=True):
        if st.button("Accept all non-empty pages", key=f"accept_all_{document_id}"):
            for item in document.pages:
                if item.authoritative_text.strip():
                    item.corrected_text = item.authoritative_text.strip()
                    item.accepted = True
            st.session_state.pending_pdf_documents[document_id] = document.model_dump(mode="json")
            st.rerun()
        if st.button("Force OCR on all pages", key=f"ocr_all_{document_id}"):
            with st.spinner("Running local OCR on all pages…"):
                refreshed = process_pdf("document.pdf", payload, force_ocr_all=True)
            st.session_state.pending_pdf_documents[document_id] = refreshed.model_dump(mode="json")
            st.session_state.pdf_editor_versions[document_id] = version + 1
            st.rerun()
        if st.button("Discard PDF", key=f"discard_pdf_{document_id}"):
            _clear_pending_pdf(document_id)
            _rotate_sensitive_loader_state()
            st.rerun()

    can_continue = all(item.accepted and item.authoritative_text.strip() for item in document.pages)
    if st.button(
        "Continue to abstraction", type="primary", width="stretch",
        disabled=not can_continue, key=f"continue_pdf_{document_id}",
    ):
        try:
            authoritative_text, page_ranges = assemble_accepted_text(document)
            prepared = prepare_report(
                authoritative_text, source="uploaded", approved=approved,
                report_id=document.document_id,
                description="Approved PDF pathology report",
                cancer_type=None if cancer_type == "Not specified" else cancer_type,
            )
        except (PDFProcessingError, ReportInputError) as error:
            st.error(str(error))
        else:
            document.provenance.reviewer_accepted_at = datetime.now(timezone.utc)
            prepared["source_report_digest"] = document.provenance.source_report_digest
            prepared["processed_document"] = document.model_dump(mode="json")
            prepared["page_ranges"] = page_ranges
            _clear_pending_pdf(document_id)
            _load_prepared_report(prepared)


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
                st.info(
                    "Synthetic examples are unavailable. Paste approved text or upload "
                    "an approved report using the adjacent tabs."
                )
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
            st.button(
                "Load pasted report", type="primary", width="stretch",
                key="load_pasted_report", on_click=_load_pasted_from_state,
                args=(loader_version,),
            )

        with upload_tab:
            upload_file_key = _loader_widget_key("uploaded_report_file", loader_version)
            upload_approval_key = _loader_widget_key("upload_approved", loader_version)
            uploaded = st.file_uploader(
                "Upload pathology report file",
                type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
                accept_multiple_files=False, key=upload_file_key,
                on_change=_revoke_upload_approval,
                args=(loader_version,),
                help="PDFs use native text extraction first and local OCR only when needed. "
                     "No document content is sent to an external service.",
            )
            if uploaded is not None:
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                if ext in ("jpg", "jpeg", "png"):
                    st.caption("📷 Image file detected — text will be extracted via OCR.")
                elif ext == "pdf":
                    st.caption("📄 PDF detected — each page must pass OCR & text review before abstraction.")
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
            if st.button("Process uploaded file", type="primary", width="stretch",
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
                        if not approval_matches_content:
                            raise ReportInputError(
                                "Confirm this exact file is de-identified or approved before processing."
                            )
                        if uploaded.name.lower().endswith(".pdf"):
                            with st.spinner("Extracting native text and selectively running local OCR…"):
                                document = process_pdf(uploaded.name, uploaded_bytes)
                            st.session_state.pending_pdf_documents[document.document_id] = (
                                document.model_dump(mode="json")
                            )
                            st.session_state.pending_pdf_payloads[document.document_id] = uploaded_bytes
                            st.session_state.pdf_active_page[document.document_id] = 1
                            st.session_state.pdf_editor_versions[document.document_id] = 0
                            prepared = None
                        else:
                            extracted_text = _extract_text_from_file(uploaded.name, uploaded_bytes)
                            if not extracted_text.strip():
                                raise ReportInputError("No text could be extracted from this file.")
                            prepared = prepare_report(
                                extracted_text, source="uploaded", approved=approval_matches_content,
                                description="Approved uploaded pathology report",
                                cancer_type=None if cancer_type == "Not specified" else cancer_type,
                            )
                    except (ReportInputError, PDFProcessingError) as error:
                        st.error(str(error))
                    else:
                        if prepared is not None:
                            _load_prepared_report(prepared)

            if uploaded is not None and uploaded.name.lower().endswith(".pdf"):
                pending_id = "TCGA-PDF-" + hashlib.sha256(uploaded.getvalue()).hexdigest()[:12].upper()
                if pending_id in st.session_state.pending_pdf_documents:
                    _render_pdf_text_review(pending_id, cancer_type, approved)


def _render_case_strip(report: dict[str, Any]) -> None:
    report_id = str(report["report_id"])
    source = str(report.get("source", "local")).replace("_", " ").title()
    cancer_type = report.get("cancer_type") or "Cancer type not specified"
    description = report.get("description") or "Local report"
    draft_indicator = (
        '<span class="status-pill tone-warning" role="status" '
        'aria-label="Unsaved changes"><span aria-hidden="true">✎</span> '
        'Autosaved draft</span>'
        if _report_has_dirty_review(report_id) else ""
    )
    st.markdown(
        f'<div class="case-strip" aria-label="Current report">'
        f'<span class="case-id">{html.escape(report_id)}</span>'
        '<span class="status-pill tone-positive">Approved for local use</span>'
        f'{draft_indicator}'
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
    """Focus one field in the persistent evidence pane."""
    st.session_state.workspace_area = "Human Review"
    st.session_state.focused_variable = variable_name
    st.session_state[f"evidence_focus_{report_id}_{method}"] = variable_name


def _review_shortcut_state(key: str) -> Any:
    """Return the most recent CCv2 shortcut payload, if present."""

    state = st.session_state.get(key)
    if state is None:
        return None
    return getattr(state, "shortcut", None) or (
        state.get("shortcut") if isinstance(state, dict) else None
    )


def _apply_review_shortcut(
    *, component_key: str, draft_key: str, scope: str,
    variable_names: list[str], report_id: str, method: str,
) -> None:
    """Apply one keyboard event before Streamlit redraws the review widgets."""

    payload = _review_shortcut_state(component_key)
    action = payload.get("action") if isinstance(payload, dict) else None
    if not variable_names or action is None:
        return
    active_fields = st.session_state.keyboard_active_fields
    active_name = active_fields.get(draft_key)
    index = variable_names.index(active_name) if active_name in variable_names else 0
    if action == "next":
        index = min(index + 1, len(variable_names) - 1)
    elif action == "previous":
        index = max(index - 1, 0)
    elif action == "evidence":
        _focus_evidence(report_id, method, variable_names[index])
    elif action == "complete":
        st.session_state.keyboard_save_requests[draft_key] = True
    elif action in {"accept", "correct", "flag"}:
        mapped = {
            "accept": ReviewAction.ACCEPTED.value,
            "correct": ReviewAction.CORRECTED.value,
            "flag": ReviewAction.FLAGGED.value,
        }[action]
        variable_name = variable_names[index]
        decisions = st.session_state.review_decisions.get(draft_key, {})
        if variable_name in decisions:
            decisions[variable_name]["action"] = mapped
            st.session_state[f"review_action_{scope}_{variable_name}"] = mapped
    active_fields[draft_key] = variable_names[index]


def _sync_case_assignment(report_id: str, widget_key: str) -> None:
    value = str(st.session_state.get(widget_key, "")).strip()
    st.session_state.case_assignments[report_id] = value or "Unassigned"


def _sync_queue_method(report_id: str) -> None:
    method = str(st.session_state.get("workspace_method", "evidence_first"))
    if method in METHOD_LABELS:
        st.session_state.queue_methods[report_id] = method


def _field_triage_reasons(
    result: ExtractionResult,
    variable: VariableExtraction,
    decision: dict[str, Any],
) -> list[str]:
    """Return transparent reasons that move a field into exception review."""
    reasons: list[str] = []
    status = variable.documentation_status
    if status == DocumentationStatus.CONFLICTING:
        reasons.append("Conflicting evidence")
    if status in {
        DocumentationStatus.NOT_DOCUMENTED,
        DocumentationStatus.CANNOT_BE_ASSIGNED,
    }:
        reasons.append("Missing or unassignable documentation")
    if status in {
        DocumentationStatus.UNCERTAIN,
        DocumentationStatus.UNSUPPORTED,
        DocumentationStatus.MANUAL_REVIEW_REQUIRED,
    }:
        reasons.append("Uncertain, low-confidence, or unsupported output")
    if status == DocumentationStatus.SUPERSEDED:
        reasons.append("Corrected or superseded statement")
    if status == DocumentationStatus.NEGATED:
        reasons.append("Explicitly negated finding")
    notes = (variable.notes or "").casefold()
    if any(marker in notes for marker in ("low-confidence", "not decisive", "withheld")):
        if "Uncertain, low-confidence, or unsupported output" not in reasons:
            reasons.append("Uncertain, low-confidence, or unsupported output")

    other_results = st.session_state.extraction_results.get(result.report_id, {})
    signatures = set()
    for method_result in other_results.values():
        matching = next(
            (
                item for item in method_result.variables
                if item.variable_name == variable.variable_name
            ),
            None,
        )
        if matching is not None:
            signatures.add((matching.extracted_value, matching.documentation_status.value))
    if len(signatures) > 1:
        reasons.append("Extraction methods disagree")

    if decision.get("action") == ReviewAction.CORRECTED.value:
        reasons.append("Reviewer correction in progress")
    return reasons


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
        'available. Select the method above, then choose “Run extraction” to populate '
        'this area.</div>',
        unsafe_allow_html=True,
    )


def _render_evidence_excerpt(
    text: str, start: int, end: int, page_number: int | None = None
) -> None:
    page_detail = f" · Source page: {page_number}" if page_number else ""
    st.markdown(
        f'<div class="evidence-excerpt">{html.escape(text)}</div>'
        f'<details class="technical-details"><summary>Technical details</summary>'
        f'<div>Character offsets: {start}–{end} (zero-based, end exclusive)'
        f'{page_detail}</div></details>',
        unsafe_allow_html=True,
    )


def _render_ai_abstraction(result: ExtractionResult | None, method: str) -> None:
    st.markdown('<div class="section-heading">Extraction results</div>', unsafe_allow_html=True)
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
    requested_focus = st.query_params.get("focus")
    current_focus = (
        str(requested_focus)
        if requested_focus in focus_options
        else st.session_state.focused_variable
    )
    if current_focus not in focus_options:
        current_focus = "all"
    focus_widget_key = (
        f"evidence_focus_{report['report_id']}_{result.method if result else 'none'}"
    )
    if requested_focus in focus_options:
        st.session_state[focus_widget_key] = current_focus
    focused = st.selectbox(
        "Highlight field", focus_options, index=focus_options.index(current_focus),
        format_func=lambda item: "All evidence" if item == "all" else variable_label(item),
        key=focus_widget_key,
    )
    st.session_state.focused_variable = focused
    evidence_variables = [v for v in variables if v.evidence]
    if evidence_variables:
        links = "".join(
            f'<a href="?focus={v.variable_name}#{evidence_anchor(v.variable_name)}">'
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
                    evidence.text, evidence.start_offset, evidence.end_offset,
                    evidence.page_number,
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
        st.info("No issues match these filters. Clear a filter to inspect all recorded issues.")
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
                    evidence.text, evidence.start_offset, evidence.end_offset,
                    evidence.page_number,
                )


def _render_review_field(
    variable: VariableExtraction,
    decision: dict[str, Any],
    scope: str,
    report_id: str,
    method: str,
    triage_reasons: list[str],
    keyboard_active: bool = False,
) -> None:
    action = str(decision.get("action", ReviewAction.PENDING.value))
    if action not in ACTION_OPTIONS:
        action = ReviewAction.PENDING.value
    with st.container(border=True):
        if keyboard_active:
            st.badge("Keyboard focus", icon=":material/keyboard:", color="blue")
        if triage_reasons:
            st.markdown(
                '<div class="triage-label triage-label--exception" role="status">'
                '<span aria-hidden="true">⚠</span> Review exception first</div>'
                f'<div class="triage-reasons">{html.escape(" · ".join(triage_reasons))}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="triage-label triage-label--routine" role="status">'
                '<span aria-hidden="true">✓</span> Lower priority · Fully supported; '
                'final verification is still required</div>',
                unsafe_allow_html=True,
            )
        field_label = html.escape(variable_label(variable.variable_name))
        if variable.evidence:
            st.markdown(
                f'<h3><a class="review-field-link" '
                f'href="?focus={variable.variable_name}#{evidence_anchor(variable.variable_name)}" '
                f'title="Focus and jump to exact evidence for {field_label}">'
                f'{field_label} <span aria-hidden="true">↗</span></a></h3>',
                unsafe_allow_html=True,
            )
        else:
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
            st.markdown(
                f'<a class="anchor-link" '
                f'href="?focus={variable.variable_name}#{evidence_anchor(variable.variable_name)}">'
                f'Jump to exact evidence'
                f'</a>',
                unsafe_allow_html=True,
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
    reviewer_identity = st.session_state.case_assignments.get(
        result.report_id, "Unassigned"
    ).strip() or "Unassigned"
    with st.expander("Extraction provenance", icon=":material/fingerprint:"):
        attributes = [
            "Extraction method", "Model / rules version", "Source report digest",
            "Extraction timestamp", "Reviewer identity", "Identity assurance",
        ]
        values = [
            _method_label(result.method), result.model_version,
            result.source_report_digest or "Unavailable for legacy result",
            result.timestamp.isoformat() if result.timestamp else "Unavailable",
            reviewer_identity, "Session-entered · unverified (authentication disabled)",
        ]
        if result.document_provenance:
            provenance = result.document_provenance
            attributes.extend(["PDF processor", "PDF pages", "Page extraction methods"])
            values.extend([
                provenance.processor_version,
                str(provenance.page_count),
                f"{provenance.native_text_pages} native text · {provenance.ocr_pages} local OCR",
            ])
        st.table(
            {
                "Attribute": attributes,
                "Recorded value": values,
            }
        )
    progress_placeholder = st.empty()
    progress_caption = st.empty()
    triage = {
        variable.variable_name: _field_triage_reasons(
            result, variable, decisions[variable.variable_name]
        )
        for variable in result.variables
    }
    exception_variables = [
        variable for variable in result.variables if triage[variable.variable_name]
    ]
    routine_variables = [
        variable for variable in result.variables if not triage[variable.variable_name]
    ]
    ordered_variables = exception_variables + routine_variables
    ordered_names = [variable.variable_name for variable in ordered_variables]
    shortcut_component_key = f"review_shortcuts_{scope}"
    active_name = st.session_state.keyboard_active_fields.get(key)
    active_index = ordered_names.index(active_name) if active_name in ordered_names else 0
    if ordered_names:
        st.session_state.keyboard_active_fields[key] = ordered_names[active_index]
    with st.expander(
        "Keyboard reviewing",
        icon=":material/keyboard:",
    ):
        st.caption(
            "Shortcuts work while this review view is open. Letter shortcuts are disabled "
            "while typing; Cmd/Ctrl + Enter remains available to complete the review."
        )
        try:
            _REVIEW_SHORTCUTS(
                key=shortcut_component_key,
                data={"active_field": ordered_names[active_index] if ordered_names else None},
                on_shortcut_change=lambda: _apply_review_shortcut(
                    component_key=shortcut_component_key,
                    draft_key=key,
                    scope=scope,
                    variable_names=ordered_names,
                    report_id=result.report_id,
                    method=result.method,
                ),
            )
        except TypeError:
            st.caption("A accept · C correct · F flag · J/K field · E evidence · Cmd/Ctrl+Enter complete")
        if ordered_names:
            st.caption(
                f"Active field: {variable_label(ordered_names[active_index])} · "
                "Use J/K to move field focus."
            )
    if exception_variables:
        st.markdown(
            f"#### :material/priority_high: Review exceptions first "
            f"({len(exception_variables)})"
        )
        st.caption(
            "Exceptions include documentation problems, withheld output, method "
            "disagreement, and corrected or superseded statements."
        )
    for index, variable in enumerate(ordered_variables):
        if index == len(exception_variables) and routine_variables:
            st.markdown(
                f"#### :material/check_circle: Fully supported fields "
                f"({len(routine_variables)})"
            )
            st.caption(
                "These fields are lower priority, but each still requires a final reviewer decision."
            )
        _render_review_field(
            variable, decisions[variable.variable_name], scope,
            result.report_id, result.method, triage[variable.variable_name],
            keyboard_active=index == active_index,
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
    returned_for_clarification = st.checkbox(
        "Return this report for clarification",
        value=bool(st.session_state.review_returned_for_clarification.get(key, False)),
        help=("Use when the report cannot be completed without additional or corrected "
              "source documentation. Explain the request in the overall review note."),
        key=f"return_for_clarification_{scope}",
    )
    st.session_state.review_returned_for_clarification[key] = returned_for_clarification
    dirty = _review_is_dirty(key)
    if dirty:
        st.session_state.review_started_at.setdefault(
            key, datetime.now(timezone.utc).isoformat()
        )
        st.session_state.review_draft_saved_at[key] = datetime.now(timezone.utc).isoformat()
    saved_at = st.session_state.review_draft_saved_at.get(key)
    st.caption(
        f":material/cloud_done: Draft autosaved in this session"
        + (f" · {saved_at}" if saved_at else "")
    )
    complete_clicked = st.button(
        "Complete review", type="primary", width="stretch",
        icon=":material/task_alt:",
        disabled=not dirty, key=f"save_review_{scope}",
    )
    keyboard_complete = bool(
        st.session_state.keyboard_save_requests.pop(key, False)
    )
    if complete_clicked or keyboard_complete:
        errors = validate_review_decisions(result, decisions, str(report["text"]))
        if reviewer_identity == "Unassigned":
            errors.append(
                "Assigned reviewer: enter a reviewer identity before completing the review."
            )
        if returned_for_clarification and not overall_note.strip():
            errors.append(
                "Overall review note: explain what clarification is required."
            )
        if errors:
            st.error("Complete the review before saving:")
            for error in errors:
                st.write(f"• {error}")
        else:
            try:
                completed_at = datetime.now(timezone.utc)
                started_at_text = st.session_state.review_started_at.get(key)
                try:
                    started_at = datetime.fromisoformat(started_at_text)
                except (TypeError, ValueError):
                    started_at = completed_at
                    started_at_text = started_at.isoformat()
                duration_seconds = max(
                    0.0, (completed_at - started_at).total_seconds()
                )
                snapshot, records = create_review_snapshot(
                    result, copy_decisions(decisions), report_text=str(report["text"]),
                    overall_note=overall_note,
                    timestamp=completed_at.isoformat(),
                    review_started_at=started_at_text,
                    review_duration_seconds=duration_seconds,
                    returned_for_clarification=returned_for_clarification,
                    reviewer_identity=reviewer_identity,
                    reviewer_identity_verified=False,
                )
            except ValueError as error:
                st.error(str(error))
            else:
                st.session_state.review_snapshots.append(snapshot)
                st.session_state.audit_records.extend(records)
                st.session_state.review_baselines[key] = copy_decisions(decisions)
                st.session_state.review_note_baselines[key] = overall_note
                st.session_state.review_draft_saved_at.pop(key, None)
                st.session_state.review_started_at.pop(key, None)
                st.session_state.review_returned_for_clarification.pop(key, None)
                _set_flash("success", f"Completed review {snapshot.review_id} with "
                           f"{len(records)} audit records.")
                st.rerun()


def _render_split_review(
    report: dict[str, Any], result: ExtractionResult | None, method: str
) -> None:
    """Keep source evidence and reviewer decisions visible together."""
    reader_mode = st.toggle(
        "Distraction-free report reading",
        key="reader_mode",
        help="Temporarily hide review controls and give the report the full workspace width.",
    )
    if reader_mode:
        _render_evidence_viewer(report, result)
        return
    report_pane, review_pane = st.columns([1.08, .92], gap="large")
    with report_pane.container(height=760):
        _render_evidence_viewer(report, result)
    with review_pane.container(height=760):
        _render_human_review(report, result, method)


def page_workspace() -> None:
    _page_heading(
        "Review workspace",
        "Load a report, extract four pathology fields, verify their evidence, and complete the review.",
    )
    _research_banner()

    reports_loaded = len(st.session_state.get("loaded_reports", {}))
    extractions_run = sum(
        len(results) for results in st.session_state.get("extraction_results", {}).values()
    )
    reviews_saved = len(st.session_state.get("review_snapshots", []))
    audit_records = len(st.session_state.get("audit_records", []))

    report = _active_report()
    active_results = (
        st.session_state.extraction_results.get(str(report["report_id"]), {})
        if report is not None else {}
    )
    _workflow_steps(
        has_report=report is not None,
        has_result=bool(active_results),
        has_review=bool(reviews_saved),
        has_pending_pdf=bool(st.session_state.pending_pdf_documents),
    )

    if report is None:
        with st.container(border=True):
            st.subheader(":material/rocket_launch: Start with a safe demonstration")
            st.write(
                "Try the complete review workflow with a bundled synthetic report; "
                "no clinical data or setup is required."
            )
            st.button(
                "Try a synthetic example",
                type="primary",
                icon=":material/play_arrow:",
                width="stretch",
                on_click=_try_synthetic_example,
                disabled=not bool(st.session_state.sample_reports),
                key="try_synthetic_example",
            )

    if reports_loaded:
        st.markdown(
            f'<div class="stats-grid">'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-blue">01</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Reports loaded</p>'
        f'<p class="stat-value">{reports_loaded}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-green">02</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Extractions run</p>'
        f'<p class="stat-value">{extractions_run}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-amber">03</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Reviews saved</p>'
        f'<p class="stat-value">{reviews_saved}</p>'
        f'</div></div>'
        f'<div class="stat-card">'
        f'<div class="stat-icon stat-icon-red">04</div>'
        f'<div class="stat-content">'
        f'<p class="stat-label">Audit records</p>'
        f'<p class="stat-value">{audit_records}</p>'
        f'</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander(
        "Understanding pN0, pNX, and not documented",
        icon=":material/help:",
    ):
        st.markdown(
            "- **pN0:** The report explicitly documents pathological node category N0.\n"
            "- **pNX:** The report explicitly states that pathological nodal status cannot "
            "be assessed or assigned.\n"
            "- **Not documented:** No explicit pathological N category appears in the report.\n\n"
            "These states are not interchangeable. The extractor copies explicit staging "
            "language and does not infer pN from node counts."
        )
    
    _render_case_queue()
    _render_source_loader()
    report = _active_report()
    if report is None:
        st.markdown(
            '<div class="empty-panel"><strong>Choose how to begin.</strong><br>'
            'Use “Try a synthetic example” for a guided demonstration, or open the '
            'report loader to paste or upload approved text.</div>', unsafe_allow_html=True,
        )
        return
    _render_case_strip(report)
    st.subheader(":material/fact_check: Extract and verify")
    method_col, reviewer_col, run_col, clear_col = st.columns(
        [1.45, 1.05, .75, .65], vertical_alignment="bottom"
    )
    with method_col:
        method = st.selectbox(
            "Extraction method", ["evidence_first", "baseline", "ml"],
            format_func=_method_label, key="workspace_method",
            help=("Evidence-first validates exact support and may abstain. "
                  "Baseline is rule-based comparison. "
                  "ML uses a local TF-IDF + logistic regression model with evidence anchoring."),
            on_change=_sync_queue_method,
            args=(str(report["report_id"]),),
        )
        st.session_state.queue_methods[str(report["report_id"])] = method
    with reviewer_col:
        assignment_key = f"case_assignment_{report['report_id']}"
        if assignment_key not in st.session_state:
            st.session_state[assignment_key] = st.session_state.case_assignments.get(
                str(report["report_id"]), "Unassigned"
            )
        assigned_reviewer = st.text_input(
            "Assigned reviewer", key=assignment_key,
            help="Session-only assignment; authentication is currently disabled.",
            on_change=_sync_case_assignment,
            args=(str(report["report_id"]), assignment_key),
        ).strip() or "Unassigned"
        st.session_state.case_assignments[str(report["report_id"])] = assigned_reviewer
    with run_col:
        if st.button("Run extraction", type="primary", width="stretch",
                     icon=":material/play_arrow:",
                     key="run_workspace_extraction"):
            _request_extraction(method)
    with clear_col:
        if st.button("Clear report", width="stretch", key="request_clear_report",
                     icon=":material/close:"):
            st.session_state.show_clear_dialog = True
            st.rerun()
    result = _result_for(str(report["report_id"]), method)
    if result:
        _render_priority(result)
        st.caption("The result passed the structured pipeline contract. Reviewer verification is still required.")
    workspace_areas = (
        "Extraction results", "Evidence Viewer", "Documentation Issues", "Human Review",
    )
    if st.session_state.workspace_area not in workspace_areas:
        st.session_state.workspace_area = workspace_areas[0]
    workspace_area = st.radio(
        "Workspace area", workspace_areas, horizontal=True, key="workspace_area",
        help="Choose one report task area. Evidence buttons open the viewer with the field focused.",
    )
    if workspace_area == "Extraction results":
        _render_ai_abstraction(result, method)
    elif workspace_area == "Evidence Viewer":
        _render_evidence_viewer(report, result)
    elif workspace_area == "Documentation Issues":
        _render_documentation_issues(result, method)
    else:
        _render_split_review(report, result, method)


def _render_comparison_panel(method: str, variable: VariableExtraction) -> None:
    st.markdown(f"**{_method_label(method)}**")
    st.write(variable.extracted_value or "No value returned")
    st.markdown(_status_badge(variable.documentation_status), unsafe_allow_html=True)
    spans = variable.evidence or []
    if spans:
        st.caption(f"{len(spans)} exact evidence span(s)")
        for evidence in spans:
            _render_evidence_excerpt(
                evidence.text, evidence.start_offset, evidence.end_offset,
                evidence.page_number,
            )
    else:
        st.caption("No evidence span returned.")


def _comparison_cell(variable: VariableExtraction) -> str:
    value = variable.extracted_value or "No value returned"
    status = status_presentation(variable.documentation_status).label
    return f"{value} · {status}"


def page_comparison() -> None:
    _page_heading(
        "Method Comparison",
        "Inspect each value, QA status, and exact evidence side by side for the current report.",
    )
    report = _active_report()
    if report is None:
        st.info(
            "Open Workspace and load a synthetic or approved report, then return here "
            "to compare extraction methods."
        )
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
    matrix_rows: list[dict[str, str]] = []
    disagreements: list[str] = []
    for name in CORE_VARIABLES:
        variables = {
            method: next(
                variable for variable in results[method].variables
                if variable.variable_name == name
            )
            for method in methods
        }
        signatures = {
            (variable.extracted_value, variable.documentation_status.value)
            for variable in variables.values()
        }
        agrees = len(signatures) == 1
        if not agrees:
            disagreements.append(name)
        matrix_rows.append({
            "Field": variable_label(name),
            "Baseline": _comparison_cell(variables["baseline"]),
            "Evidence-first": _comparison_cell(variables["evidence_first"]),
            "ML": _comparison_cell(variables["ml"]),
            "Agreement": "Same output" if agrees else "Different output",
        })
    st.caption(
        f"{len(disagreements)} of {len(CORE_VARIABLES)} fields differ in value or QA status. "
        "Agreement describes consistency between methods, not correctness."
    )
    st.dataframe(
        pd.DataFrame(matrix_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Field": st.column_config.TextColumn("Field", pinned=True),
            "Baseline": st.column_config.TextColumn("Baseline", width="medium"),
            "Evidence-first": st.column_config.TextColumn(
                "Evidence-first", width="medium"
            ),
            "ML": st.column_config.TextColumn("ML", width="medium"),
            "Agreement": st.column_config.TextColumn("Agreement", width="small"),
        },
    )
    st.caption(
        "Method cells use plain text. Color appears only in documentation-status badges "
        "inside the evidence details below."
    )

    if not disagreements:
        st.markdown(
            "**All methods produced the same output.** Reviewer verification is still "
            "required; agreement is not evidence of correctness."
        )
        return

    st.subheader(":material/difference: Disagreement evidence")
    for name in disagreements:
        with st.expander(
            f"{variable_label(name)} · Different output",
            icon=":material/unfold_more:",
        ):
            columns = st.columns(3)
            for column, method in zip(columns, methods):
                variable = next(
                    item for item in results[method].variables
                    if item.variable_name == name
                )
                with column:
                    _render_comparison_panel(method, variable)


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


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    minutes, remaining = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}m {remaining:02d}s"
    return f"{remaining}s"


def _render_reviewer_efficiency_metrics() -> None:
    snapshots: list[ReviewSnapshot] = st.session_state.review_snapshots
    st.subheader(":material/speed: Reviewer efficiency")
    st.caption(
        "Session workflow measures from completed reviews. These describe reviewer "
        "activity, not model accuracy or clinical performance."
    )
    if not snapshots:
        st.info(
            "Complete at least one review to populate reviewer-efficiency metrics."
        )
        return

    fields = [field for snapshot in snapshots for field in snapshot.fields]
    accepted_count = sum(
        field.review_action == ReviewAction.ACCEPTED for field in fields
    )
    acceptance_rate = accepted_count / len(fields) if fields else None

    durations_by_report: dict[str, float] = {}
    for snapshot in snapshots:
        if snapshot.review_duration_seconds is not None:
            durations_by_report[snapshot.report_id] = (
                durations_by_report.get(snapshot.report_id, 0.0)
                + snapshot.review_duration_seconds
            )
    median_duration = (
        float(median(durations_by_report.values()))
        if durations_by_report else None
    )
    clarification_reports = {
        snapshot.report_id for snapshot in snapshots
        if snapshot.returned_for_clarification
    }
    report_ids = {snapshot.report_id for snapshot in snapshots}
    edit_free = sum(
        all(field.review_action == ReviewAction.ACCEPTED for field in snapshot.fields)
        and not snapshot.returned_for_clarification
        for snapshot in snapshots
    )
    edit_free_rate = edit_free / len(snapshots) if snapshots else None

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Median review time / report",
        _format_duration(median_duration),
        border=True,
    )
    metric_columns[1].metric(
        "Field acceptance rate",
        "N/A" if acceptance_rate is None else f"{acceptance_rate:.1%}",
        f"{accepted_count}/{len(fields)} fields",
        border=True,
    )
    metric_columns[2].metric(
        "Returned for clarification",
        str(len(clarification_reports)),
        f"of {len(report_ids)} reports",
        border=True,
    )
    metric_columns[3].metric(
        "Completed without editing",
        "N/A" if edit_free_rate is None else f"{edit_free_rate:.1%}",
        f"{edit_free}/{len(snapshots)} reviews",
        border=True,
    )

    correction_rows = []
    for variable_name in CORE_VARIABLES:
        variable_fields = [
            field for field in fields if field.variable_name == variable_name
        ]
        correction_count = sum(
            field.review_action == ReviewAction.CORRECTED
            for field in variable_fields
        )
        correction_rows.append({
            "Field": variable_label(variable_name),
            "Corrections": correction_count,
            "Reviewed": len(variable_fields),
            "Correction rate": (
                correction_count / len(variable_fields) if variable_fields else None
            ),
        })
    detail_col, disagreement_col = st.columns(2)
    with detail_col:
        st.markdown("#### Corrections by field")
        st.dataframe(
            pd.DataFrame(correction_rows), hide_index=True, width="stretch",
            column_config={
                "Correction rate": st.column_config.NumberColumn(format="percent")
            },
        )
    with disagreement_col:
        st.markdown("#### Most common disagreement types")
        disagreement_labels = {
            ReviewAction.CORRECTED: "Corrected value or status",
            ReviewAction.REJECTED: "Rejected extraction",
            ReviewAction.FLAGGED: "Flagged for further review",
        }
        disagreement_counts = Counter(
            disagreement_labels.get(
                field.review_action, _action_label(field.review_action.value)
            )
            for field in fields
            if field.review_action != ReviewAction.ACCEPTED
        )
        if disagreement_counts:
            disagreement_rows = pd.DataFrame([
                {"Type": label, "Count": count}
                for label, count in disagreement_counts.most_common()
            ])
            st.dataframe(disagreement_rows, hide_index=True, width="stretch")
        else:
            st.caption("No reviewer disagreements have been recorded.")
    st.caption(
        "Timing starts with the first changed reviewer decision and ends when the "
        "review is completed. Multiple completed methods for one report contribute "
        "cumulatively to that report's time."
    )


def page_evaluation() -> None:
    _page_heading(
        "Evaluation and reviewer efficiency",
        "Compare illustrative synthetic model results with session-based reviewer workflow measures.",
    )
    _research_banner()
    st.subheader(":material/analytics: Illustrative synthetic model results")
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
            '<div class="empty-panel"><strong>Start a synthetic evaluation.</strong><br>'
            'Choose “Run synthetic evaluation” above to calculate research metrics from '
            'the bundled gold annotations.</div>',
            unsafe_allow_html=True,
        )
        _render_reviewer_efficiency_metrics()
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
    _render_reviewer_efficiency_metrics()


def _audit_rows(records: list[AuditRecord]) -> pd.DataFrame:
    return pd.DataFrame([
        {"Timestamp": record.timestamp, "Report": record.report_id,
         "Method": _method_label(record.method),
         "Reviewer": record.reviewer_identity,
         "Field": variable_label(record.variable_name),
         "Action": _action_label(record.reviewer_action.value),
         "Original": record.original_output.extracted_value or "—",
         "Result": record.resulting_value or "—",
         "Status": status_presentation(record.resulting_status).label}
        for record in records
    ])


def _render_audit_detail(record: AuditRecord) -> None:
    title = (f"Provenance · {record.timestamp} · {record.report_id} · "
             f"{variable_label(record.variable_name)} · "
             f"{_action_label(record.reviewer_action.value)}")
    with st.expander(title, icon=":material/fingerprint:"):
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
        with st.container():
            st.table(
                {
                    "Attribute": [
                        "Extraction method", "Model / rules version",
                        "Source report digest", "Extraction timestamp",
                        "Reviewer identity", "Identity assurance", "Review timestamp",
                        "Original value", "Corrected value", "Resulting value",
                        "Review reason",
                    ],
                    "Recorded value": [
                        _method_label(record.method), record.model_version,
                        record.source_report_digest,
                        record.extraction_timestamp or "Unavailable",
                        record.reviewer_identity,
                        ("Verified" if record.reviewer_identity_verified else
                         "Session-entered · unverified"),
                        record.timestamp,
                        record.original_output.extracted_value or "No value returned",
                        record.corrected_value or "Not corrected",
                        record.resulting_value or "No reviewed value",
                        record.review_reason,
                    ],
                }
            )
            evidence = record.resulting_evidence or record.original_output.evidence or []
            if evidence:
                st.caption("Exact evidence offsets")
                st.table(
                    {
                        "Evidence": [item.text for item in evidence],
                        "Start": [item.start_offset for item in evidence],
                        "End": [item.end_offset for item in evidence],
                    }
                )
            else:
                st.caption("Exact evidence offsets: no evidence retained for this result.")
            st.caption(f"Audit ID {record.audit_id} · Review ID {record.review_id}")
        if record.resulting_evidence:
            st.markdown("**Resulting exact evidence**")
            for evidence in record.resulting_evidence:
                _render_evidence_excerpt(
                    evidence.text, evidence.start_offset, evidence.end_offset,
                    evidence.page_number,
                )
        if record.original_output.evidence:
            st.markdown("**Original exact evidence**")
            for evidence in record.original_output.evidence:
                _render_evidence_excerpt(
                    evidence.text, evidence.start_offset, evidence.end_offset,
                    evidence.page_number,
                )


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
            st.info(
                "Complete a case review in Workspace to create its field-level audit trail."
            )
        else:
            audit_search = st.text_input(
                "Search audit history",
                placeholder="Search case, field, value, action, status, or reason",
                icon=":material/search:",
                key="audit_search",
            ).strip().casefold()
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
                        and (not action_filter or record.reviewer_action.value in action_filter)
                        and (
                            not audit_search
                            or audit_search in " ".join([
                                record.report_id,
                                record.method,
                                record.model_version,
                                record.source_report_digest,
                                record.reviewer_identity,
                                variable_label(record.variable_name),
                                record.reviewer_action.value,
                                record.original_output.extracted_value or "",
                                record.resulting_value or "",
                                record.resulting_status.value,
                                record.review_reason,
                            ]).casefold()
                        )]
            filtered.sort(key=lambda item: item.timestamp, reverse=True)
            st.caption(f"Showing {len(filtered)} of {len(records)} audit record(s).")
            if filtered:
                st.dataframe(_audit_rows(filtered), hide_index=True, width="stretch")
                st.markdown("#### Record details")
                for record in filtered:
                    _render_audit_detail(record)
            else:
                st.info("No records match these filters. Clear one or more filters to broaden the history.")
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

            st.markdown("#### Export one reviewed case")
            reviewed_report_ids = sorted({snapshot.report_id for snapshot in snapshots})
            selected_report_id = st.selectbox(
                "Reviewed case", reviewed_report_ids, key="single_reviewed_case_export"
            )
            selected_snapshots = [
                snapshot for snapshot in snapshots
                if snapshot.report_id == selected_report_id
            ]
            selected_review_ids = {snapshot.review_id for snapshot in selected_snapshots}
            selected_records = [
                record for record in records if record.review_id in selected_review_ids
            ]
            single_json_col, single_csv_col, single_audit_col = st.columns(3)
            safe_report_id = _safe_filename(selected_report_id)
            with single_json_col:
                st.download_button(
                    "Case JSON",
                    data=reviewed_results_json(selected_snapshots, selected_records),
                    file_name=f"{safe_report_id}-reviewed.json",
                    mime="application/json", width="stretch",
                    key="download_single_reviewed_json",
                )
            with single_csv_col:
                st.download_button(
                    "Case CSV", data=reviewed_results_csv(selected_snapshots),
                    file_name=f"{safe_report_id}-reviewed.csv",
                    mime="text/csv", width="stretch",
                    key="download_single_reviewed_csv",
                )
            with single_audit_col:
                st.download_button(
                    "Case audit CSV", data=audit_history_csv(selected_records),
                    file_name=f"{safe_report_id}-audit.csv",
                    mime="text/csv", width="stretch",
                    key="download_single_audit_csv",
                )
        else:
            st.info("Complete a review in Workspace, then return here to export it.")
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


def main() -> None:
    _initialize_state()
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
