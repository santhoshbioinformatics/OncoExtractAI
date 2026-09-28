from __future__ import annotations

import pymupdf as fitz
from streamlit.testing.v1 import AppTest

from src.pdf_processor import process_pdf


def _three_page_pdf() -> bytes:
    document = fitz.open()
    for marker in ("ONE", "TWO", "THREE"):
        page = document.new_page()
        text = (
            f"PAGE {marker} FINAL DIAGNOSIS. Invasive adenocarcinoma. "
            "Tumor size: 2.1 cm. Pathologic stage pT1c pN0. " * 3
        )
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def _review_harness() -> AppTest:
    payload = _three_page_pdf()
    document = process_pdf("synthetic.pdf", payload)
    document_id = document.document_id
    app = AppTest.from_string(
        """
import streamlit as st
from app import _initialize_state, _render_pdf_text_review
_initialize_state()
document_id = st.session_state.test_document_id
if document_id in st.session_state.pending_pdf_documents:
    _render_pdf_text_review(document_id, "LUAD", True)
""",
        default_timeout=20,
    )
    app.session_state["test_document_id"] = document_id
    app.session_state["pending_pdf_documents"] = {
        document_id: document.model_dump(mode="json")
    }
    app.session_state["pending_pdf_payloads"] = {document_id: payload}
    app.session_state["pdf_active_page"] = {document_id: 1}
    app.session_state["pdf_editor_versions"] = {document_id: 0}
    app.session_state["pdf_review_errors"] = {}
    app.session_state["pdf_render_cache"] = {}
    app.session_state["pdf_ocr_settings"] = {}
    app.session_state["pdf_large_preview"] = {}
    return app.run()


def test_review_advances_and_final_accept_continues_to_abstraction() -> None:
    app = _review_harness()
    document_id = app.session_state["test_document_id"]
    assert not app.exception
    assert app.button(key=f"accept_pdf_{document_id}_1").label == "Accept page and continue"

    app.button(key=f"accept_pdf_{document_id}_1").click().run()
    assert app.selectbox(key=f"pdf_page_{document_id}").value == 2
    app.button(key=f"accept_pdf_{document_id}_2").click().run()
    assert app.selectbox(key=f"pdf_page_{document_id}").value == 3
    assert app.button(key=f"accept_pdf_{document_id}_3").label == "Accept final page and continue"

    app.button(key=f"accept_pdf_{document_id}_3").click().run()
    assert not app.exception
    assert document_id not in app.session_state["pending_pdf_documents"]
    assert app.session_state["active_report_id"] == document_id


def test_blank_page_action_advances_and_excludes_page() -> None:
    app = _review_harness()
    document_id = app.session_state["test_document_id"]
    app.button(key=f"blank_pdf_{document_id}_1").click().run()
    assert not app.exception
    stored = app.session_state["pending_pdf_documents"][document_id]
    assert stored["pages"][0]["accepted"] is True
    assert stored["pages"][0]["excluded_as_blank"] is True
    assert app.selectbox(key=f"pdf_page_{document_id}").value == 2
    accept_buttons = [button for button in app.button if button.label.startswith("Accept")]
    assert len(accept_buttons) == 1
