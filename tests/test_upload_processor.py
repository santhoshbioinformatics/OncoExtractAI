from __future__ import annotations

from io import BytesIO

import pymupdf as fitz
import pytest
from docx import Document
from PIL import Image

from src.report_input import MAX_REPORT_BYTES
from src.upload_processor import (
    MAX_DOCX_BYTES,
    MAX_IMAGE_BYTES,
    UploadProcessingError,
    extract_non_pdf_text,
    inspect_upload,
)


def _pdf() -> bytes:
    document = fitz.open()
    document.new_page().insert_text((50, 50), "Synthetic pathology report")
    payload = document.tobytes()
    document.close()
    return payload


def _docx(text: str = "Synthetic pathology report text") -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _png() -> bytes:
    image = Image.new("RGB", (300, 100), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_inspection_reports_route_size_and_pdf_pages() -> None:
    text = inspect_upload("report.txt", b"Synthetic pathology report text")
    pdf = inspect_upload("report.pdf", _pdf())
    docx = inspect_upload("report.docx", _docx())
    image = inspect_upload("report.png", _png())
    assert text.kind == "txt" and text.processing_route == "Validated UTF-8 text"
    assert pdf.kind == "pdf" and pdf.page_count == 1
    assert docx.kind == "docx"
    assert image.kind == "image"


def test_txt_upload_uses_safe_utf8_and_size_validator() -> None:
    with pytest.raises(UploadProcessingError, match="500 KB"):
        inspect_upload("large.txt", b"A" * (MAX_REPORT_BYTES + 1))
    with pytest.raises(UploadProcessingError, match="UTF-8"):
        inspect_upload("report.txt", b"\xff\xfe\xfa")
    assert extract_non_pdf_text(
        "report.txt", b"Synthetic pathology report text"
    ) == "Synthetic pathology report text"


def test_docx_signature_structure_and_size_are_validated() -> None:
    with pytest.raises(UploadProcessingError, match="signature"):
        inspect_upload("report.docx", b"not a zip")
    with pytest.raises(UploadProcessingError, match="10 MB"):
        inspect_upload("report.docx", b"PK" + b"x" * MAX_DOCX_BYTES)
    assert "Synthetic pathology" in extract_non_pdf_text("report.docx", _docx())


def test_image_signature_and_size_are_validated() -> None:
    with pytest.raises(UploadProcessingError, match="signature"):
        inspect_upload("report.png", b"not an image")
    with pytest.raises(UploadProcessingError, match="15 MB"):
        inspect_upload("report.png", b"\x89PNG\r\n\x1a\n" + b"x" * MAX_IMAGE_BYTES)


def test_extension_allowlist_and_pdf_handoff() -> None:
    with pytest.raises(UploadProcessingError, match="TXT, PDF"):
        inspect_upload("report.exe", b"MZ")
    with pytest.raises(UploadProcessingError, match="page-aware"):
        extract_non_pdf_text("report.pdf", _pdf())
