from __future__ import annotations

from io import BytesIO

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

from src.pdf_processor import (
    PDFProcessingError,
    assemble_accepted_text,
    process_pdf,
    validate_pdf_upload,
)
from src.schemas import ProcessedDocument


def _searchable_pdf(*page_texts: str) -> bytes:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page(width=612, height=792)
        page.insert_textbox(fitz.Rect(50, 50, 560, 740), text, fontsize=12)
    payload = document.tobytes()
    document.close()
    return payload


def _scanned_pdf(text: str) -> bytes:
    image = Image.new("RGB", (1600, 1000), "white")
    ImageDraw.Draw(image).multiline_text((80, 80), text, fill="black", spacing=20)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    document = fitz.open()
    page = document.new_page(width=800, height=500)
    page.insert_image(page.rect, stream=buffer.getvalue())
    payload = document.tobytes()
    document.close()
    return payload


def test_rejects_invalid_signature_and_oversized_payload() -> None:
    with pytest.raises(PDFProcessingError, match="signature"):
        validate_pdf_upload("report.pdf", b"not-a-pdf")
    with pytest.raises(PDFProcessingError, match="25 MB"):
        validate_pdf_upload("report.pdf", b"%PDF-" + b"x" * (25 * 1024 * 1024))


def test_rejects_malformed_encrypted_and_excessive_page_count() -> None:
    with pytest.raises(PDFProcessingError, match="malformed|unreadable"):
        process_pdf("broken.pdf", b"%PDF-this-is-not-valid")

    protected = fitz.open()
    protected.new_page().insert_text((50, 50), "Protected pathology report")
    encrypted = protected.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    protected.close()
    with pytest.raises(PDFProcessingError, match="Password-protected|encrypted"):
        process_pdf("protected.pdf", encrypted)

    long_document = fitz.open()
    for _ in range(101):
        long_document.new_page()
    excessive = long_document.tobytes()
    long_document.close()
    with pytest.raises(PDFProcessingError, match="100-page"):
        process_pdf("long.pdf", excessive)


def test_searchable_pdf_uses_native_text_and_serializes() -> None:
    payload = _searchable_pdf(
        "FINAL DIAGNOSIS\nInvasive adenocarcinoma. Tumor size: 2.1 cm. "
        "Pathologic stage pT1c pN0."
    )
    result = process_pdf("case.pdf", payload)
    assert result.pages[0].extraction_method == "native_text"
    assert "adenocarcinoma" in result.pages[0].raw_text
    assert result.document_id.startswith("TCGA-PDF-")
    assert ProcessedDocument.model_validate_json(result.model_dump_json()) == result


def test_scanned_pdf_uses_ocr() -> None:
    result = process_pdf(
        "scan.pdf",
        _scanned_pdf("FINAL DIAGNOSIS\nADENOCARCINOMA\nTUMOR SIZE 2.1 CM\npT1c pN0"),
    )
    assert result.pages[0].extraction_method == "ocr"
    assert result.provenance.ocr_pages == 1
    assert result.pages[0].character_count > 20


def test_mixed_pdf_selects_ocr_only_for_image_page() -> None:
    native = fitz.open(stream=_searchable_pdf("A" * 250), filetype="pdf")
    scanned = fitz.open(stream=_scanned_pdf("SCANNED PAGE WITH PATHOLOGY TEXT"), filetype="pdf")
    native.insert_pdf(scanned)
    payload = native.tobytes()
    native.close()
    scanned.close()
    result = process_pdf("mixed.pdf", payload)
    assert [page.extraction_method for page in result.pages] == ["native_text", "ocr"]


def test_force_ocr_and_acceptance_gate() -> None:
    result = process_pdf("case.pdf", _searchable_pdf("B" * 250), force_ocr_all=True)
    assert result.pages[0].extraction_method == "ocr"
    with pytest.raises(PDFProcessingError, match="Accept every page"):
        assemble_accepted_text(result)
    result.pages[0].corrected_text = "Reviewer corrected authoritative text."
    result.pages[0].accepted = True
    text, ranges = assemble_accepted_text(result)
    assert text == "Reviewer corrected authoritative text."
    assert ranges == [(1, 0, len(text))]
