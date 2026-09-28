"""Secure, page-aware PDF text extraction with selective local OCR."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Iterable, Mapping

from .ocr_service import OCRService, OCRServiceError
from .schemas import (
    DocumentProvenance,
    OCRQualityAssessment,
    ProcessedDocument,
    ProcessedPage,
)
from .text_normalizer import normalize_ocr_text

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 100
PROCESSOR_VERSION = "pymupdf-tesseract-1.0"


class PDFProcessingError(ValueError):
    """Raised for rejected or unreadable PDF input."""


def validate_pdf_upload(filename: str, payload: bytes) -> None:
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Choose a file with a .pdf extension.")
    if not payload:
        raise PDFProcessingError("The uploaded PDF is empty.")
    if len(payload) > MAX_PDF_BYTES:
        raise PDFProcessingError("The PDF exceeds the 25 MB local processing limit.")
    if not payload.startswith(b"%PDF-"):
        raise PDFProcessingError("The file signature does not identify a valid PDF.")


def inspect_pdf_page_count(filename: str, payload: bytes) -> int:
    """Validate a PDF and return its page count without extracting report text."""

    validate_pdf_upload(filename, payload)
    try:
        import pymupdf as fitz

        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as error:
        raise PDFProcessingError(f"The PDF is malformed or unreadable: {error}") from error
    with document:
        if document.needs_pass:
            raise PDFProcessingError("Password-protected or encrypted PDFs are not supported.")
        if document.page_count == 0:
            raise PDFProcessingError("The PDF contains no pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise PDFProcessingError("The PDF exceeds the 100-page local processing limit.")
        return document.page_count


def assess_text_quality(text: str) -> OCRQualityAssessment:
    stripped = text.strip()
    reasons: list[str] = []
    if not stripped:
        return OCRQualityAssessment(label="Poor", score=0.0, reasons=["No text was extracted."])
    visible = [char for char in stripped if not char.isspace()]
    alpha_numeric_ratio = sum(char.isalnum() for char in visible) / max(1, len(visible))
    replacement_count = stripped.count("�")
    if len(stripped) < 40:
        reasons.append("Very little text was extracted.")
    if alpha_numeric_ratio < 0.55:
        reasons.append("The text contains a high proportion of non-alphanumeric symbols.")
    if replacement_count:
        reasons.append("Unreadable replacement characters were detected.")
    if re.search(r"\bsynoptic\s+report\b", stripped, re.IGNORECASE) and re.search(
        r"\b(?:lymph\s+node\s+involvement|pathologic(?:al)?\s+stag(?:e|ing)|pTNM)\s*:\s*$",
        stripped,
        re.IGNORECASE,
    ):
        reasons.append(
            "Synoptic text ends at a staging label; verify that the continuation was captured."
        )
    score = min(1.0, len(stripped) / 200) * min(1.0, alpha_numeric_ratio / 0.8)
    score = max(0.0, score - min(0.4, replacement_count * 0.05))
    if not reasons and score >= 0.75:
        label = "Good"
    elif score >= 0.35:
        label = "Review recommended"
        if not reasons:
            reasons.append("Extracted text is readable but should be checked against the page.")
    else:
        label = "Poor"
    return OCRQualityAssessment(label=label, score=round(score, 3), reasons=reasons)


def _render_page(page: object, dpi: int = 300, rotation_degrees: int = 0) -> bytes:
    import pymupdf as fitz

    matrix = fitz.Matrix(dpi / 72, dpi / 72).prerotate(rotation_degrees)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png")


def render_pdf_page(
    payload: bytes, page_number: int, dpi: int = 150, rotation_degrees: int = 0
) -> bytes:
    """Render one page for an in-memory reviewer preview."""

    import pymupdf as fitz

    with fitz.open(stream=payload, filetype="pdf") as document:
        if page_number < 1 or page_number > document.page_count:
            raise PDFProcessingError("Page number is outside the document.")
        if rotation_degrees not in {0, 90, 180, 270}:
            raise PDFProcessingError("Page rotation must be 0, 90, 180, or 270 degrees.")
        return _render_page(
            document[page_number - 1], dpi=dpi, rotation_degrees=rotation_degrees
        )


def process_pdf(
    filename: str,
    payload: bytes,
    *,
    force_ocr_pages: Iterable[int] = (),
    force_ocr_all: bool = False,
    ocr_service: OCRService | None = None,
    ocr_language: str = "eng",
    ocr_page_segmentation: int = 6,
    auto_orient: bool = False,
    page_rotations: Mapping[int, int] | None = None,
) -> ProcessedDocument:
    """Extract native text first and OCR only poor or explicitly selected pages."""

    validate_pdf_upload(filename, payload)
    try:
        import pymupdf as fitz
    except ImportError as error:
        raise PDFProcessingError("PyMuPDF is required for PDF processing.") from error

    forced = set(force_ocr_pages)
    rotations = dict(page_rotations or {})
    digest_hex = hashlib.sha256(payload).hexdigest()
    try:
        document = fitz.open(stream=payload, filetype="pdf")
    except Exception as error:
        raise PDFProcessingError(f"The PDF is malformed or unreadable: {error}") from error
    with document:
        if document.needs_pass:
            raise PDFProcessingError("Password-protected or encrypted PDFs are not supported.")
        if document.page_count == 0:
            raise PDFProcessingError("The PDF contains no pages.")
        if document.page_count > MAX_PDF_PAGES:
            raise PDFProcessingError("The PDF exceeds the 100-page local processing limit.")
        invalid_forced = forced - set(range(1, document.page_count + 1))
        if invalid_forced:
            raise PDFProcessingError("A requested OCR page is outside the document.")
        if any(page < 1 or page > document.page_count for page in rotations):
            raise PDFProcessingError("A rotated page is outside the document.")
        if any(rotation not in {0, 90, 180, 270} for rotation in rotations.values()):
            raise PDFProcessingError("Page rotation must be 0, 90, 180, or 270 degrees.")

        service = ocr_service or OCRService()
        pages: list[ProcessedPage] = []
        for index, page in enumerate(document, start=1):
            native_text = page.get_text("text").strip()
            native_quality = assess_text_quality(native_text)
            use_ocr = force_ocr_all or index in forced or native_quality.label == "Poor"
            method = "ocr" if use_ocr else "native_text"
            rotation = rotations.get(index, 0)
            try:
                raw_text = (
                    service.extract_text(
                        _render_page(page, rotation_degrees=rotation),
                        language=ocr_language,
                        page_segmentation=ocr_page_segmentation,
                        auto_orient=auto_orient,
                    )
                    if use_ocr else native_text
                )
            except OCRServiceError as error:
                raise PDFProcessingError(str(error)) from error
            normalized, transformations = normalize_ocr_text(raw_text)
            quality = assess_text_quality(raw_text)
            flags = list(quality.reasons)
            if use_ocr and native_text and index not in forced and not force_ocr_all:
                flags.append("Native text was poor, so local OCR was used automatically.")
            pages.append(ProcessedPage(
                page_number=index,
                extraction_method=method,
                raw_text=raw_text,
                normalized_text=normalized,
                character_count=len(raw_text),
                quality=quality,
                warning_flags=flags,
                transformations=transformations,
                rotation_degrees=rotation,
                ocr_language=ocr_language if use_ocr else None,
                ocr_page_segmentation=ocr_page_segmentation if use_ocr else None,
            ))

    native_count = sum(page.extraction_method == "native_text" for page in pages)
    ocr_count = len(pages) - native_count
    return ProcessedDocument(
        document_id=f"TCGA-PDF-{digest_hex[:12].upper()}",
        pages=pages,
        provenance=DocumentProvenance(
            source_report_digest=f"sha256:{digest_hex}",
            extraction_timestamp=datetime.now(timezone.utc),
            processor_version=PROCESSOR_VERSION,
            page_count=len(pages),
            native_text_pages=native_count,
            ocr_pages=ocr_count,
        ),
    )


def assemble_accepted_text(document: ProcessedDocument) -> tuple[str, list[tuple[int, int, int]]]:
    """Join accepted page text and return page-aware character ranges."""

    if not all(page.accepted for page in document.pages):
        raise PDFProcessingError("Accept every page before continuing to abstraction.")
    chunks: list[str] = []
    ranges: list[tuple[int, int, int]] = []
    cursor = 0
    for page in document.pages:
        if page.excluded_as_blank:
            continue
        text = page.authoritative_text.strip()
        if not text:
            raise PDFProcessingError(f"Page {page.page_number} has no accepted text.")
        if chunks:
            chunks.append("\n\n")
            cursor += 2
        start = cursor
        chunks.append(text)
        cursor += len(text)
        ranges.append((page.page_number, start, cursor))
    if not chunks:
        raise PDFProcessingError("At least one accepted page must contain report text.")
    return "".join(chunks), ranges


def merge_reprocessed_document(
    existing: ProcessedDocument,
    refreshed: ProcessedDocument,
    page_numbers: Iterable[int],
) -> ProcessedDocument:
    """Merge new OCR output while preserving reviewer corrections for re-verification."""

    if existing.document_id != refreshed.document_id:
        raise PDFProcessingError("Cannot merge OCR output from a different document.")
    selected = set(page_numbers)
    merged = existing.model_copy(deep=True)
    for page_number in selected:
        if page_number < 1 or page_number > len(merged.pages):
            raise PDFProcessingError("A reprocessed page is outside the document.")
        previous = merged.pages[page_number - 1]
        replacement = refreshed.pages[page_number - 1].model_copy(deep=True)
        replacement.corrected_text = previous.corrected_text
        if previous.excluded_as_blank:
            replacement.accepted = True
            replacement.excluded_as_blank = True
        else:
            replacement.accepted = False
        merged.pages[page_number - 1] = replacement
    merged.provenance.native_text_pages = sum(
        page.extraction_method == "native_text" for page in merged.pages
    )
    merged.provenance.ocr_pages = len(merged.pages) - merged.provenance.native_text_pages
    return merged


def page_for_offset(offset: int, ranges: list[tuple[int, int, int]]) -> int | None:
    for page_number, start, end in ranges:
        if start <= offset < end:
            return page_number
    return None
