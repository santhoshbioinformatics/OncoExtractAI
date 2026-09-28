"""Validation and local text extraction for non-PDF report uploads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal
from zipfile import BadZipFile, ZipFile

from PIL import Image, UnidentifiedImageError

from .ocr_service import OCRService, OCRServiceError
from .pdf_processor import PDFProcessingError, inspect_pdf_page_count
from .report_input import ReportInputError, decode_uploaded_text

MAX_DOCX_BYTES = 10 * 1024 * 1024
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

UploadKind = Literal["txt", "pdf", "docx", "image"]


class UploadProcessingError(ValueError):
    """Safe reviewer-facing upload validation or extraction failure."""


@dataclass(frozen=True)
class UploadInspection:
    kind: UploadKind
    extension: str
    size_bytes: int
    processing_route: str
    page_count: int | None = None

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


def _suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx", ".jpg", ".jpeg", ".png"}:
        raise UploadProcessingError("Choose a TXT, PDF, DOCX, JPG, JPEG, or PNG file.")
    return suffix


def _validate_docx(payload: bytes) -> None:
    if not payload:
        raise UploadProcessingError("The uploaded DOCX file is empty.")
    if len(payload) > MAX_DOCX_BYTES:
        raise UploadProcessingError("The DOCX file exceeds the 10 MB local limit.")
    if not payload.startswith(b"PK"):
        raise UploadProcessingError("The file signature does not identify a valid DOCX file.")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            names = set(archive.namelist())
    except BadZipFile as error:
        raise UploadProcessingError("The DOCX file is malformed or unreadable.") from error
    if not {"[Content_Types].xml", "word/document.xml"}.issubset(names):
        raise UploadProcessingError("The archive does not contain a valid Word document.")


def _validate_image(suffix: str, payload: bytes) -> None:
    if not payload:
        raise UploadProcessingError("The uploaded image is empty.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise UploadProcessingError("The image exceeds the 15 MB local limit.")
    valid_signature = (
        suffix == ".png" and payload.startswith(b"\x89PNG\r\n\x1a\n")
    ) or (
        suffix in {".jpg", ".jpeg"} and payload.startswith(b"\xff\xd8\xff")
    )
    if not valid_signature:
        raise UploadProcessingError("The file signature does not match its image extension.")
    try:
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise UploadProcessingError("The image is malformed or unreadable.") from error
    if width * height > MAX_IMAGE_PIXELS:
        raise UploadProcessingError("The image exceeds the 40-megapixel safety limit.")


def inspect_upload(filename: str, payload: bytes) -> UploadInspection:
    """Validate an upload and describe the route before processing begins."""

    suffix = _suffix(filename)
    if suffix == ".txt":
        try:
            decode_uploaded_text(filename, payload)
        except ReportInputError as error:
            raise UploadProcessingError(str(error)) from error
        return UploadInspection("txt", suffix, len(payload), "Validated UTF-8 text")
    if suffix == ".pdf":
        try:
            pages = inspect_pdf_page_count(filename, payload)
        except PDFProcessingError as error:
            raise UploadProcessingError(str(error)) from error
        return UploadInspection(
            "pdf", suffix, len(payload), "Native PDF text with selective local OCR", pages
        )
    if suffix == ".docx":
        _validate_docx(payload)
        return UploadInspection("docx", suffix, len(payload), "Local Word text extraction")
    _validate_image(suffix, payload)
    return UploadInspection("image", suffix, len(payload), "Local Tesseract OCR")


def extract_non_pdf_text(filename: str, payload: bytes) -> str:
    """Extract already-validated TXT, DOCX, or image content locally."""

    inspection = inspect_upload(filename, payload)
    if inspection.kind == "pdf":
        raise UploadProcessingError("PDFs must use the page-aware PDF review workflow.")
    if inspection.kind == "txt":
        try:
            return decode_uploaded_text(filename, payload)
        except ReportInputError as error:
            raise UploadProcessingError(str(error)) from error
    if inspection.kind == "docx":
        try:
            import docx

            document = docx.Document(BytesIO(payload))
            text = "\n\n".join(
                paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
            ).strip()
        except Exception as error:
            raise UploadProcessingError(f"Word text extraction failed: {error}") from error
    else:
        try:
            text = OCRService().extract_text(payload)
        except OCRServiceError as error:
            raise UploadProcessingError(str(error)) from error
    if not text.strip():
        raise UploadProcessingError("No report text could be extracted from this file.")
    return text.strip()
