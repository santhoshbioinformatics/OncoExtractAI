"""Local-only Tesseract OCR service for rendered PDF pages."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


class OCRServiceError(RuntimeError):
    """Raised when local OCR cannot be completed."""


class OCRService:
    """Run Tesseract locally; page image bytes are never persisted."""

    def extract_text(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise OCRServiceError("The rendered page image is empty.")
        try:
            import pytesseract

            with Image.open(BytesIO(image_bytes)) as image:
                return pytesseract.image_to_string(image, config="--psm 6").strip()
        except ImportError as error:
            raise OCRServiceError(
                "Local OCR is unavailable. Install Pillow, pytesseract, and Tesseract."
            ) from error
        except Exception as error:
            raise OCRServiceError(f"Local OCR failed: {error}") from error
