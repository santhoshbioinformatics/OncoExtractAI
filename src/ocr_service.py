"""Local-only Tesseract OCR service for rendered PDF pages."""

from __future__ import annotations

from io import BytesIO

from PIL import Image
import re


class OCRServiceError(RuntimeError):
    """Raised when local OCR cannot be completed."""


class OCRService:
    """Run Tesseract locally; page image bytes are never persisted."""

    @staticmethod
    def available_languages() -> list[str]:
        try:
            import pytesseract

            return sorted(pytesseract.get_languages(config="")) or ["eng"]
        except Exception:
            return ["eng"]

    def extract_text(
        self,
        image_bytes: bytes,
        *,
        language: str = "eng",
        page_segmentation: int = 6,
        auto_orient: bool = False,
    ) -> str:
        if not image_bytes:
            raise OCRServiceError("The rendered page image is empty.")
        try:
            import pytesseract

            if page_segmentation not in {3, 4, 6, 11}:
                raise OCRServiceError("Unsupported OCR page segmentation mode.")
            languages = self.available_languages()
            if language not in languages:
                raise OCRServiceError(f"Tesseract language {language!r} is not installed.")
            with Image.open(BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
                if auto_orient:
                    try:
                        osd = pytesseract.image_to_osd(image)
                        match = re.search(r"Rotate:\s*(\d+)", osd)
                        rotation = int(match.group(1)) if match else 0
                        if rotation:
                            image = image.rotate(-rotation, expand=True, fillcolor="white")
                    except pytesseract.TesseractError:
                        pass
                return pytesseract.image_to_string(
                    image,
                    lang=language,
                    config=f"--psm {page_segmentation}",
                ).strip()
        except ImportError as error:
            raise OCRServiceError(
                "Local OCR is unavailable. Install Pillow, pytesseract, and Tesseract."
            ) from error
        except Exception as error:
            raise OCRServiceError(f"Local OCR failed: {error}") from error
