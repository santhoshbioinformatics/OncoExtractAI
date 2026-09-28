"""Conservative OCR cleanup with a complete transformation record."""

from __future__ import annotations

import re
import unicodedata

from .schemas import TextTransformation


def normalize_ocr_text(text: str) -> tuple[str, list[TextTransformation]]:
    """Normalize layout noise without changing clinical values or punctuation."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    transformations: list[TextTransformation] = []
    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")

    control_count = sum(
        1 for char in normalized
        if char not in {"\n", "\t"} and unicodedata.category(char) == "Cc"
    )
    if control_count:
        normalized = "".join(
            char for char in normalized
            if char in {"\n", "\t"} or unicodedata.category(char) != "Cc"
        )
        transformations.append(TextTransformation(
            transformation="remove_control_characters",
            count=control_count,
            description="Removed non-printing control characters; clinical text was unchanged.",
        ))

    trailing_matches = re.findall(r"[ \t]+(?=\n|$)", normalized)
    if trailing_matches:
        normalized = re.sub(r"[ \t]+(?=\n|$)", "", normalized)
        transformations.append(TextTransformation(
            transformation="trim_trailing_whitespace",
            count=len(trailing_matches),
            description="Removed trailing spaces at line endings.",
        ))

    blank_runs = re.findall(r"\n{4,}", normalized)
    if blank_runs:
        normalized = re.sub(r"\n{4,}", "\n\n\n", normalized)
        transformations.append(TextTransformation(
            transformation="collapse_excess_blank_lines",
            count=len(blank_runs),
            description="Collapsed runs of more than two blank lines.",
        ))
    return normalized.strip(), transformations
