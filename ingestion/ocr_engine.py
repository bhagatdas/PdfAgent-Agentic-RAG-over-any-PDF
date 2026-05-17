"""
OCR engine using EasyOCR — pip-installable, no system dependencies.
Extracts text from images (charts, scanned pages, embedded graphics).
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-load EasyOCR to avoid slow import at startup
_reader = None


def _get_reader(languages: Optional[list[str]] = None):
    """Get or create the EasyOCR reader (lazy singleton)."""
    global _reader
    if _reader is None:
        import easyocr
        langs = languages or ["en"]
        logger.info("Initializing EasyOCR reader — languages=%s", langs)
        _reader = easyocr.Reader(langs, gpu=False)  # CPU mode for compatibility
    return _reader


def extract_text_from_image(
    image_path: str,
    languages: Optional[list[str]] = None,
    detail: int = 0,
) -> str:
    """
    Extract text from an image using EasyOCR.

    Args:
        image_path: Path to the image file
        languages: List of language codes (default: ["en"])
        detail: 0 = text only, 1 = include bounding boxes and confidence

    Returns:
        Extracted text as a single string
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Image not found for OCR: %s", image_path)
        return ""

    try:
        reader = _get_reader(languages)
        results = reader.readtext(str(path), detail=detail)

        if detail == 0:
            # results is a list of strings
            text = "\n".join(results)
        else:
            # results is list of (bbox, text, confidence)
            text = "\n".join([r[1] for r in results])

        logger.debug("OCR extracted %d chars from %s", len(text), path.name)
        return text.strip()

    except Exception as e:
        logger.error("OCR failed for %s: %s", image_path, e)
        return ""


def extract_text_from_image_bytes(
    image_bytes: bytes,
    languages: Optional[list[str]] = None,
) -> str:
    """
    Extract text from image bytes (useful when image is already in memory).

    Args:
        image_bytes: Raw image bytes
        languages: List of language codes

    Returns:
        Extracted text as string
    """
    try:
        reader = _get_reader(languages)
        results = reader.readtext(image_bytes, detail=0)
        return "\n".join(results).strip()
    except Exception as e:
        logger.error("OCR from bytes failed: %s", e)
        return ""
