"""
PDF text extraction using PyMuPDF.
Extracts clean text per page with metadata tracking.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Extracted text from a single PDF page."""
    page_number: int
    text: str
    document_name: str
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


def extract_text_from_pdf(pdf_path: str) -> list[PageText]:
    """
    Extract clean text from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of PageText objects, one per page
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    doc_name = path.stem
    pages: list[PageText] = []

    try:
        doc = fitz.open(str(path))
        logger.info("Processing PDF — file=%s, pages=%d", doc_name, len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")

            # Clean the extracted text
            cleaned = _clean_text(raw_text)

            if cleaned.strip():
                pages.append(PageText(
                    page_number=page_num + 1,  # 1-indexed
                    text=cleaned,
                    document_name=doc_name,
                ))

        doc.close()
        total_chars = sum(p.char_count for p in pages)
        logger.info(
            "Text extraction complete — doc=%s, pages=%d, chars=%d",
            doc_name, len(pages), total_chars,
        )

    except Exception as e:
        logger.error("Failed to extract text from %s: %s", pdf_path, e)

    return pages


def _clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts.

    Handles:
    - Excessive whitespace
    - Page number artifacts
    - Header/footer repetitions
    - Hyphenated line breaks
    """
    import re

    # Rejoin hyphenated words at line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Replace multiple newlines with double newline (paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    # Remove common page number patterns at start/end of text
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Remove very short lines that are likely headers/footers (< 5 chars, standalone)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep lines that are either empty (paragraph break) or have content
        if stripped == "" or len(stripped) > 3:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
