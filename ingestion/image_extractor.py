"""
Image extraction from PDFs with OCR and LLM captioning.
All image processing happens during preprocessing — no runtime vision agent.
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF

from config.settings import settings
from ingestion.ocr_engine import extract_text_from_image
from utils.llm import invoke_vision_llm

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """An image extracted from a PDF with OCR text and caption."""
    page_number: int
    image_index: int
    document_name: str
    image_path: str
    ocr_text: str = ""
    caption: str = ""
    combined_text: str = ""  # OCR + caption merged into a rich text chunk

    def __post_init__(self):
        """Build combined text from OCR and caption."""
        parts = []
        if self.caption:
            parts.append(f"Image description: {self.caption}")
        if self.ocr_text:
            parts.append(f"Text in image: {self.ocr_text}")
        if not parts:
            parts.append("(Image with no extractable text content)")

        self.combined_text = (
            f"[IMAGE from {self.document_name}, Page {self.page_number}]\n"
            + "\n".join(parts)
        )


def extract_images_from_pdf(
    pdf_path: str,
    output_dir: Optional[str] = None,
    use_vision_model: bool = True,
    min_image_size: int = 100,
) -> list[ExtractedImage]:
    """
    Extract images from a PDF, run OCR + optional LLM captioning.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted images (default: settings.image_dir)
        use_vision_model: Whether to generate captions using vision LLM
        min_image_size: Minimum image dimension in pixels (skip tiny icons)

    Returns:
        List of ExtractedImage objects with text content
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    doc_name = path.stem
    img_dir = Path(output_dir or settings.image_dir) / doc_name
    img_dir.mkdir(parents=True, exist_ok=True)

    extracted_images: list[ExtractedImage] = []

    try:
        doc = fitz.open(str(path))
        logger.info("Scanning for images — file=%s, pages=%d", doc_name, len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_idx, img_info in enumerate(image_list):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)

                    if base_image is None:
                        continue

                    image_bytes = base_image["image"]
                    image_ext = base_image.get("ext", "png")
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)

                    # Skip tiny images (likely icons/bullets)
                    if width < min_image_size or height < min_image_size:
                        continue

                    # Save image to disk
                    img_filename = f"{doc_name}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                    img_path = img_dir / img_filename

                    with open(img_path, "wb") as f:
                        f.write(image_bytes)

                    # Run OCR on the image
                    ocr_text = extract_text_from_image(str(img_path))

                    # Generate caption using vision model (if enabled)
                    caption = ""
                    if use_vision_model:
                        caption = _generate_caption(str(img_path), doc_name, page_num + 1)

                    extracted = ExtractedImage(
                        page_number=page_num + 1,
                        image_index=img_idx,
                        document_name=doc_name,
                        image_path=str(img_path),
                        ocr_text=ocr_text,
                        caption=caption,
                    )
                    extracted_images.append(extracted)

                    logger.debug(
                        "Image extracted — page=%d, idx=%d, size=%dx%d, ocr_len=%d",
                        page_num + 1, img_idx, width, height, len(ocr_text),
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to extract image on page %d, index %d: %s",
                        page_num + 1, img_idx, e,
                    )

        doc.close()
        logger.info(
            "Image extraction complete — doc=%s, images=%d",
            doc_name, len(extracted_images),
        )

    except Exception as e:
        logger.error("Failed to extract images from %s: %s", pdf_path, e)

    return extracted_images


def _generate_caption(image_path: str, doc_name: str, page: int) -> str:
    """Generate a descriptive caption for an image using the vision LLM."""
    try:
        prompt = (
            f"This image is from '{doc_name}' (page {page}), an ESG/sustainability report. "
            "Describe what this image shows in detail. If it's a chart or graph, "
            "describe the data it presents, including any trends, values, and labels. "
            "If it's a diagram, describe its structure and meaning. "
            "Be specific and factual — this description will be used for information retrieval."
        )
        caption = invoke_vision_llm(prompt, image_paths=[image_path])
        return caption.strip()[:1000]

    except Exception as e:
        logger.warning("Failed to generate caption for %s: %s", image_path, e)
        return ""
