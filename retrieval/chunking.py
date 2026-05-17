"""
Parent-child chunking strategy.
Creates small child chunks (for precise retrieval) linked to larger parent chunks (for context).
"""

import logging
import uuid
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


def create_parent_child_chunks(
    text: str,
    document_name: str,
    page_number: int,
    child_size: Optional[int] = None,
    parent_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[dict]:
    """
    Split text into parent-child chunks.

    - Parent chunks: ~1600 chars, provide full context for LLM
    - Child chunks: ~400 chars, used for precise vector search
    - Each child stores its parent_id for context retrieval

    Args:
        text: Source text to chunk
        document_name: Name of source document
        page_number: Page number in source document
        child_size: Child chunk size in characters
        parent_size: Parent chunk size in characters
        overlap: Overlap between child chunks

    Returns:
        List of chunk dicts with content, metadata, and parent-child relationships
    """
    child_size = child_size or settings.chunk_size_child
    parent_size = parent_size or settings.chunk_size_parent
    overlap = overlap or settings.chunk_overlap

    if not text.strip():
        return []

    chunks = []

    # First, create parent chunks
    parent_chunks = _split_text(text, parent_size, overlap=overlap * 2)

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"{document_name}_p{page_number}_parent{p_idx}"

        # Store the parent chunk (used for context expansion)
        chunks.append({
            "content": parent_text,
            "document_name": document_name,
            "page_number": page_number,
            "chunk_type": "parent",
            "parent_id": parent_id,
            "chunk_id": parent_id,
            "raptor_level": -1,  # -1 = parent, 0 = child
        })

        # Create child chunks from this parent
        child_chunks = _split_text(parent_text, child_size, overlap=overlap)

        for c_idx, child_text in enumerate(child_chunks):
            child_id = f"{document_name}_p{page_number}_parent{p_idx}_child{c_idx}"

            chunks.append({
                "content": child_text,
                "document_name": document_name,
                "page_number": page_number,
                "chunk_type": "child",
                "parent_id": parent_id,
                "chunk_id": child_id,
                "raptor_level": 0,  # Level 0 = leaf node
            })

    logger.debug(
        "Chunked page %d — parents=%d, children=%d",
        page_number,
        len([c for c in chunks if c["chunk_type"] == "parent"]),
        len([c for c in chunks if c["chunk_type"] == "child"]),
    )

    return chunks


def _split_text(text: str, chunk_size: int, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks, preferring sentence boundaries.

    Args:
        text: Text to split
        chunk_size: Target chunk size in characters
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    # Split by sentences first
    sentences = _split_into_sentences(text)

    chunks = []
    current_chunk = ""
    current_sentences = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence
            current_sentences.append(sentence)
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Calculate overlap: keep last N characters worth of sentences
            overlap_text = ""
            for s in reversed(current_sentences):
                if len(overlap_text) + len(s) <= overlap:
                    overlap_text = s + overlap_text
                else:
                    break

            current_chunk = overlap_text + sentence
            current_sentences = [sentence]

    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving the delimiter."""
    import re
    # Split on period, question mark, exclamation mark, or newline
    # But preserve the delimiter with the sentence
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    return [s for s in sentences if s.strip()]
