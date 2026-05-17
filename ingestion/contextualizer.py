"""
Contextual chunk enrichment (Anthropic-style Contextual Retrieval).
Prepends each chunk with a 2-3 sentence context that situates it within the document.
Reduces retrieval failures by ~49%.
"""

import logging
from dataclasses import dataclass

from utils.llm import invoke_llm

logger = logging.getLogger(__name__)

CONTEXTUALIZE_PROMPT = """You are given a chunk from a document. Provide a short (2-3 sentence) context
that situates this chunk within the broader document. Include: what document this is from,
what section/topic is being discussed, and any critical context needed to understand this chunk in isolation.

Document: {doc_title}
Page: {page_number}

Surrounding text (before this chunk):
{preceding_text}

CHUNK TO CONTEXTUALIZE:
{chunk_text}

CONTEXT (2-3 sentences, be concise):"""


def contextualize_chunk(
    chunk_text: str,
    doc_title: str,
    page_number: int,
    preceding_text: str = "",
) -> str:
    """
    Generate a contextual prefix for a chunk and prepend it.

    Args:
        chunk_text: The raw chunk text
        doc_title: Source document title
        page_number: Page number
        preceding_text: Text that comes before this chunk (for context)

    Returns:
        Contextualized chunk: "[Context: ...] Original chunk text"
    """
    try:
        prompt = CONTEXTUALIZE_PROMPT.format(
            doc_title=doc_title,
            page_number=page_number,
            preceding_text=preceding_text[:500] if preceding_text else "(start of document)",
            chunk_text=chunk_text[:800],
        )

        context = invoke_llm(prompt, task_type="light")
        context = context.strip()

        if context:
            return f"[Context: {context}]\n{chunk_text}"
        else:
            return chunk_text

    except Exception as e:
        logger.warning("Failed to contextualize chunk: %s", e)
        return chunk_text


def contextualize_chunks_batch(
    chunks: list[dict],
    doc_title: str,
) -> list[dict]:
    """
    Contextualize a batch of chunks from the same document.

    Args:
        chunks: List of chunk dicts with 'content', 'page_number', 'metadata'
        doc_title: Source document title

    Returns:
        Same chunks with 'content' field updated to include context prefix
    """
    logger.info("Contextualizing %d chunks for '%s'", len(chunks), doc_title)

    contextualized = []
    for i, chunk in enumerate(chunks):
        # Get preceding text from previous chunk for context
        preceding = chunks[i - 1]["content"][-300:] if i > 0 else ""

        new_content = contextualize_chunk(
            chunk_text=chunk["content"],
            doc_title=doc_title,
            page_number=chunk.get("page_number", 0),
            preceding_text=preceding,
        )

        updated_chunk = {**chunk, "content": new_content}
        contextualized.append(updated_chunk)

        if (i + 1) % 20 == 0:
            logger.info("Contextualized %d/%d chunks", i + 1, len(chunks))

    logger.info("Contextualization complete — %d chunks processed", len(contextualized))
    return contextualized
