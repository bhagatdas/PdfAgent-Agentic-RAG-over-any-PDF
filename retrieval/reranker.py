"""
Cross-encoder reranking for precision filtering.
Re-scores retrieved candidates against the query using a cross-encoder model.
"""

import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

# Lazy-load to avoid slow import
_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        logger.info("Loading cross-encoder reranker: %s", model_name)
        _reranker = CrossEncoder(model_name, max_length=512)
    return _reranker


def rerank(
    query: str,
    results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rerank search results using a cross-encoder model.

    The cross-encoder scores (query, document) pairs directly,
    which is more accurate than cosine similarity but slower.
    This is the "precision layer" after initial broad retrieval.

    Args:
        query: Original search query
        results: List of result dicts (must have 'content' field)
        top_k: Number of top results to return after reranking

    Returns:
        Reranked and filtered list of results
    """
    if not results:
        return []

    if len(results) <= top_k:
        return results

    try:
        reranker = _get_reranker()

        # Create (query, document) pairs for scoring
        pairs = [(query, r["content"][:512]) for r in results]  # Truncate to model limit

        # Score all pairs
        scores = reranker.predict(pairs)

        # Attach scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "Reranking — input=%d, output=%d, top_score=%.3f, bottom_score=%.3f",
            len(results),
            top_k,
            reranked[0]["rerank_score"] if reranked else 0,
            reranked[min(top_k, len(reranked)) - 1]["rerank_score"] if reranked else 0,
        )

        return reranked[:top_k]

    except Exception as e:
        logger.error("Reranking failed, returning original results: %s", e)
        return results[:top_k]
