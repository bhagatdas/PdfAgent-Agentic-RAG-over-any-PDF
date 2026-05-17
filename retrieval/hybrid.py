"""
Hybrid retrieval combining semantic (vector) and keyword (BM25) search.
Uses Reciprocal Rank Fusion (RRF) to merge results.
"""

import logging
from typing import Optional

from retrieval.vector_store import vector_store
from retrieval.bm25_retriever import bm25_index

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    top_k: int = 20,
    semantic_weight: float = 0.6,
    bm25_weight: float = 0.4,
    rrf_k: int = 60,
    where: Optional[dict] = None,
) -> list[dict]:
    """
    Perform hybrid search combining semantic and BM25 retrieval with RRF fusion.

    Args:
        query: Search query
        top_k: Number of final results to return
        semantic_weight: Weight for semantic search in RRF
        bm25_weight: Weight for BM25 search in RRF
        rrf_k: RRF constant (higher = less emphasis on top ranks)
        where: Optional metadata filter for vector search

    Returns:
        Merged, deduplicated, and ranked list of results
    """
    # Run both searches
    semantic_results = vector_store.search(query, top_k=top_k * 2, where=where)
    bm25_results = bm25_index.search(query, top_k=top_k * 2) if bm25_index.is_ready else []

    # Apply Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(
        result_lists=[semantic_results, bm25_results],
        weights=[semantic_weight, bm25_weight],
        k=rrf_k,
    )

    # Take top-K
    final = fused[:top_k]

    logger.info(
        "Hybrid search — semantic=%d, bm25=%d, fused=%d, final=%d",
        len(semantic_results), len(bm25_results), len(fused), len(final),
    )

    return final


def _reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    weights: list[float],
    k: int = 60,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = sum(weight_i / (k + rank_i(doc)))

    Args:
        result_lists: List of ranked result lists
        weights: Weight for each result list
        k: RRF constant

    Returns:
        Merged and re-ranked list of results
    """
    # Build a map of doc_id → aggregated score + best content
    doc_scores: dict[str, float] = {}
    doc_data: dict[str, dict] = {}

    for list_idx, results in enumerate(result_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0

        for rank, result in enumerate(results):
            doc_id = result.get("id", result.get("content", "")[:100])

            # RRF score contribution
            rrf_score = weight / (k + rank + 1)

            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
            else:
                doc_scores[doc_id] = rrf_score
                doc_data[doc_id] = result

    # Sort by aggregated RRF score (descending)
    sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    # Build final result list
    merged = []
    for doc_id in sorted_ids:
        result = doc_data[doc_id].copy()
        result["rrf_score"] = doc_scores[doc_id]
        merged.append(result)

    return merged
