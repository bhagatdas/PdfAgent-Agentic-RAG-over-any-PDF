"""
Retrieval Agent — advanced RAG with multi-query rewriting,
hybrid search, reranking, RAPTOR tree search, and map-reduce.
"""

import logging
import time
from langsmith import traceable

from graph.state import AgentState
from config.settings import settings
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.vector_store import vector_store
from utils.llm import invoke_llm

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="RetrievalAgent")
def retrieval_node(state: AgentState) -> dict:
    """
    Execute the retrieval strategy determined by the planner.
    Supports: standard (hybrid+rerank), raptor_global, map_reduce.
    """
    start = time.time()
    strategy = state.get("retrieval_strategy", "standard")
    query = state.get("rewritten_query", state.get("original_query", ""))

    logger.info("Retrieval starting — strategy=%s, query='%s...'", strategy, query[:60])

    if strategy == "none":
        return _empty_retrieval_result(start)
    elif strategy == "raptor_global":
        return _raptor_retrieval(query, start)
    elif strategy == "map_reduce":
        return _map_reduce_retrieval(query, state, start)
    else:
        return _standard_retrieval(query, start)


def _standard_retrieval(query: str, start_time: float) -> dict:
    """Standard hybrid retrieval with multi-query rewriting and reranking."""
    # Step 1: Multi-query rewriting — generate 3 query variants
    queries = _generate_query_variants(query)
    all_results = []

    # Step 2: Run hybrid search for each query variant
    for q in queries:
        results = hybrid_search(
            query=q,
            top_k=settings.retrieval_top_k,
            where={"chunk_type": {"$ne": "parent"}},  # Search children, not parents
        )
        all_results.extend(results)

    # Deduplicate by content
    seen = set()
    unique_results = []
    for r in all_results:
        content_key = r["content"][:200]
        if content_key not in seen:
            seen.add(content_key)
            unique_results.append(r)

    # Step 3: Rerank for precision
    reranked = rerank(
        query=query,
        results=unique_results,
        top_k=settings.rerank_top_k,
    )

    # Step 4: Expand context — fetch parent chunks for top results
    expanded = _expand_with_parents(reranked)

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        "Standard retrieval — queries=%d, raw=%d, unique=%d, reranked=%d, final=%d",
        len(queries), len(all_results), len(unique_results), len(reranked), len(expanded),
    )

    return {
        "retrieved_chunks": expanded,
        "retrieval_queries": queries,
        "raptor_level_used": 0,
        "execution_trace": [{
            "agent": "RetrievalAgent",
            "action": "standard_hybrid_retrieval",
            "input_summary": f"query='{query[:80]}', variants={len(queries)}",
            "output_summary": f"chunks={len(expanded)}, top_score={expanded[0].get('rerank_score', 'N/A') if expanded else 'N/A'}",
            "duration_ms": round(elapsed, 1),
        }],
    }


def _raptor_retrieval(query: str, start_time: float) -> dict:
    """Search RAPTOR tree summaries for global queries."""
    # Search higher levels (L1-L3) for broad themes
    global_results = vector_store.search_by_raptor_level(
        query=query,
        min_level=1,
        max_level=3,
        top_k=8,
    )

    # Also get some Level 0 (leaf) results for supporting detail
    local_results = hybrid_search(
        query=query,
        top_k=5,
        where={"raptor_level": 0},
    )

    combined = global_results + local_results

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        "RAPTOR retrieval — global=%d (L1-L3), local=%d (L0), total=%d",
        len(global_results), len(local_results), len(combined),
    )

    return {
        "retrieved_chunks": combined,
        "retrieval_queries": [query],
        "raptor_level_used": 2,
        "execution_trace": [{
            "agent": "RetrievalAgent",
            "action": "raptor_global_search",
            "input_summary": f"query='{query[:80]}'",
            "output_summary": f"global={len(global_results)}, local={len(local_results)}",
            "duration_ms": round(elapsed, 1),
        }],
    }


def _map_reduce_retrieval(query: str, state: AgentState, start_time: float) -> dict:
    """
    Map-Reduce: scan ALL chunks for exhaustive aggregation queries.
    E.g., "List ALL sustainability initiatives mentioned in the report."
    """
    entities = state.get("extracted_entities", [])
    target = ", ".join(entities) if entities else "relevant items"

    all_docs = vector_store.get_all_documents()

    if not all_docs:
        return _empty_retrieval_result(start_time)

    logger.info("Map-Reduce starting — scanning %d chunks for '%s'", len(all_docs), target)

    extraction_prompt = f"""From the following text, extract any {target} that are explicitly mentioned.
Return ONLY the extracted items as a simple list, one per line. If none found, respond with "NONE".

Text:
{{chunk_text}}

Extracted items:"""

    # MAP phase: extract from each chunk (batched)
    all_extracted = []
    source_chunks = []
    batch_size = 5

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        for doc in batch:
            try:
                result = invoke_llm(
                    extraction_prompt.format(chunk_text=doc["content"][:1000]),
                    task_type="light",
                )
                if result.strip() and result.strip().upper() != "NONE":
                    items = [item.strip("- ").strip() for item in result.strip().split("\n") if item.strip()]
                    all_extracted.extend(items)
                    source_chunks.append(doc)
            except Exception as e:
                logger.debug("Map-reduce extraction error: %s", e)

    # REDUCE phase: deduplicate
    unique_items = list(set(all_extracted))
    unique_items.sort()

    elapsed = (time.time() - start_time) * 1000

    logger.info(
        "Map-Reduce complete — scanned=%d, extracted=%d, unique=%d, time=%.1fs",
        len(all_docs), len(all_extracted), len(unique_items), elapsed / 1000,
    )

    # Return the aggregated results + supporting chunks
    return {
        "retrieved_chunks": source_chunks[:10],  # Top supporting chunks
        "retrieval_queries": [query],
        "map_reduce_results": unique_items,
        "raptor_level_used": 0,
        "execution_trace": [{
            "agent": "RetrievalAgent",
            "action": "map_reduce_exhaustive",
            "input_summary": f"query='{query[:80]}', chunks_scanned={len(all_docs)}",
            "output_summary": f"extracted={len(all_extracted)}, unique={len(unique_items)}",
            "duration_ms": round(elapsed, 1),
        }],
    }


def _generate_query_variants(query: str) -> list[str]:
    """
    Generate multiple query variants for multi-query retrieval.
    Improves recall by searching from different angles.
    """
    try:
        prompt = f"""Generate 2 alternative versions of this query for better search coverage.
Each version should approach the question from a different angle or use different keywords.
Return ONLY the queries, one per line. Do not include numbering.

Original: {query}

Alternative queries:"""

        response = invoke_llm(prompt, task_type="light")
        variants = [q.strip() for q in response.strip().split("\n") if q.strip() and len(q.strip()) > 10]
        # Always include original query
        return [query] + variants[:2]

    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [query]


def _expand_with_parents(chunks: list[dict]) -> list[dict]:
    """Expand top chunks by fetching their parent chunks for additional context."""
    expanded = []
    seen_parents = set()

    for chunk in chunks:
        expanded.append(chunk)
        parent_id = chunk.get("metadata", {}).get("parent_id", "")

        if parent_id and parent_id not in seen_parents:
            parent = vector_store.get_parent_chunk(parent_id)
            if parent:
                parent["is_parent_context"] = True
                expanded.append(parent)
                seen_parents.add(parent_id)

    return expanded


def _empty_retrieval_result(start_time: float) -> dict:
    """Return empty retrieval result."""
    elapsed = (time.time() - start_time) * 1000
    return {
        "retrieved_chunks": [],
        "retrieval_queries": [],
        "raptor_level_used": -1,
        "execution_trace": [{
            "agent": "RetrievalAgent",
            "action": "no_retrieval_needed",
            "input_summary": "strategy=none",
            "output_summary": "skipped",
            "duration_ms": round(elapsed, 1),
        }],
    }
