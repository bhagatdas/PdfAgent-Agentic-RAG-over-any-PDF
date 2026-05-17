"""
Memory Agent — reads/writes LangGraph Store for cross-session long-term memory.
Short-term memory is handled automatically by the SqliteSaver checkpointer.
"""

import logging
import time
from datetime import datetime
from langsmith import traceable
from langgraph.store.base import BaseStore

from graph.state import AgentState

logger = logging.getLogger(__name__)


@traceable(run_type="chain", name="MemoryRead")
def memory_read_node(state: AgentState, *, store: BaseStore, config: dict) -> dict:
    """
    Read relevant past interactions from long-term memory.
    Uses LangGraph Store's semantic search (namespaced by user_id).
    """
    start = time.time()
    query = state.get("rewritten_query", state.get("original_query", ""))
    user_id = config.get("configurable", {}).get("user_id", "default")

    memory_context = ""
    past_interactions = []

    try:
        namespace = ("user_memory", user_id)
        # Search for relevant past interactions
        results = store.search(namespace, query=query, limit=5)

        if results:
            memory_items = []
            for item in results:
                val = item.value
                memory_items.append(
                    f"Q: {val.get('query', '?')}\nA: {val.get('answer_summary', '?')}"
                )
                past_interactions.append(val)

            memory_context = "RELEVANT PAST INTERACTIONS:\n" + "\n---\n".join(memory_items)
            logger.info("Memory: found %d relevant past interactions", len(results))
        else:
            memory_context = "(No relevant past interactions found)"

    except Exception as e:
        logger.debug("Memory read skipped: %s", e)
        memory_context = "(Memory not available)"

    elapsed = (time.time() - start) * 1000
    return {
        "memory_context": memory_context,
        "past_interactions": past_interactions,
        "execution_trace": [{
            "agent": "MemoryRead", "action": "search_long_term",
            "input_summary": f"user={user_id}, query='{query[:60]}'",
            "output_summary": f"found={len(past_interactions)} interactions",
            "duration_ms": round(elapsed, 1),
        }],
    }


@traceable(run_type="chain", name="MemorySave")
def memory_save_node(state: AgentState, *, store: BaseStore, config: dict) -> dict:
    """Save the current interaction to long-term memory after response."""
    start = time.time()
    user_id = config.get("configurable", {}).get("user_id", "default")

    try:
        namespace = ("user_memory", user_id)
        key = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save a summary of this interaction
        store.put(
            namespace,
            key,
            {
                "query": state.get("original_query", ""),
                "rewritten_query": state.get("rewritten_query", ""),
                "answer_summary": state.get("answer", "")[:500],
                "query_type": state.get("query_type", ""),
                "entities": state.get("extracted_entities", []),
                "confidence": state.get("confidence_score", 0.0),
                "timestamp": datetime.now().isoformat(),
            },
        )
        logger.info("Memory saved — key=%s, user=%s", key, user_id)

    except Exception as e:
        logger.debug("Memory save skipped: %s", e)

    elapsed = (time.time() - start) * 1000
    return {
        "execution_trace": [{
            "agent": "MemorySave", "action": "store_interaction",
            "input_summary": f"user={user_id}",
            "output_summary": "saved",
            "duration_ms": round(elapsed, 1),
        }],
    }
