"""
LangGraph StateGraph workflow — wires all 7 agents with conditional routing.

Flow:
  Query → Understanding → Memory Read → Planner → [Retrieval|Table|Both] → Reasoning → Validation → Memory Save

CRAG Loop: Validation can route back to Retrieval or Reasoning on failure.
"""

import logging
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from graph.checkpointer import get_checkpointer, get_memory_store
from agents.query_understanding import query_understanding_node
from agents.planner import planner_node
from agents.retrieval import retrieval_node
from agents.table_agent import table_agent_node
from agents.reasoning import reasoning_node
from agents.validation import validation_node
from agents.memory_agent import memory_read_node, memory_save_node

logger = logging.getLogger(__name__)


def _plan_router(state: AgentState) -> str:
    """
    Route based on planner's decisions.

    Branches:
      - end_early           : greeting / off-topic — planner wrote the answer, skip to END
      - retrieval_only      : RAG only
      - table_only          : SQL only
      - retrieval_then_table: BOTH (richer mixed context)
      - direct_reasoning    : no retrieval needed but still synthesize (rare)
    """
    # Planner detected smalltalk / off-topic → bypass everything
    if state.get("should_end_early", False):
        return "end_early"

    strategy = state.get("retrieval_strategy", "standard")
    use_table = state.get("use_table_agent", False)
    # Backward-compatible: treat strategy=="none" as "no RAG" when use_rag_agent missing
    use_rag = state.get("use_rag_agent", strategy != "none")

    # Map-reduce confirmation hook (kept for future user-pause UI)
    if state.get("needs_confirmation", False) and strategy == "map_reduce":
        pass

    if use_rag and use_table:
        return "retrieval_then_table"
    if use_table and not use_rag:
        return "table_only"
    if use_rag and not use_table:
        return "retrieval_only"
    # Neither agent — synthesize directly (only when planner explicitly asks for it)
    return "direct_reasoning"


def _post_retrieval_router(state: AgentState) -> str:
    """After retrieval, check if table agent is also needed."""
    if state.get("use_table_agent", False):
        return "table_agent"
    return "reasoning"


def _validation_router(state: AgentState) -> str:
    """CRAG: route based on validation verdict."""
    verdict = state.get("validation_verdict", "pass")
    retries = state.get("validation_retries", 0)

    if verdict == "pass":
        return "memory_save"
    elif verdict == "rewrite_answer" and retries < 2:
        return "retry_reasoning"
    elif verdict == "re_retrieve" and retries < 2:
        return "retry_retrieval"
    else:
        # Max retries exceeded or give_up — proceed with what we have
        return "memory_save"


def build_workflow():
    """Build and compile the LangGraph workflow."""
    builder = StateGraph(AgentState)

    # ── Add all nodes ──
    builder.add_node("query_understanding", query_understanding_node)
    builder.add_node("memory_read", memory_read_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("table_agent", table_agent_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("validation", validation_node)
    builder.add_node("memory_save", memory_save_node)

    # ── Fixed edges: start → understanding → memory → planner ──
    builder.add_edge(START, "query_understanding")
    builder.add_edge("query_understanding", "memory_read")
    builder.add_edge("memory_read", "planner")

    # ── Conditional: planner routes to retrieval strategy ──
    # end_early skips the rest of the pipeline (greetings / off-topic).
    builder.add_conditional_edges("planner", _plan_router, {
        "end_early": END,
        "retrieval_only": "retrieval",
        "table_only": "table_agent",
        "retrieval_then_table": "retrieval",
        "direct_reasoning": "reasoning",
    })

    # ── After retrieval: maybe also run table agent ──
    builder.add_conditional_edges("retrieval", _post_retrieval_router, {
        "table_agent": "table_agent",
        "reasoning": "reasoning",
    })

    # ── Table agent always goes to reasoning ──
    builder.add_edge("table_agent", "reasoning")

    # ── Reasoning goes to validation ──
    builder.add_edge("reasoning", "validation")

    # ── CRAG loop: validation routes back or forward ──
    builder.add_conditional_edges("validation", _validation_router, {
        "memory_save": "memory_save",
        "retry_reasoning": "reasoning",
        "retry_retrieval": "retrieval",
    })

    # ── Memory save ends the flow ──
    builder.add_edge("memory_save", END)

    # ── Compile with persistence ──
    checkpointer = get_checkpointer()
    store = get_memory_store()

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    logger.info("LangGraph workflow compiled — nodes=%d", len(builder.nodes))
    return graph


# Build the workflow singleton
workflow = build_workflow()


def invoke_query(
    query: str,
    thread_id: str = "default",
    user_id: str = "default",
) -> dict:
    """
    Invoke the full agent pipeline for a user query.

    Args:
        query: User's question
        thread_id: Session/conversation ID (for state continuity)
        user_id: User ID (for long-term memory namespacing)

    Returns:
        Final state dict with answer, citations, confidence, trace, etc.
    """
    from langchain_core.messages import HumanMessage

    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id,
        }
    }

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "original_query": query,
        "validation_retries": 0,
        "sql_retries": 0,
        "needs_confirmation": False,
    }

    logger.info("=" * 50)
    logger.info("QUERY: %s", query)
    logger.info("Thread: %s | User: %s", thread_id, user_id)
    logger.info("=" * 50)

    result = workflow.invoke(initial_state, config=config)

    logger.info("ANSWER: %s", result.get("answer", "")[:200])
    logger.info("Confidence: %.2f", result.get("confidence_score", 0.0))
    logger.info("=" * 50)

    return result
