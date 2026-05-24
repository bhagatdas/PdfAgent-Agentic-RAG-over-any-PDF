"""
Generate a PNG visualization of the LangGraph multi-agent workflow.

Rebuilds the StateGraph topology directly (without the SQLite checkpointer
or agent imports) so we can render the diagram regardless of which optional
deps are installed.
"""

import sys
from pathlib import Path
from typing import TypedDict

OUT_PNG = Path(__file__).parent / "workflow.png"
OUT_MMD = Path(__file__).parent / "workflow.mmd"


def _build_topology_only():
    """Reconstruct just the graph structure — no agent imports, no compile."""
    from langgraph.graph import StateGraph, START, END

    class _State(TypedDict, total=False):
        retrieval_strategy: str
        use_rag_agent: bool
        use_table_agent: bool
        should_end_early: bool
        validation_verdict: str
        validation_retries: int

    def _noop(state):  # placeholder for every node
        return {}

    def _plan_router(state):
        if state.get("should_end_early", False):
            return "end_early"
        strategy = state.get("retrieval_strategy", "standard")
        use_table = state.get("use_table_agent", False)
        use_rag = state.get("use_rag_agent", strategy != "none")
        if use_rag and use_table:
            return "retrieval_then_table"
        if use_table and not use_rag:
            return "table_only"
        if use_rag and not use_table:
            return "retrieval_only"
        return "direct_reasoning"

    def _post_retrieval_router(state):
        # Both branches converge on entity_metric_extractor before reasoning —
        # the pre-synthesis fact-extraction node added for grounded numerical
        # attribution. Keep in sync with graph/workflow.py.
        return "table_agent" if state.get("use_table_agent", False) else "entity_metric_extractor"

    def _validation_router(state):
        verdict = state.get("validation_verdict", "pass")
        retries = state.get("validation_retries", 0)
        if verdict == "pass":
            return "memory_save"
        if verdict == "rewrite_answer" and retries < 2:
            return "retry_reasoning"
        if verdict == "re_retrieve" and retries < 2:
            return "retry_retrieval"
        return "memory_save"

    b = StateGraph(_State)
    for n in [
        "query_understanding", "memory_read", "planner",
        "retrieval", "table_agent",
        "entity_metric_extractor",
        "reasoning", "validation", "memory_save",
    ]:
        b.add_node(n, _noop)

    b.add_edge(START, "query_understanding")
    b.add_edge("query_understanding", "memory_read")
    b.add_edge("memory_read", "planner")
    b.add_conditional_edges("planner", _plan_router, {
        "end_early": END,
        "retrieval_only": "retrieval",
        "table_only": "table_agent",
        "retrieval_then_table": "retrieval",
        "direct_reasoning": "reasoning",
    })
    b.add_conditional_edges("retrieval", _post_retrieval_router, {
        "table_agent": "table_agent",
        "entity_metric_extractor": "entity_metric_extractor",
    })
    # Table agent → extractor → reasoning. The extractor runs before reasoning
    # so the prompt has a pre-extracted (entity, metric, value) facts table.
    b.add_edge("table_agent", "entity_metric_extractor")
    b.add_edge("entity_metric_extractor", "reasoning")
    b.add_edge("reasoning", "validation")
    b.add_conditional_edges("validation", _validation_router, {
        "memory_save": "memory_save",
        # CRAG retry skips the extractor — facts persist in state from the
        # first pass, so re-extracting would waste an LLM call.
        "retry_reasoning": "reasoning",
        "retry_retrieval": "retrieval",
    })
    b.add_edge("memory_save", END)

    return b.compile()


def main() -> int:
    print("Building workflow topology...")
    graph_obj = _build_topology_only().get_graph()

    # ── Always write Mermaid source (works offline) ──
    try:
        mermaid_src = graph_obj.draw_mermaid()
        OUT_MMD.write_text(mermaid_src, encoding="utf-8")
        print(f"Mermaid source written: {OUT_MMD}")
    except Exception as e:
        print(f"Warning: could not generate Mermaid source: {e}")

    # ── Try mermaid.ink PNG renderer (needs internet) ──
    try:
        png_bytes = graph_obj.draw_mermaid_png()
        OUT_PNG.write_bytes(png_bytes)
        print(f"PNG written: {OUT_PNG}  ({len(png_bytes):,} bytes)")
        return 0
    except Exception as e:
        print(f"draw_mermaid_png() (mermaid.ink) failed: {e}")

    # ── Pyppeteer fallback ──
    try:
        from langchain_core.runnables.graph import MermaidDrawMethod
        png_bytes = graph_obj.draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)
        OUT_PNG.write_bytes(png_bytes)
        print(f"PNG written via Pyppeteer: {OUT_PNG}")
        return 0
    except Exception as e:
        print(f"Pyppeteer fallback failed: {e}")

    print()
    print("PNG rendering failed. Render the Mermaid file locally with:")
    print("  npm install -g @mermaid-js/mermaid-cli")
    print(f"  mmdc -i {OUT_MMD.name} -o {OUT_PNG.name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
