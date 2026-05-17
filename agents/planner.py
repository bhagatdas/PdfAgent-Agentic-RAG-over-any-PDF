"""
Planner Agent — routes queries to the correct retrieval strategy and agents.

Decisions the planner makes:
1. Is this query in-scope (about ESG/sustainability docs)?
   - If it's a greeting / smalltalk / clearly off-topic  → early_exit=True,
     write a polite direct_response, and the workflow ends immediately.
2. What context does the answer need?
   - RAG only  (text passages)            → use_rag_agent=True,  use_table_agent=False
   - Table only (pure numerical/SQL)      → use_rag_agent=False, use_table_agent=True
   - Both      (richer mixed context)     → use_rag_agent=True,  use_table_agent=True
3. Which retrieval strategy for RAG?
   - "standard"     — local, specific facts
   - "raptor_global"— global themes / overviews
   - "map_reduce"   — exhaustive aggregation (expensive)
   - "none"         — no RAG needed (table_only or early_exit)

Uses structured output (ExecutionPlan Pydantic model).
"""

import logging
import time
from langsmith import traceable

from graph.state import AgentState, ExecutionPlan
from utils.llm import get_structured_llm

logger = logging.getLogger(__name__)

# Lightweight rule-based smalltalk/greeting detector used by the fallback.
_GREETING_TOKENS = {
    "hi", "hello", "hey", "yo", "hiya", "howdy",
    "thanks", "thank", "thx", "ty",
    "bye", "goodbye", "cya",
    "ok", "okay", "cool", "nice", "great", "good",
    "test", "ping",
}

OFF_TOPIC_RESPONSE = (
    "I'm ESG Insight Pro — I only answer questions about the ESG / sustainability "
    "reports that have been uploaded to this system. Try asking about emissions, "
    "ESG initiatives, governance, or any specific data in the reports."
)

GREETING_RESPONSE = (
    "Hello! I'm ESG Insight Pro, an analyst for your uploaded ESG / sustainability "
    "reports. Ask me about emissions data, initiatives, governance metrics, or "
    "anything else from the documents."
)

PLANNER_PROMPT = """You are the execution planner for ESG Insight Pro, an ESG/Sustainability
knowledge system. You decide HOW (or whether) to answer the user's query.

QUERY: {query}
QUERY TYPE: {query_type}      (text_query | table_query | mixed)
QUERY SCOPE: {query_scope}    (local | global | aggregation | direct)
INTENT: {intent}
ENTITIES: {entities}
MEMORY CONTEXT: {memory_context}

═══════════════════════════════════════════════════════════════════
STEP 1 — Is this query in-scope?
═══════════════════════════════════════════════════════════════════
Set early_exit=true and write a polite direct_response IF the query is:
- A greeting or smalltalk ("hi", "hello", "thanks", "how are you", "good morning")
- A meta/system question not about the documents ("what can you do", "who are you")
- Clearly OFF-TOPIC (weather, sports, coding help, general knowledge unrelated to ESG)
- Empty, gibberish, or under 3 meaningful words with no ESG keyword

When early_exit=true:
- Set retrieval_strategy="none", use_rag_agent=false, use_table_agent=false
- direct_response should briefly greet/redirect the user toward ESG questions

Otherwise (the query IS about ESG / the uploaded reports), set early_exit=false
and continue to Step 2.

═══════════════════════════════════════════════════════════════════
STEP 2 — What context does the answer need?
═══════════════════════════════════════════════════════════════════
Choose ONE of these three combinations:

(A) RAG only        → use_rag_agent=true,  use_table_agent=false
    For text-based questions: descriptions, policies, narratives, themes.
    Example: "What are the company's climate commitments?"

(B) Table only      → use_rag_agent=false, use_table_agent=true
    For pure numerical lookups answerable from structured tables alone.
    Example: "What was Scope 1 emissions in 2023?" (exact value from a table)

(C) Both (RAG + Table)  → use_rag_agent=true, use_table_agent=true
    PREFER this when the answer benefits from BOTH narrative context AND numbers,
    e.g., comparisons, trends-with-explanation, or when the user asks "why" alongside a number.
    Example: "How did emissions change YoY and what drove the change?"

═══════════════════════════════════════════════════════════════════
STEP 3 — Pick the RAG retrieval_strategy (only if use_rag_agent=true)
═══════════════════════════════════════════════════════════════════
- "standard"       → local/specific facts (most queries)
- "raptor_global"  → broad themes, overviews, "what is this report about"
- "map_reduce"     → exhaustive enumeration ("list ALL...", "every initiative")  [EXPENSIVE]
- "none"           → use_rag_agent=false (table_only or early_exit)

Mapping hints from query_scope:
  local       → "standard"
  global      → "raptor_global"
  aggregation → "map_reduce"
  direct      → "none" (and likely early_exit if truly off-topic)

Return the full ExecutionPlan. Be decisive.
"""


@traceable(run_type="chain", name="Planner")
def planner_node(state: AgentState) -> dict:
    """
    Create an execution plan based on query classification.
    Decides: early_exit, RAG/Table/Both, and which retrieval strategy.
    """
    start = time.time()

    query = state.get("rewritten_query", state.get("original_query", ""))
    query_type = state.get("query_type", "text_query")
    query_scope = state.get("query_scope", "local")
    entities = state.get("extracted_entities", [])

    try:
        structured_llm = get_structured_llm(ExecutionPlan, task_type="light")
        prompt = PLANNER_PROMPT.format(
            query=query,
            query_type=query_type,
            query_scope=query_scope,
            intent=state.get("query_intent", "factual"),
            entities=", ".join(entities),
            memory_context=state.get("memory_context", "(none)")[:300],
        )
        plan: ExecutionPlan = structured_llm.invoke(prompt)

    except Exception as e:
        logger.warning("Structured planner failed, using rule-based fallback: %s", e)
        plan = _rule_based_plan(query, query_type, query_scope, entities)

    # Belt-and-braces: catch obvious greetings even if the LLM missed them.
    if not plan.early_exit and _looks_like_smalltalk(query, entities):
        logger.info("Smalltalk detected by rule-based check — overriding plan to early_exit")
        plan = _smalltalk_plan(query)

    # Map-reduce confirmation flag (unchanged behavior)
    needs_confirmation = plan.retrieval_strategy == "map_reduce" and not plan.early_exit
    confirmation_msg = ""
    if needs_confirmation:
        confirmation_msg = (
            "This query requires scanning ALL document chunks for exhaustive results. "
            "This may take several minutes and use significant compute. Proceed?"
        )

    elapsed = (time.time() - start) * 1000

    # ── Early-exit short-circuit: write the canned answer and signal end ──
    if plan.early_exit:
        logger.info("Planner: early_exit (smalltalk/off-topic) — skipping rest of pipeline")
        response = plan.direct_response or GREETING_RESPONSE
        return {
            "retrieval_strategy": "none",
            "use_rag_agent": False,
            "use_table_agent": False,
            "should_end_early": True,
            "execution_plan_reasoning": plan.reasoning or "Early exit: greeting / off-topic",
            "answer": response,
            "citations": [],
            "confidence_score": 1.0,
            "reasoning_summary": "Direct response (no retrieval needed)",
            "information_gaps": [],
            "is_validated": True,
            "validation_verdict": "pass",
            "validation_issues": [],
            "validation_suggestion": "",
            "needs_confirmation": False,
            "confirmation_message": "",
            "execution_trace": [{
                "agent": "Planner",
                "action": "early_exit_smalltalk",
                "input_summary": f"query='{query[:80]}'",
                "output_summary": "skipped pipeline — direct response",
                "duration_ms": round(elapsed, 1),
            }],
        }

    # ── Normal plan: route to RAG / Table / Both ──
    logger.info(
        "Plan — strategy=%s, rag=%s, table=%s, steps=%s",
        plan.retrieval_strategy, plan.use_rag_agent, plan.use_table_agent, plan.steps,
    )

    return {
        "retrieval_strategy": plan.retrieval_strategy,
        "use_rag_agent": plan.use_rag_agent,
        "use_table_agent": plan.use_table_agent,
        "should_end_early": False,
        "execution_plan_reasoning": plan.reasoning,
        "needs_confirmation": needs_confirmation,
        "confirmation_message": confirmation_msg,
        "execution_trace": [{
            "agent": "Planner",
            "action": "create_plan",
            "input_summary": f"type={query_type}, scope={query_scope}",
            "output_summary": (
                f"strategy={plan.retrieval_strategy}, rag={plan.use_rag_agent}, "
                f"table={plan.use_table_agent}"
            ),
            "duration_ms": round(elapsed, 1),
        }],
    }


def _looks_like_smalltalk(query: str, entities: list[str]) -> bool:
    """Conservative rule-based detector for greetings / trivial inputs."""
    q = (query or "").strip().lower()
    if not q:
        return True
    # Strip basic punctuation
    stripped = "".join(ch for ch in q if ch.isalnum() or ch.isspace()).strip()
    tokens = stripped.split()

    if not tokens:
        return True
    # Very short input with no extracted ESG entities
    if len(tokens) <= 3 and not entities:
        if any(t in _GREETING_TOKENS for t in tokens):
            return True
        # Single-word inputs with no entities are almost never real queries
        if len(tokens) == 1:
            return True
    return False


def _smalltalk_plan(query: str) -> ExecutionPlan:
    """Build a ready-made plan for greetings / off-topic input."""
    q = (query or "").strip().lower()
    is_greeting = any(t in _GREETING_TOKENS for t in q.split())
    response = GREETING_RESPONSE if is_greeting else OFF_TOPIC_RESPONSE
    return ExecutionPlan(
        retrieval_strategy="none",
        use_rag_agent=False,
        use_table_agent=False,
        early_exit=True,
        direct_response=response,
        reasoning="Detected greeting or off-topic input — skipping retrieval/reasoning.",
        steps=["early_exit"],
    )


def _rule_based_plan(
    query: str,
    query_type: str,
    query_scope: str,
    entities: list[str],
) -> ExecutionPlan:
    """Fallback rule-based planning when LLM structured output fails."""
    # First check for smalltalk
    if _looks_like_smalltalk(query, entities):
        return _smalltalk_plan(query)

    strategy_map = {
        "local": "standard",
        "global": "raptor_global",
        "aggregation": "map_reduce",
        "direct": "none",
    }
    strategy = strategy_map.get(query_scope, "standard")
    use_table = query_type in ("table_query", "mixed")
    use_rag = query_type in ("text_query", "mixed") and strategy != "none"

    # Safety: at least one agent must run
    if not use_rag and not use_table:
        use_rag = True
        strategy = "standard"

    return ExecutionPlan(
        retrieval_strategy=strategy,
        use_rag_agent=use_rag,
        use_table_agent=use_table,
        early_exit=False,
        direct_response="",
        reasoning=f"Rule-based fallback: scope={query_scope}, type={query_type}",
        steps=["retrieval" if use_rag else "", "table_agent" if use_table else "", "reasoning", "validation"],
    )
