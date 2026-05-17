"""
Query Understanding Agent — classifies query type, scope, intent,
extracts entities, handles follow-ups, and rewrites for clarity.
Uses structured output (QueryAnalysis Pydantic model).
"""

import logging
import time
from langsmith import traceable

from graph.state import AgentState, QueryAnalysis
from utils.llm import get_structured_llm, invoke_llm

logger = logging.getLogger(__name__)

QUERY_UNDERSTANDING_PROMPT = """You are an expert query analyzer for an ESG/Sustainability knowledge system.
Analyze the user's query and classify it precisely.

CONVERSATION HISTORY (if any):
{conversation_history}

USER QUERY: {query}

Classify the query along these dimensions:

1. REWRITTEN QUERY: If the query is vague, a follow-up, or uses pronouns referring to previous context, rewrite it as a clear, self-contained question. If it's already clear, keep it as-is.

2. QUERY TYPE:
   - "text_query": Needs information from text passages (descriptions, explanations, policies)
   - "table_query": Needs numerical data, comparisons, or structured data from tables
   - "mixed": Needs both text context and table/numerical data

3. QUERY SCOPE:
   - "local": About a specific fact, page, section, or data point
   - "global": About themes, overviews, summaries across the whole document
   - "aggregation": Requires listing/counting ALL instances of something across the entire document
   - "direct": Simple definition or general knowledge that doesn't need document retrieval

4. INTENT: "factual", "comparative", "trend", or "definition"

5. ENTITIES: Key ESG/sustainability terms, metrics, or concepts mentioned

6. IS FOLLOW-UP: Whether this query references or builds on previous conversation

Analyze carefully:"""


@traceable(run_type="chain", name="QueryUnderstanding")
def query_understanding_node(state: AgentState) -> dict:
    """
    Analyze and classify the user query.
    Returns structured QueryAnalysis via Pydantic model.
    """
    start = time.time()
    query = state.get("original_query", "")

    if not query:
        # Extract from latest message
        messages = state.get("messages", [])
        if messages:
            query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # Build conversation history string for follow-up detection
    conversation_history = ""
    messages = state.get("messages", [])
    if len(messages) > 1:
        history_items = []
        for msg in messages[:-1]:  # Exclude current message
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            history_items.append(f"{role}: {content[:200]}")
        conversation_history = "\n".join(history_items[-6:])  # Last 3 exchanges

    # Use structured output for reliable parsing
    try:
        structured_llm = get_structured_llm(QueryAnalysis, task_type="light")
        prompt = QUERY_UNDERSTANDING_PROMPT.format(
            conversation_history=conversation_history or "(No previous conversation)",
            query=query,
        )
        analysis: QueryAnalysis = structured_llm.invoke(prompt)

    except Exception as e:
        logger.warning("Structured output failed, using fallback: %s", e)
        analysis = QueryAnalysis(
            rewritten_query=query,
            query_type="text_query",
            query_scope="local",
            intent="factual",
            entities=[],
            is_followup=False,
        )

    elapsed = (time.time() - start) * 1000

    logger.info(
        "Query understood — type=%s, scope=%s, intent=%s, followup=%s",
        analysis.query_type, analysis.query_scope, analysis.intent, analysis.is_followup,
    )

    return {
        "original_query": query,
        "rewritten_query": analysis.rewritten_query,
        "query_type": analysis.query_type,
        "query_scope": analysis.query_scope,
        "query_intent": analysis.intent,
        "extracted_entities": analysis.entities,
        "is_followup": analysis.is_followup,
        "execution_trace": [{
            "agent": "QueryUnderstanding",
            "action": "classify_query",
            "input_summary": query[:100],
            "output_summary": f"type={analysis.query_type}, scope={analysis.query_scope}",
            "duration_ms": round(elapsed, 1),
        }],
    }
