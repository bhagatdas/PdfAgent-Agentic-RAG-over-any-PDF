"""
Reasoning Agent — synthesizes information from retrieval and table agents.
Generates a comprehensive answer with citations using structured output.
"""

import logging
import time
import json
from langsmith import traceable

from graph.state import AgentState, ReasonedAnswer, Citation
from utils.llm import get_structured_llm

logger = logging.getLogger(__name__)

REASONING_PROMPT = """You are an expert ESG/Sustainability analyst working for Bhagat Labs' ESG Insight Pro system.
You answer questions EXCLUSIVELY from the provided ESG report context below.

CRITICAL RULES (MUST FOLLOW):
- You may ONLY use information from the RETRIEVED CONTEXT, SQL RESULTS, and MAP-REDUCE RESULTS below.
- You must NEVER use your own training knowledge, general facts, or external information.
- If the query is NOT related to ESG, sustainability, or the ingested reports, respond EXACTLY:
  "I can only answer questions about the ESG reports that have been uploaded to this system. Your question appears to be outside the scope of the available documents."
- If the query IS ESG-related but the answer is NOT in the provided context, respond EXACTLY:
  "This specific information is not available in the uploaded ESG reports. The documents do not contain data on [topic]."
- NEVER fabricate data, statistics, or claims. Every factual statement must come from the context.
- Cite EVERY claim with [Document, Page X] format.

QUERY: {query}

RETRIEVED TEXT CONTEXT:
{text_context}

SQL/TABLE RESULTS:
{table_context}

MAP-REDUCE AGGREGATION RESULTS:
{map_reduce_context}

MEMORY CONTEXT (previous interactions):
{memory_context}

VALIDATION FEEDBACK (if retrying):
{validation_feedback}

INSTRUCTIONS:
1. Answer using ONLY the context above. Zero tolerance for outside knowledge.
2. Cite every claim: [DocumentName, Page X]
3. For table/numerical data, reference the specific table, column, and values.
4. Confidence score: 1.0 = every claim is directly supported, 0.5 = partially, 0.0 = no support found.
5. List information gaps: what specific data is missing from the reports.
6. If context is empty or irrelevant, say so — do NOT guess.

Generate your answer:"""


@traceable(run_type="chain", name="Reasoning")
def reasoning_node(state: AgentState) -> dict:
    """Synthesize a final answer from all retrieved context with citations."""
    start = time.time()
    query = state.get("rewritten_query", state.get("original_query", ""))

    # Build text context from retrieved chunks
    text_context = _format_chunks(state.get("retrieved_chunks", []))

    # Build table context from SQL results
    table_context = _format_sql_results(state)

    # Build map-reduce context
    mr_results = state.get("map_reduce_results", [])
    map_reduce_context = "\n".join(f"- {item}" for item in mr_results) if mr_results else "(none)"

    # Memory context
    memory_context = state.get("memory_context", "(none)")

    # Validation feedback (for retry loops)
    validation_feedback = ""
    if state.get("validation_retries", 0) > 0:
        validation_feedback = (
            f"PREVIOUS ANSWER WAS REJECTED. Issues: {state.get('validation_issues', [])}\n"
            f"Suggestion: {state.get('validation_suggestion', '')}\n"
            "Please fix these issues in your new answer."
        )

    try:
        structured_llm = get_structured_llm(ReasonedAnswer, task_type="heavy")
        prompt = REASONING_PROMPT.format(
            query=query,
            text_context=text_context[:6000],
            table_context=table_context[:2000],
            map_reduce_context=map_reduce_context[:2000],
            memory_context=memory_context[:500],
            validation_feedback=validation_feedback,
        )
        answer: ReasonedAnswer = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("Structured reasoning failed, using raw: %s", e)
        from utils.llm import invoke_llm
        raw = invoke_llm(
            REASONING_PROMPT.format(
                query=query, text_context=text_context[:4000],
                table_context=table_context[:1500], map_reduce_context=map_reduce_context[:1000],
                memory_context=memory_context[:300], validation_feedback=validation_feedback,
            ),
            task_type="heavy",
        )
        answer = ReasonedAnswer(
            answer=raw, citations=[], confidence=0.5,
            reasoning_summary="Raw LLM response (structured output failed)", gaps=[],
        )

    elapsed = (time.time() - start) * 1000
    logger.info("Reasoning complete — confidence=%.2f, citations=%d", answer.confidence, len(answer.citations))

    return {
        "answer": answer.answer,
        "citations": [c.model_dump() for c in answer.citations],
        "confidence_score": answer.confidence,
        "reasoning_summary": answer.reasoning_summary,
        "information_gaps": answer.gaps,
        "execution_trace": [{
            "agent": "Reasoning", "action": "synthesize_answer",
            "input_summary": f"chunks={len(state.get('retrieved_chunks', []))}, sql_rows={len(state.get('sql_results', []))}",
            "output_summary": f"confidence={answer.confidence:.2f}, citations={len(answer.citations)}",
            "duration_ms": round(elapsed, 1),
        }],
    }


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context string."""
    if not chunks:
        return "(No text context retrieved)"
    lines = []
    for i, chunk in enumerate(chunks):
        meta = chunk.get("metadata", {})
        doc = meta.get("document_name", "Unknown")
        page = meta.get("page_number", "?")
        score = chunk.get("rerank_score", chunk.get("rrf_score", chunk.get("score", "")))
        lines.append(f"[Source: {doc}, Page {page}] (score: {score})")
        lines.append(chunk.get("content", "")[:800])
        lines.append("")
    return "\n".join(lines)


def _format_sql_results(state: AgentState) -> str:
    """Format SQL results into a readable context string."""
    sql_results = state.get("sql_results", [])
    if not sql_results:
        return "(No table data)"
    sql = state.get("generated_sql", "")
    explanation = state.get("sql_explanation", "")
    rows_str = json.dumps(sql_results[:20], indent=2, default=str)
    return f"SQL Query: {sql}\nExplanation: {explanation}\nResults:\n{rows_str}"
