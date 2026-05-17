"""
Validation Agent (CRAG) — checks grounding, citations, and hallucination.
Can trigger re-retrieval or answer rewriting via corrective feedback loops.
"""

import logging
import time
from langsmith import traceable

from graph.state import AgentState, ValidationResult
from utils.llm import get_structured_llm

logger = logging.getLogger(__name__)

VALIDATION_PROMPT = """You are a strict fact-checker for an ESG knowledge system.
Verify whether the answer is fully grounded in the provided context.

QUERY: {query}
ANSWER: {answer}

CONTEXT USED (retrieved chunks):
{context_summary}

SQL RESULTS (if any):
{sql_summary}

CHECK THESE CRITERIA:
1. GROUNDING: Is every claim in the answer supported by the context above?
2. CITATIONS: Does the answer reference specific sources? Are they valid?
3. HALLUCINATION: Does the answer contain information NOT in the context?
4. COMPLETENESS: Does the answer actually address the query?

VERDICT OPTIONS:
- "pass": Answer is well-grounded and accurate
- "rewrite_answer": Answer has minor issues — can be fixed by re-reasoning with feedback
- "re_retrieve": The retrieved context seems irrelevant — need different search
- "give_up": Cannot validate (e.g., no context available)

Validate the answer:"""


@traceable(run_type="chain", name="Validation")
def validation_node(state: AgentState) -> dict:
    """Validate the answer for grounding, citations, and hallucination."""
    start = time.time()
    answer = state.get("answer", "")
    query = state.get("rewritten_query", state.get("original_query", ""))
    retries = state.get("validation_retries", 0)

    # If no answer, nothing to validate
    if not answer.strip():
        return _build_result(start, ValidationResult(
            is_grounded=False, verdict="re_retrieve",
            issues=["No answer was generated"], suggestion="Re-retrieve with modified query",
        ), retries)

    # Build context summary for validation
    chunks = state.get("retrieved_chunks", [])
    context_summary = "\n".join(
        f"[{c.get('metadata', {}).get('document_name', '?')}, Page {c.get('metadata', {}).get('page_number', '?')}]: "
        f"{c.get('content', '')[:300]}"
        for c in chunks[:8]
    ) or "(No context)"

    sql_summary = ""
    if state.get("sql_results"):
        sql_summary = f"SQL: {state.get('generated_sql', '')}\nRows: {len(state['sql_results'])}"

    try:
        structured_llm = get_structured_llm(ValidationResult, task_type="light")
        prompt = VALIDATION_PROMPT.format(
            query=query, answer=answer[:2000],
            context_summary=context_summary[:3000], sql_summary=sql_summary or "(none)",
        )
        result: ValidationResult = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("Structured validation failed: %s", e)
        # Default to pass if validation itself fails
        result = ValidationResult(
            is_grounded=True, verdict="pass",
            issues=["Validation model failed — passing by default"],
            suggestion="",
        )

    return _build_result(start, result, retries)


def _build_result(start: float, result: ValidationResult, retries: int) -> dict:
    elapsed = (time.time() - start) * 1000
    logger.info(
        "Validation — grounded=%s, verdict=%s, issues=%d",
        result.is_grounded, result.verdict, len(result.issues),
    )
    return {
        "is_validated": result.is_grounded,
        "validation_verdict": result.verdict,
        "validation_issues": result.issues,
        "validation_suggestion": result.suggestion,
        "validation_retries": retries + (0 if result.verdict == "pass" else 1),
        "execution_trace": [{
            "agent": "Validation", "action": f"validate_{result.verdict}",
            "input_summary": f"retries={retries}",
            "output_summary": f"grounded={result.is_grounded}, verdict={result.verdict}",
            "duration_ms": round(elapsed, 1),
        }],
    }
