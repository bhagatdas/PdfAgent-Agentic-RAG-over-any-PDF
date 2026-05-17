"""
Table Agent — schema-aware SQL generation and execution.
Generates SQLite SELECT queries from natural language using the schema catalog.
"""

import logging
import time
from langsmith import traceable

from graph.state import AgentState, SQLGeneration
from storage.sql_store import sql_store
from storage.schema_manager import schema_manager
from utils.llm import get_structured_llm

logger = logging.getLogger(__name__)

SQL_GENERATION_PROMPT = """You are a SQL expert. Given the database schema below, generate a SQLite SELECT query.

SCHEMA:
{schema_catalog}

RULES:
- Use ONLY existing tables/columns from the schema
- Only SELECT queries allowed
- Use aggregations (SUM, AVG, COUNT) when needed
- If question cannot be answered, set can_answer=false

QUESTION: {query}
{error_context}

Generate the SQL:"""


@traceable(run_type="chain", name="TableAgent")
def table_agent_node(state: AgentState) -> dict:
    """Generate and execute SQL against the table store with retry logic."""
    start = time.time()
    query = state.get("rewritten_query", state.get("original_query", ""))
    sql_retries = state.get("sql_retries", 0)
    catalog = schema_manager.generate_catalog()

    if "NO TABLES AVAILABLE" in catalog:
        return _trace_result(start, "no_tables", query, "No tables available", {
            "sql_results": [], "generated_sql": "", "sql_explanation": "No tables ingested.",
            "schema_catalog": catalog,
        })

    error_context = ""
    if sql_retries > 0:
        error_context = f"PREVIOUS SQL FAILED. Fix it using correct names from schema."

    try:
        structured_llm = get_structured_llm(SQLGeneration, task_type="light")
        prompt = SQL_GENERATION_PROMPT.format(
            schema_catalog=catalog, query=query, error_context=error_context,
        )
        sql_gen: SQLGeneration = structured_llm.invoke(prompt)
    except Exception as e:
        logger.warning("SQL generation failed: %s", e)
        sql_gen = SQLGeneration(sql_query="", explanation="Failed", tables_used=[], can_answer=False)

    if not sql_gen.can_answer or not sql_gen.sql_query.strip():
        return _trace_result(start, "no_match", query, sql_gen.explanation, {
            "sql_results": [], "generated_sql": "", "sql_explanation": sql_gen.explanation,
            "schema_catalog": catalog,
        })

    logger.info("Executing SQL: %s", sql_gen.sql_query)
    success, result = sql_store.safe_execute(sql_gen.sql_query)

    if success:
        sql_results = result.to_dict(orient="records")
        logger.info("SQL success — rows=%d", len(sql_results))
        return _trace_result(start, "sql_success", sql_gen.sql_query[:100], f"rows={len(sql_results)}", {
            "sql_results": sql_results, "generated_sql": sql_gen.sql_query,
            "sql_explanation": sql_gen.explanation, "schema_catalog": catalog, "sql_retries": 0,
        })
    else:
        logger.warning("SQL error: %s", result)
        return _trace_result(start, "sql_error", sql_gen.sql_query[:100], str(result)[:100], {
            "sql_results": [], "generated_sql": sql_gen.sql_query,
            "sql_explanation": f"SQL Error: {result}", "schema_catalog": catalog,
            "sql_retries": sql_retries + 1,
        })


def _trace_result(start: float, action: str, inp: str, out: str, updates: dict) -> dict:
    elapsed = (time.time() - start) * 1000
    updates["execution_trace"] = [{
        "agent": "TableAgent", "action": action,
        "input_summary": inp[:100], "output_summary": out[:100],
        "duration_ms": round(elapsed, 1),
    }]
    return updates
