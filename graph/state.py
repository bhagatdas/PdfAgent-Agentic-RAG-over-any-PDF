"""
Shared AgentState and structured output models (Pydantic).
All agents operate on and return typed state updates.
"""

from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
import operator


# ══════════════════════════════════════════════════════════
# STRUCTURED OUTPUT MODELS (Pydantic)
# Every agent returns one of these typed models.
# ══════════════════════════════════════════════════════════

class QueryAnalysis(BaseModel):
    """Structured output from Query Understanding Agent."""
    rewritten_query: str = Field(description="Clarified, self-contained version of the query")
    query_type: Literal["text_query", "table_query", "mixed"] = Field(
        description="text_query=needs text retrieval, table_query=needs SQL, mixed=both"
    )
    query_scope: Literal["local", "global", "aggregation", "direct"] = Field(
        description="local=specific fact, global=theme/overview, aggregation=exhaustive list, direct=simple definition"
    )
    intent: str = Field(description="Query intent: factual, comparative, trend, or definition")
    entities: list[str] = Field(default_factory=list, description="Key terms extracted from query")
    is_followup: bool = Field(default=False, description="Whether this references previous conversation")


class ExecutionPlan(BaseModel):
    """Structured output from Planner Agent."""
    retrieval_strategy: Literal["standard", "raptor_global", "map_reduce", "none"] = Field(
        description="Which retrieval approach to use for text/RAG context"
    )
    use_table_agent: bool = Field(
        default=False,
        description="Whether to route to Table Agent for SQL on structured numerical data",
    )
    use_rag_agent: bool = Field(
        default=True,
        description="Whether to run the Retrieval (RAG) agent for text context",
    )
    early_exit: bool = Field(
        default=False,
        description="True for greetings, smalltalk, or non-ESG queries — skip the entire pipeline",
    )
    direct_response: str = Field(
        default="",
        description="Canned reply returned to the user when early_exit=True",
    )
    reasoning: str = Field(description="Why this plan was chosen")
    steps: list[str] = Field(default_factory=list, description="Ordered execution steps")


class SQLGeneration(BaseModel):
    """Structured output from Table Agent's SQL generation."""
    sql_query: str = Field(default="", description="Valid SQLite SELECT query")
    explanation: str = Field(default="", description="What this query does in plain English")
    tables_used: list[str] = Field(default_factory=list, description="Tables referenced")
    can_answer: bool = Field(default=True, description="False if no matching table exists")


class Citation(BaseModel):
    """A single source citation."""
    document: str = Field(description="Source document name")
    page: int = Field(description="Page number")
    chunk_id: str = Field(default="", description="Chunk identifier")
    relevance: float = Field(default=0.0, description="Relevance score")


class ReasonedAnswer(BaseModel):
    """Structured output from Reasoning Agent."""
    answer: str = Field(description="Comprehensive answer using ONLY provided context")
    citations: list[Citation] = Field(default_factory=list, description="Source references")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score")
    reasoning_summary: str = Field(default="", description="How the answer was derived")
    gaps: list[str] = Field(default_factory=list, description="Information not available in context")


class ValidationResult(BaseModel):
    """Structured output from Validation Agent (CRAG)."""
    is_grounded: bool = Field(default=False, description="Whether answer is supported by context")
    verdict: Literal["pass", "rewrite_answer", "re_retrieve", "give_up"] = Field(
        default="pass", description="Action to take"
    )
    issues: list[str] = Field(default_factory=list, description="Specific grounding issues found")
    suggestion: str = Field(default="", description="How to fix if not grounded")


# ══════════════════════════════════════════════════════════
# AGENT STATE (LangGraph TypedDict)
# This is the shared state that flows through the graph.
# ══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    Shared state for the multi-agent LangGraph workflow.
    Fields use annotations for automatic accumulation where needed.
    """

    # ── Conversation (auto-accumulated via add_messages) ──
    messages: Annotated[list, add_messages]

    # ── Query Understanding ──
    original_query: str
    rewritten_query: str
    query_type: str           # "text_query" | "table_query" | "mixed"
    query_scope: str          # "local" | "global" | "aggregation" | "direct"
    query_intent: str
    extracted_entities: list[str]
    is_followup: bool

    # ── Planning ──
    retrieval_strategy: str   # "standard" | "raptor_global" | "map_reduce" | "none"
    use_table_agent: bool
    use_rag_agent: bool
    should_end_early: bool    # Planner short-circuit for greetings/off-topic
    execution_plan_reasoning: str

    # ── Memory ──
    memory_context: str
    past_interactions: list[dict]

    # ── Retrieval ──
    retrieved_chunks: list[dict]
    retrieval_queries: list[str]
    raptor_level_used: int
    map_reduce_results: list[str]

    # ── Table / SQL ──
    schema_catalog: str
    generated_sql: str
    sql_explanation: str
    sql_results: list[dict]
    sql_retries: int

    # ── Reasoning ──
    answer: str
    citations: list[dict]
    confidence_score: float
    reasoning_summary: str
    information_gaps: list[str]

    # ── Validation / CRAG ──
    is_validated: bool
    validation_verdict: str
    validation_issues: list[str]
    validation_suggestion: str
    validation_retries: int

    # ── Observability ──
    execution_trace: Annotated[list[dict], operator.add]
    total_tokens_used: int
    total_duration_ms: float
    needs_confirmation: bool    # For map-reduce confirmation from user
    confirmation_message: str
