# ESG Insight Pro by Bhagat Labs — Complete System Specification

> **Purpose**: This document captures the full end-to-end design of the ESG Insight Pro system.
> Use this as a prompt, reference, or onboarding guide for understanding or reproducing the system.

---

## What This System Does

An AI-powered **Subject Matter Expert** for ESG (Environmental, Social, Governance) that answers complex sustainability queries from large enterprise PDF reports (50+ pages, multimodal: text, tables, images). It is NOT a simple chatbot — it is a **stateful, multi-agent, goal-driven reasoning system**.

---

## Architecture Overview

```
User Query
  |
  v
[Query Understanding Agent] -- classifies type, scope, intent, extracts entities
  |
  v
[Memory Agent (Read)] -- retrieves relevant past interactions from LangGraph Store
  |
  v
[Planner Agent] -- decides retrieval strategy + which agents to invoke
  |
  +---> [Retrieval Agent] -- hybrid search (semantic + BM25) + reranking
  |       |                    OR RAPTOR tree search (for global queries)
  |       |                    OR Map-Reduce scan (for aggregation queries)
  |       v
  +---> [Table Agent] -- generates SQL from natural language, executes against SQLite
  |       |
  |       v
  +-------> [Reasoning Agent] -- synthesizes answer from ALL context with citations
              |
              v
            [Validation Agent (CRAG)] -- checks grounding, detects hallucination
              |
              +--- PASS --> [Memory Agent (Save)] --> Response to User
              +--- FAIL --> loops back to Retrieval or Reasoning (max 2 retries)
```

---

## 7 Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| **Query Understanding** | Light (deepseek-r1:1.5b) | Classify query type/scope/intent, rewrite for clarity, detect follow-ups |
| **Memory** | None (uses LangGraph Store) | Read/write long-term cross-session memory |
| **Planner** | Light | Route to correct retrieval strategy and agents |
| **Retrieval** | Light (for query rewriting) | Multi-query rewriting → hybrid search → reranking → context expansion |
| **Table** | Light | Generate SQLite SELECT queries from natural language using schema catalog |
| **Reasoning** | Heavy (deepseek-r1:7b) | Synthesize comprehensive answer with citations and confidence score |
| **Validation** | Light | CRAG: check grounding, validate citations, detect hallucination |

---

## Key Techniques

### Advanced RAG
- **Parent-child chunking**: Small children for precise search, large parents for context
- **Hybrid retrieval**: Semantic (ChromaDB) + keyword (BM25) merged via Reciprocal Rank Fusion
- **Cross-encoder reranking**: Precision filter using `ms-marco-MiniLM-L-6-v2`
- **Multi-query rewriting**: Generate 3 query variants for better recall
- **Context compression**: Only top-K reranked chunks sent to LLM

### Solving the Harry Potter Problem (Global vs Local Queries)
- **RAPTOR Tree**: Hierarchical summaries (chunk clusters → section summaries → document overview)
  - Local queries → search Level 0 (raw chunks)
  - Global queries → search Level 2-3 (section/document summaries)
- **Map-Reduce**: For exhaustive aggregation queries ("list ALL X"), scan every chunk with focused extraction prompt, then merge/deduplicate
- **Adaptive routing**: Query Understanding classifies scope → Planner picks strategy

### Contextual Retrieval (Anthropic)
- Each chunk gets a 2-3 sentence context prefix during preprocessing
- Reduces retrieval failures by ~49%

### Corrective RAG (CRAG)
- Validation agent can: pass, rewrite answer, or re-retrieve with modified query
- Max 2 retry loops before giving up with low confidence

### Structured Output
- Every agent returns a Pydantic model (QueryAnalysis, ExecutionPlan, SQLGeneration, ReasonedAnswer, ValidationResult)
- Type-safe, parseable, validated at every step

---

## Preprocessing Pipeline (Offline)

```
PDF → PyMuPDF Parse
  ├── Text per page → Parent-child chunking → Contextual enrichment → ChromaDB + BM25
  ├── Tables per page → DataFrame → SQLite + Schema catalog + Text representation → ChromaDB
  ├── Images per page → EasyOCR + LLava captioning → Text chunks → ChromaDB
  └── All chunks → RAPTOR tree builder → Hierarchical summaries → ChromaDB
```

### Table Storage (SQL)
- All PDF tables stored in SQLite with auto-generated schema catalog
- Schema includes: table names, columns, types, descriptions, sample data
- Table Agent generates SQL queries aware of the full schema (Databricks-style)
- Read-only enforcement at query time (SELECT only)

---

## State & Memory

### Short-term (per session)
- **LangGraph SqliteSaver checkpointer** — auto-persists full state per `thread_id`
- Conversation history via `add_messages` reducer — auto-accumulated
- Follow-up detection uses message history

### Long-term (cross session)
- **LangGraph InMemoryStore** — namespaced by `user_id`
- Stores past Q&A summaries with topics and entities
- Semantic search for relevant past interactions

---

## Observability

### LangSmith Tracing
- Every agent node traced with `@traceable` decorator
- Captures: input/output, token counts, latency, cost
- Hierarchical trace tree visible in LangSmith dashboard
- Environment variables: `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`

### Execution Trace
- Every agent appends to `execution_trace` in state
- Shows: agent name, action, input/output summary, duration
- Displayed in the UI trace panel

---

## Hallucination Control

1. **Strict prompting**: "Use ONLY the provided context. If not found, say not available."
2. **Validation Agent**: Checks every claim against retrieved context
3. **Citation enforcement**: Answer must reference [Document, Page X] for every claim
4. **CRAG loop**: Can re-retrieve or rewrite answer on validation failure
5. **Confidence scoring**: 0.0 (no support) to 1.0 (fully grounded)
6. **Information gaps**: System explicitly reports what was NOT found

---

## Explainability (Output)

Every response includes:
- **Answer** — comprehensive, cited text
- **Citations** — document name, page number, relevance score
- **Confidence score** — how well-grounded the answer is
- **Execution trace** — step-by-step agent actions with timing
- **Reasoning summary** — how the answer was derived
- **Information gaps** — what's missing from the context
- **Generated SQL** — if table agent was used
- **Query classification** — type, scope, intent

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LangGraph StateGraph |
| LLM (light) | Ollama: deepseek-r1:1.5b |
| LLM (heavy) | Ollama: deepseek-r1:7b |
| LLM (vision) | Ollama: llava (preprocessing only) |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | ChromaDB |
| Keyword search | rank_bm25 |
| Reranking | sentence-transformers cross-encoder |
| PDF processing | PyMuPDF (fitz) |
| OCR | EasyOCR (pip-installable, no system deps) |
| Table storage | SQLite |
| State persistence | LangGraph SqliteSaver |
| Long-term memory | LangGraph InMemoryStore |
| Tracing | LangSmith |
| API | FastAPI |
| UI | HTML + CSS + Vanilla JS |
| Evaluation | RAGAS + custom metrics |

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Ollama models
```bash
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
ollama pull llava          # optional, for image captioning
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your LangSmith API key
```

### 4. Add PDFs
Place ESG/sustainability PDF reports in `data/pdfs/`

### 5. Preprocess
```bash
python main.py ingest
```

### 6. Run the API
```bash
uvicorn app:app --reload --port 8000
```

### 7. Open UI
Navigate to `http://localhost:8000`

### 8. CLI mode
```bash
python main.py query "What are the main sustainability themes?"
python main.py chat   # Interactive mode
```

---

## Project Structure

```
agentic_rag/
├── app.py                    # FastAPI entry point
├── main.py                   # CLI entry point
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── SUMMARY.md                # This file — full system spec
├── config/settings.py        # Centralized configuration
├── graph/
│   ├── state.py              # AgentState + Pydantic structured output models
│   ├── workflow.py           # LangGraph StateGraph (7 nodes + routing)
│   └── checkpointer.py       # SqliteSaver + InMemoryStore
├── agents/
│   ├── query_understanding.py # Query classification + rewriting
│   ├── planner.py            # Adaptive routing
│   ├── retrieval.py          # Hybrid RAG + RAPTOR + Map-Reduce
│   ├── table_agent.py        # Text-to-SQL
│   ├── reasoning.py          # Answer synthesis
│   ├── validation.py         # CRAG validation
│   └── memory_agent.py       # LangGraph Store read/write
├── retrieval/
│   ├── chunking.py           # Parent-child chunking
│   ├── vector_store.py       # ChromaDB
│   ├── bm25_retriever.py     # BM25
│   ├── hybrid.py             # RRF fusion
│   ├── reranker.py           # Cross-encoder
│   └── raptor.py             # RAPTOR tree
├── ingestion/
│   ├── preprocessor.py       # Master pipeline
│   ├── pdf_processor.py      # Text extraction
│   ├── table_extractor.py    # Table → SQL
│   ├── image_extractor.py    # OCR + captioning
│   ├── ocr_engine.py         # EasyOCR wrapper
│   ├── contextualizer.py     # Anthropic-style context
│   └── schema_catalog.py     # Schema generator
├── storage/
│   ├── sql_store.py          # SQLite CRUD
│   └── schema_manager.py     # Schema introspection
├── evaluation/
│   ├── ragas_eval.py         # RAGAS metrics
│   └── custom_eval.py        # Semantic similarity + logging
├── utils/
│   ├── llm.py                # LLM routing (light/heavy/vision)
│   ├── embeddings.py         # Embedding wrapper
│   ├── cache.py              # LRU cache
│   └── logging_config.py     # Structured logging
├── static/                   # Frontend UI
└── data/                     # PDFs, images, databases
```



