# ESG Insight Pro — Agentic RAG System

**By Bhagat Labs** — An AI-powered, multi-agent ESG/Sustainability Report Intelligence platform built on LangGraph, Ollama, and ChromaDB.

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Ingestion Pipeline (Offline)](#ingestion-pipeline-offline)
7. [Retrieval Layer](#retrieval-layer)
8. [Storage Layer](#storage-layer)
9. [Multi-Agent Workflow (Runtime)](#multi-agent-workflow-runtime)
10. [Agents — Detailed](#agents--detailed)
11. [LangGraph State & Persistence](#langgraph-state--persistence)
12. [Evaluation](#evaluation)
13. [API & Frontend](#api--frontend)
14. [CLI Usage](#cli-usage)
15. [End-to-End Query Flow](#end-to-end-query-flow)

---

## Overview

ESG Insight Pro is a **Subject-Matter-Expert (SME)** RAG system for ESG/Sustainability PDF reports. It ingests PDFs (text, tables, images), builds a hierarchical retrieval index, and answers questions through a **7-agent LangGraph workflow** with corrective feedback loops (CRAG).

**Key features:**
- Multi-modal ingestion (text, tables → SQLite, images → OCR + vision LLM captions)
- Hybrid retrieval (BM25 + semantic) with cross-encoder reranking
- RAPTOR hierarchical summary tree for global queries
- Map-Reduce strategy for exhaustive aggregation queries
- Schema-aware SQL generation for tabular queries
- Structured-output agents (Pydantic) with retry/correction loops
- Short-term (SQLite checkpointer) + long-term (InMemoryStore) memory
- LangSmith tracing built-in
- FastAPI REST API + browser UI + CLI

---

## High-Level Architecture

```
                          ┌─────────────────────────────────┐
                          │  PDF INGESTION (offline)        │
                          │  text → tables → images         │
                          │  → chunks → contextualize       │
                          │  → embed → RAPTOR tree          │
                          └────────────────┬────────────────┘
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  ▼                        ▼                        ▼
            ┌──────────┐           ┌──────────────┐         ┌──────────────┐
            │ ChromaDB │           │   SQLite     │         │ Schema       │
            │ (vector) │           │  (tables)    │         │ Catalog      │
            └──────────┘           └──────────────┘         └──────────────┘
                  │                        │                        │
                  └────────────────────────┼────────────────────────┘
                                           ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │       RUNTIME — LangGraph Multi-Agent Workflow                   │
        │                                                                  │
        │   Query → Understanding → Memory Read → Planner                  │
        │     ↓                                                            │
        │     ├──→ Retrieval Agent (standard | RAPTOR | map-reduce)        │
        │     ├──→ Table Agent (NL → SQL → SQLite)                         │
        │     ↓                                                            │
        │   Reasoning → Validation (CRAG) → Memory Save → Answer           │
        │                     │                                            │
        │                     └─loop back─► Re-retrieve / Re-reason        │
        └──────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph 0.4+, LangChain 0.3+ |
| **LLM Backend** | Ollama (DeepSeek light/heavy, LLaVA vision, Nomic embeddings) |
| **Vector DB** | ChromaDB (cosine, HNSW) |
| **Sparse Retrieval** | rank_bm25 (BM25Okapi) |
| **Reranker** | sentence-transformers cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Tables** | SQLite + pandas |
| **PDF** | PyMuPDF (`fitz`), pymupdf4llm |
| **OCR** | EasyOCR (pip-installable, no system deps) |
| **API** | FastAPI + Uvicorn |
| **Tracing** | LangSmith |
| **Eval** | RAGAS + custom semantic similarity |
| **Frontend** | Vanilla HTML/CSS/JS chat UI |

---

## Project Structure

```
agentic_rag/
├── main.py                       # CLI entry (ingest / query / chat / schema)
├── app.py                        # FastAPI REST API
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py               # Pydantic Settings (env-driven)
├── utils/
│   ├── llm.py                    # ChatOllama wrapper, structured output, retry
│   ├── embeddings.py             # OllamaEmbeddings singleton
│   ├── cache.py                  # Thread-safe LRU cache w/ TTL
│   └── logging_config.py         # Colored console + file logging
├── ingestion/
│   ├── pdf_processor.py          # Text extraction (PyMuPDF)
│   ├── table_extractor.py        # Table detection → SQLite
│   ├── image_extractor.py        # Image extraction → OCR + vision caption
│   ├── ocr_engine.py             # EasyOCR wrapper
│   ├── contextualizer.py         # Anthropic-style chunk prefixing
│   ├── schema_catalog.py         # LLM-enhanced schema doc
│   └── preprocessor.py           # Master pipeline orchestrator
├── storage/
│   ├── sql_store.py              # SQLite table store (read-only at runtime)
│   └── schema_manager.py         # Schema catalog generator (cached)
├── retrieval/
│   ├── chunking.py               # Parent-child chunking
│   ├── vector_store.py           # ChromaDB abstraction
│   ├── bm25_retriever.py         # BM25 sparse index
│   ├── hybrid.py                 # RRF fusion of semantic + BM25
│   ├── reranker.py               # Cross-encoder reranking
│   └── raptor.py                 # Hierarchical KMeans summary tree
├── graph/
│   ├── state.py                  # AgentState TypedDict + Pydantic models
│   ├── checkpointer.py           # SqliteSaver + InMemoryStore
│   └── workflow.py               # StateGraph wiring with conditional routing
├── agents/
│   ├── query_understanding.py    # Classify type/scope/intent + rewrite
│   ├── planner.py                # Choose retrieval strategy
│   ├── memory_agent.py           # Read/save long-term memory
│   ├── retrieval.py              # Standard / RAPTOR / Map-Reduce
│   ├── table_agent.py            # NL → SQL → SQLite execution
│   ├── reasoning.py              # Synthesize answer with citations
│   └── validation.py             # CRAG: grounding & hallucination check
├── evaluation/
│   ├── ragas_eval.py             # RAGAS faithfulness/relevancy/precision
│   └── custom_eval.py            # Semantic similarity + response logging
├── static/
│   ├── index.html                # Chat UI
│   ├── app.js                    # Frontend logic
│   └── style.css
└── data/
    ├── pdfs/                     # Source PDFs
    ├── chroma/                   # ChromaDB persistence
    ├── images/                   # Extracted images
    ├── tables.db                 # SQLite table store
    └── logs/                     # Daily rolling logs
```

---

## Configuration

All settings live in [config/settings.py](config/settings.py) and load from `.env`:

| Group | Key Defaults |
|---|---|
| **Ollama** | `ollama_base_url=http://localhost:11434`, light/heavy = `deepseek-v3.1:671b-cloud`, vision = `llava`, embed = `nomic-embed-text` |
| **Storage** | `chroma_persist_dir=./data/chroma`, `sqlite_table_db=./data/tables.db`, `checkpoint_db=./data/checkpoints.db` |
| **Chunking** | child=400, parent=1600, overlap=50 chars |
| **Retrieval** | `top_k=20`, `rerank_top_k=5` |
| **RAPTOR** | `cluster_size=10`, `max_levels=3` |
| **Memory** | short-term 20 messages, long-term 200 items |
| **LangSmith** | `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` |

The LLM layer routes requests by **task type** (`light` / `heavy` / `vision` / `embed`) — see [utils/llm.py](utils/llm.py) — so swapping models is a one-line change.

---

## Ingestion Pipeline (Offline)

Entry point: [ingestion/preprocessor.py::preprocess_all_pdfs](ingestion/preprocessor.py).

For each PDF in `data/pdfs/`:

1. **Text extraction** — [pdf_processor.py](ingestion/pdf_processor.py) opens via PyMuPDF, per-page, cleans whitespace, rejoins hyphenated line breaks, drops page-number artifacts.
2. **Parent-child chunking** — [chunking.py](retrieval/chunking.py) creates **parent chunks (~1600 chars)** for context expansion and **child chunks (~400 chars)** for precise retrieval. Each child carries its `parent_id` so the retrieval agent can pull richer context at synthesis time. Splitting prefers sentence boundaries.
3. **Table extraction** — [table_extractor.py](ingestion/table_extractor.py) uses `page.find_tables()`, converts to pandas, auto-detects headers, coerces numeric columns, stores in SQLite via [sql_store.py](storage/sql_store.py), and writes a **text representation** of each table into the vector store so retrieval can route to the table agent.
4. **Image extraction** — [image_extractor.py](ingestion/image_extractor.py) extracts images >100px, runs EasyOCR ([ocr_engine.py](ingestion/ocr_engine.py)), and (optionally) generates a caption via the vision LLM. Output is a combined OCR+caption chunk indexed alongside text.
5. **Contextualization** — [contextualizer.py](ingestion/contextualizer.py) prepends a 2-3 sentence `[Context: ...]` prefix to each chunk (Anthropic Contextual Retrieval pattern), claimed to reduce retrieval failures ~49%.
6. **Embedding & storage** — chunks upserted to ChromaDB ([vector_store.py](retrieval/vector_store.py)) in batches of 100. BM25 index ([bm25_retriever.py](retrieval/bm25_retriever.py)) built from non-parent chunks.
7. **RAPTOR tree** — [raptor.py](retrieval/raptor.py) clusters leaf chunks via KMeans on embeddings, summarizes each cluster with the LLM, recurses up to `max_levels=3`. Higher levels (L2/L3) are document/corpus-level summaries used by **global queries**.
8. **Schema catalog** — [schema_catalog.py](ingestion/schema_catalog.py) walks SQLite tables, asks the LLM for per-column descriptions, and emits a Databricks-style catalog used by the Table Agent's SQL prompt.

CLI: `python main.py ingest [--clear] [--no-vision] [--no-contextual] [--no-raptor]`

---

## Retrieval Layer

### Vector Store ([retrieval/vector_store.py](retrieval/vector_store.py))
- ChromaDB `PersistentClient` with cosine HNSW
- Custom embedding function wraps Ollama
- Supports metadata filtering (`chunk_type`, `raptor_level`, `parent_id`, etc.)
- `get_parent_chunk()` for context expansion
- `search_by_raptor_level()` for hierarchical queries

### BM25 ([retrieval/bm25_retriever.py](retrieval/bm25_retriever.py))
- `BM25Okapi` over whitespace/lowercase tokens (>1 char)
- Built during ingestion, queried at runtime

### Hybrid ([retrieval/hybrid.py](retrieval/hybrid.py))
- Runs semantic + BM25 in parallel
- Merges via **Reciprocal Rank Fusion**: `score(d) = Σ w_i / (k + rank_i(d))`
- Default weights: semantic 0.6, BM25 0.4, k=60

### Reranker ([retrieval/reranker.py](retrieval/reranker.py))
- Cross-encoder `ms-marco-MiniLM-L-6-v2` (lazy-loaded)
- Scores (query, doc) pairs directly — slower but more precise than cosine

### RAPTOR ([retrieval/raptor.py](retrieval/raptor.py))
- Levels: `0` raw chunks → `1` cluster summaries → `2` section summaries → `3` doc summary
- KMeans clusters on embeddings, abstractive LLM summarization per cluster
- Indexed back into ChromaDB with `raptor_level` metadata — retrievable by tree depth

---

## Storage Layer

### [storage/sql_store.py](storage/sql_store.py) — `SQLTableStore`
- One SQLite file (`data/tables.db`), one table per extracted PDF table
- `_table_metadata` table stores source PDF, page, description, column info
- **Runtime safety**: `execute_query()` enforces SELECT-only and blocks `DROP/DELETE/INSERT/UPDATE/ALTER/CREATE/TRUNCATE`
- Table & column names sanitized to alphanumeric+`_`, 64-char limit
- Pandas auto-coercion for numeric-looking columns (strips `,`, `%`, `$`)

### [storage/schema_manager.py](storage/schema_manager.py) — `SchemaManager`
- Caches a formatted catalog string for LLM prompts
- Lists every table with: source doc/page, description, columns (name+type+nullable), row count, 3-row sample
- `invalidate_cache()` called after ingestion

---

## Multi-Agent Workflow (Runtime)

Wired in [graph/workflow.py](graph/workflow.py):

```
START
  └─→ query_understanding
        └─→ memory_read
              └─→ planner
                    │
                    ├─ "direct_reasoning"      → reasoning
                    ├─ "table_only"            → table_agent ─→ reasoning
                    ├─ "retrieval_only"        → retrieval   ─→ reasoning
                    └─ "retrieval_then_table"  → retrieval ─→ table_agent ─→ reasoning
                                                              │
                                                              ▼
                                                          reasoning
                                                              │
                                                              ▼
                                                          validation
                                                              │
                              ┌───────────────────────────────┤
                              │                               │
                       verdict=pass               verdict=rewrite/re_retrieve
                              │                       (retries < 2)
                              ▼                               │
                         memory_save                  back to reasoning/retrieval
                              │
                              ▼
                             END
```

**Routing functions** (in `workflow.py`):
- `_plan_router` — picks the post-planner edge based on `retrieval_strategy` and `use_table_agent`
- `_post_retrieval_router` — runs the table agent after retrieval if planner requested both
- `_validation_router` — **CRAG corrective loop**: re-reason (≤2x), re-retrieve (≤2x), or proceed

---

## Agents — Detailed

All agents are `@traceable` (LangSmith) and return state updates with an `execution_trace` entry (operator-add accumulated).

### 1. Query Understanding — [agents/query_understanding.py](agents/query_understanding.py)
Produces a `QueryAnalysis` Pydantic with:
- `rewritten_query` — pronouns resolved, follow-ups expanded
- `query_type` ∈ `text_query | table_query | mixed`
- `query_scope` ∈ `local | global | aggregation | direct`
- `intent` ∈ `factual | comparative | trend | definition`
- `entities`, `is_followup`

### 2. Memory Read — [agents/memory_agent.py](agents/memory_agent.py)
- Searches `InMemoryStore` namespaced by `user_id`
- Returns top-5 relevant past Q&A summaries as `memory_context`

### 3. Planner — [agents/planner.py](agents/planner.py)
Returns `ExecutionPlan` with `retrieval_strategy`:

| Scope | Strategy | When |
|---|---|---|
| `local` | `standard` | Specific fact lookup |
| `global` | `raptor_global` | Themes/overviews |
| `aggregation` | `map_reduce` | "List ALL...", "Count every..." (expensive) |
| `direct` | `none` | Pure definitions, no docs needed |

`use_table_agent` flips on for `table_query` or `mixed`. Falls back to rule-based plan if structured output fails. `map_reduce` sets `needs_confirmation=True`.

### 4. Retrieval Agent — [agents/retrieval.py](agents/retrieval.py)
Three modes:
- **`_standard_retrieval`** — Multi-query rewriting (LLM generates 2 variants + original), hybrid search per variant, dedup by content prefix, rerank → top-5, then parent-chunk expansion for context.
- **`_raptor_retrieval`** — Searches L1–L3 summaries (`top_k=8`) plus a few L0 leaves (`top_k=5`) for supporting detail.
- **`_map_reduce_retrieval`** — Iterates ALL stored chunks; per-chunk LLM extraction of target entities; reduces to a unique sorted set. Slow but exhaustive.

### 5. Table Agent — [agents/table_agent.py](agents/table_agent.py)
- Injects the cached schema catalog into a SQL prompt
- Uses structured output `SQLGeneration { sql_query, explanation, tables_used, can_answer }`
- Executes via `sql_store.safe_execute()` (SELECT-only)
- On failure, increments `sql_retries`; the validation loop can route back

### 6. Reasoning — [agents/reasoning.py](agents/reasoning.py)
- Uses **heavy** LLM with strict structured output `ReasonedAnswer { answer, citations, confidence, reasoning_summary, gaps }`
- Hard system prompt: **only use provided context**, return exact refusal strings for off-topic / unsupported queries, cite every claim as `[Document, Page X]`
- Receives all of: retrieved chunks, SQL results, map-reduce items, memory context, and validation feedback (when retrying)

### 7. Validation (CRAG) — [agents/validation.py](agents/validation.py)
- Checks: grounding, citations, hallucination, completeness
- Returns `ValidationResult.verdict` ∈ `pass | rewrite_answer | re_retrieve | give_up`
- Up to 2 retries each on rewrite/re_retrieve

### Memory Save — [agents/memory_agent.py](agents/memory_agent.py)
- Persists `{query, rewritten_query, answer_summary, entities, confidence, timestamp}` to the store at key `interaction_<ts>` under `("user_memory", user_id)`

---

## LangGraph State & Persistence

### [graph/state.py](graph/state.py) — `AgentState` (TypedDict)
Fields cluster into: conversation (`messages` auto-merged via `add_messages`), query understanding, planning, memory, retrieval, table/SQL, reasoning, validation, observability (`execution_trace` accumulated via `operator.add`).

### [graph/checkpointer.py](graph/checkpointer.py)
- **`SqliteSaver`** — per-thread short-term state survives restarts (one row per checkpoint)
- **`InMemoryStore`** — long-term cross-session memory namespaced by `user_id` (not durable across processes)

---

## Evaluation

### [evaluation/ragas_eval.py](evaluation/ragas_eval.py)
- `evaluate_with_ragas(test_data)` runs **faithfulness**, **answer_relevancy**, **context_precision** on a `(question, answer, contexts, ground_truth)` dataset; persists JSON results.

### [evaluation/custom_eval.py](evaluation/custom_eval.py)
- `semantic_similarity(a, b)` — cosine via Ollama embeddings
- `ResponseLogger` — appends one JSONL line per query to `data/response_log.jsonl` (timestamp, query, answer, confidence, strategy, citations, full trace). Called from `app.py::/query`.

---

## API & Frontend

### FastAPI — [app.py](app.py)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/`       | Serve UI |
| `GET`  | `/health` | Vector store doc count + table count |
| `POST` | `/query`  | Run full pipeline; returns answer + citations + trace + duration |
| `POST` | `/ingest` | Trigger preprocessing |
| `POST` | `/upload` | Save uploaded PDF to `data/pdfs/` |
| `GET`  | `/schema` | Return schema catalog string |

`QueryResponse` returns: `answer`, `citations`, `confidence_score`, `query_type/scope`, `retrieval_strategy`, optional `generated_sql/sql_results`, `execution_trace`, `reasoning_summary`, `information_gaps`, `thread_id`, `duration_ms`.

### Frontend — [static/](static/)
- [index.html](static/index.html) — sidebar (brand + suggested queries), main chat, collapsible details panel
- [app.js](static/app.js) — `submitQuery()` POSTs to `/query`, renders messages, color-coded confidence chip (`>=0.7` green, `>=0.4` amber, else red), citations as chips, SQL block, execution-trace timing, query metadata grid
- New chat = fresh `thread_id`; Ctrl+Enter or Enter submits

---

## CLI Usage

```bash
python main.py ingest                                 # Preprocess all PDFs
python main.py ingest --clear --no-vision             # Re-ingest, skip vision captions
python main.py query "What are Scope 1 emissions?"    # Single query
python main.py chat                                   # Interactive REPL
python main.py schema                                 # Print schema catalog
```

`_print_result()` shows answer + confidence, sources, generated SQL, info gaps, and the full per-agent timing trace.

---

## End-to-End Query Flow

For `"List all ESG initiatives mentioned in the Honeywell report"`:

1. **QueryUnderstanding** → `query_type=text_query`, `query_scope=aggregation`, entities=`["ESG initiatives", "Honeywell"]`
2. **MemoryRead** → finds 0 relevant past interactions for this thread
3. **Planner** → `retrieval_strategy=map_reduce`, `use_table_agent=false`, sets `needs_confirmation=true`
4. **Retrieval** → scans every chunk, per-chunk LLM extraction, dedup; returns `map_reduce_results` (sorted unique items) + top-10 supporting chunks
5. **Reasoning** (heavy LLM) → enumerated answer with `[document, page]` citations, confidence ~0.9, info gaps listed
6. **Validation** → `verdict=pass`, `is_grounded=true`
7. **MemorySave** → stores interaction summary under `("user_memory", user_id)`
8. **API response** → JSON with answer, citations, full trace; UI renders with confidence chip and details panel

For `"What were Scope 1 emissions in 2023?"`:

1. **QueryUnderstanding** → `query_type=table_query`, `query_scope=local`
2. **Planner** → `retrieval_strategy=standard`, `use_table_agent=true`
3. **Retrieval** → hybrid + rerank → top-5 text chunks
4. **TableAgent** → reads schema catalog, generates `SELECT scope1, year FROM hon_esg_report_p12_t1 WHERE year=2023`, executes
5. **Reasoning** → fuses text context + SQL rows, cites both `[honeywell.pdf, Page 12]` and the table
6. **Validation → MemorySave → END**

---

*Generated documentation — describes the codebase as of the current working directory snapshot.*
