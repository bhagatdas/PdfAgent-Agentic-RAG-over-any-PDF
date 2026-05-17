# Agentic-RAG-System-on-ESG-Environmental-Social-and-Governance-data

**ESG Insight Pro** — A multi-agent RAG system for ESG / Sustainability report intelligence, built on **LangGraph**, **Ollama**, and **ChromaDB**.

Ingests ESG PDFs (text, tables, images), builds a hierarchical retrieval index, and answers questions through a **7-agent LangGraph workflow** with corrective feedback loops (CRAG).

> Full architecture, agent details, and end-to-end flow: see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
> Workflow diagram: [workflow.png](workflow.png).

---

## Highlights

- Multi-modal ingestion: text + tables (→ SQLite) + images (OCR + vision LLM captions)
- Hybrid retrieval (BM25 + semantic) with cross-encoder reranking
- **RAPTOR** hierarchical summary tree for global queries
- **Map-Reduce** strategy for exhaustive aggregation queries
- Schema-aware **NL → SQL** for table queries (SELECT-only enforcement)
- Structured-output agents (Pydantic) with retry / CRAG correction
- Planner short-circuits greetings / off-topic queries straight to END
- Short-term (SQLite checkpointer) + long-term (Store) memory
- LangSmith tracing on every node
- FastAPI REST API + browser chat UI + CLI

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# edit .env — set LANGCHAIN_API_KEY etc.

# 3. Start Ollama (separate terminal) and pull the models referenced in .env
ollama serve

# 4. Drop PDFs into data/pdfs/ then ingest
python main.py ingest

# 5a. CLI chat
python main.py chat

# 5b. or REST API + browser UI
uvicorn app:app --reload --port 8000
# → http://localhost:8000
```

## CLI

```bash
python main.py ingest                                 # Preprocess all PDFs
python main.py ingest --clear --no-vision             # Re-ingest, skip vision captions
python main.py query "What are Scope 1 emissions?"    # Single query
python main.py chat                                   # Interactive REPL
python main.py schema                                 # Print schema catalog
```

## Architecture (at a glance)

```
PDF → text + tables + images
     → chunk (parent/child) → contextualize → embed → ChromaDB
     → RAPTOR hierarchical summaries
     → SQLite table store + schema catalog

Query → Understanding → Memory Read → Planner
              ↓
       ├─ end_early (greetings / off-topic) ──────────────────────► END
       ├─ Retrieval (RAG)        ─┐
       ├─ Table Agent (SQL)       ├─► Reasoning → Validation (CRAG) → Memory Save → END
       └─ Both (RAG + Table)     ─┘                  │
                                                     └─ retry retrieval / reasoning (≤2x)
```

See [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) for the full design.

## Tech stack

| Layer | Tech |
|---|---|
| Orchestration | LangGraph 0.4+, LangChain 0.3+ |
| LLM | Ollama (DeepSeek light/heavy, LLaVA vision, Nomic embed) |
| Vector DB | ChromaDB (cosine, HNSW) |
| Sparse retrieval | rank_bm25 |
| Reranker | sentence-transformers cross-encoder |
| Tables | SQLite + pandas |
| PDF | PyMuPDF + pymupdf4llm |
| OCR | EasyOCR (pip-installable) |
| API | FastAPI + Uvicorn |
| Tracing | LangSmith |
| Eval | RAGAS + custom semantic similarity |

## License

Internal / proprietary — see repository owner.
