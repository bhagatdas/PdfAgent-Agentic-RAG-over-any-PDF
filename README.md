# PdfAgent — Agentic RAG over any PDF

> Grounded answers from your PDFs, with source citations.

[![PyPI](https://img.shields.io/pypi/v/agentic-rag-pdf.svg)](https://pypi.org/project/agentic-rag-pdf/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-1c3d5a.svg)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8+-4b3f72.svg)](https://github.com/facebookresearch/faiss)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-000000.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](#license)

Ask questions about any PDF — research reports, contracts, manuals, ESG / sustainability filings — and get **grounded, cited answers**. Drop a PDF into the browser UI and start asking.

![PdfAgent UI](https://raw.githubusercontent.com/bhagatdas/PdfAgent-Agentic-RAG-over-any-PDF/master/docs/ui-screenshot.png)

---

## Quick Start

```bash
# 1. Install
pip install agentic-rag-pdf

# 2. Install + start Ollama (https://ollama.com/download), pull the embedding model
ollama pull mxbai-embed-large
ollama signin                       # only if using cloud models (e.g. gpt-oss:120b-cloud)

# 3. Launch the UI
agentic-rag-pdf
```

Open **http://localhost:8000** → click **Upload PDF** → drop a file → ask questions.

> The same command is also installed under the shorter alias `pdfagent` if you prefer less typing. Both are identical.

---

## What it does

- **Multi-modal ingestion** — text, tables (extracted into SQLite), images (OCR + optional vision captions).
- **Hybrid retrieval** — BM25 + FAISS dense, merged with Reciprocal Rank Fusion, with optional cross-encoder reranking.
- **RAPTOR** hierarchical summaries for global queries; **Map-Reduce** for exhaustive aggregation.
- **Schema-aware NL → SQL** for tabular queries (SELECT-only, sandboxed).
- **Hallucination defense (CRAG + 7 layers)** — pre-synthesis entity-metric fact extraction, deterministic attribution check, citation verification, claim-level faithfulness, cross-chunk contradiction detection, arithmetic verification of delta tables.
- **Three interfaces** — browser UI (SSE-streamed agent steps), REST API, CLI.

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the workflow diagram, full feature breakdown, configuration reference, REST API, and project structure.

---

## Other ways to install

**From the latest GitHub commit** (handy between releases):

```bash
pip install git+https://github.com/bhagatdas/PdfAgent-Agentic-RAG-over-any-PDF.git
```

**For development** (editable, clone the repo):

```bash
git clone https://github.com/bhagatdas/PdfAgent-Agentic-RAG-over-any-PDF.git
cd PdfAgent-Agentic-RAG-over-any-PDF
pip install -e .
agentic-rag-pdf --reload
```

---

## Command-line usage

The wheel installs two pairs of console scripts — long-form names that match the PyPI distribution, plus short aliases that mean the same thing:

```bash
agentic-rag-pdf                              # launch the browser UI (uvicorn on :8000)
agentic-rag-pdf --host 0.0.0.0 --port 8080   # custom bind
pdfagent                                     # same thing, shorter alias

agentic-rag-pdf-cli ingest                   # CLI ingest (drop files in data/pdfs/ first)
agentic-rag-pdf-cli query "..."              # one-shot question
agentic-rag-pdf-cli chat                     # interactive REPL
agentic-rag-pdf-cli schema                   # print the table schema catalog
pdfagent-cli ...                             # short alias for the same CLI
```

Override defaults via a `.env` file in your working directory. The full settings reference lives in [config/settings.py](config/settings.py).

---

## REST API at a glance

| Method | Path | Purpose |
|:------:|:-----|:--------|
| `POST` | `/query` | Run the pipeline; JSON response |
| `POST` | `/query/stream` | Same, streamed via SSE (one event per agent) |
| `POST` | `/upload-ingest/stream` | Wipe + ingest a single PDF |
| `GET`  | `/health` | Vector store doc count + table count |
| `GET`  | `/ollama/health` | Probe Ollama readiness + per-OS install help |

Full endpoint table and request/response shape: [docs/ARCHITECTURE.md#rest-api](docs/ARCHITECTURE.md#rest-api).

---

## Tech stack

**LangGraph** + LangChain · **Ollama** (gpt-oss:120b-cloud, mxbai-embed-large) · **FAISS** + rank-bm25 · cross-encoder reranker · **SQLite** for tables · **FastAPI** (SSE) · PyMuPDF · EasyOCR.

---

## Publishing

Releases go to PyPI as `agentic-rag-pdf` on every `v*` tag push. See [PUBLISHING.md](PUBLISHING.md) for the one-time trusted-publisher setup and the tag → release flow.

---

## License

Proprietary — see repository owner.

---

*Built by [Bhagat Kumar Das](https://github.com/bhagatdas).*
