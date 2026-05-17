"""
FastAPI application — REST API for the Sustainability SME system.
Endpoints: /query, /ingest, /health, and serves the frontend UI.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils.logging_config import setup_logging

# Initialize logging before anything else
setup_logging(level="INFO")

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ESG Insight Pro",
    description="AI-powered ESG Report Intelligence by Bhagat Labs",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (UI)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Request/Response Models ──

class QueryRequest(BaseModel):
    query: str = Field(..., description="User's question")
    thread_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: str = Field(default="default", description="User ID for long-term memory")

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence_score: float
    query_type: str
    query_scope: str
    retrieval_strategy: str
    generated_sql: Optional[str] = None
    sql_results: Optional[list[dict]] = None
    execution_trace: list[dict]
    reasoning_summary: str
    information_gaps: list[str]
    thread_id: str
    duration_ms: float

class IngestRequest(BaseModel):
    use_vision: bool = Field(default=True, description="Use vision model for image captioning")
    use_contextual: bool = Field(default=True, description="Apply contextual retrieval enrichment")
    use_raptor: bool = Field(default=True, description="Build RAPTOR hierarchical tree")
    clear_existing: bool = Field(default=False, description="Clear existing data before ingesting")


# ── Endpoints ──

@app.get("/")
async def serve_ui():
    """Serve the frontend UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "UI not found. Place index.html in static/"})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from retrieval.vector_store import vector_store
    from storage.sql_store import sql_store
    return {
        "status": "healthy",
        "vector_store_docs": vector_store.count,
        "sql_tables": len(sql_store.get_table_names()),
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint — runs the full multi-agent pipeline.
    Returns answer, citations, confidence, execution trace, and more.
    """
    from graph.workflow import invoke_query
    from evaluation.custom_eval import response_logger

    start = time.time()
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = invoke_query(
            query=request.query,
            thread_id=thread_id,
            user_id=request.user_id,
        )
    except Exception as e:
        logger.error("Query failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    duration = (time.time() - start) * 1000

    response = QueryResponse(
        answer=result.get("answer", "An error occurred while processing your query."),
        citations=result.get("citations", []),
        confidence_score=result.get("confidence_score", 0.0),
        query_type=result.get("query_type", "unknown"),
        query_scope=result.get("query_scope", "unknown"),
        retrieval_strategy=result.get("retrieval_strategy", "unknown"),
        generated_sql=result.get("generated_sql"),
        sql_results=result.get("sql_results") if result.get("sql_results") else None,
        execution_trace=result.get("execution_trace", []),
        reasoning_summary=result.get("reasoning_summary", ""),
        information_gaps=result.get("information_gaps", []),
        thread_id=thread_id,
        duration_ms=round(duration, 1),
    )

    # Log the response for evaluation
    response_logger.log(request.query, result, duration)

    return response


@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """Trigger PDF preprocessing pipeline."""
    from ingestion.preprocessor import preprocess_all_pdfs

    logger.info("Ingestion triggered via API")
    try:
        stats = preprocess_all_pdfs(
            use_vision=request.use_vision,
            use_contextual=request.use_contextual,
            use_raptor=request.use_raptor,
            clear_existing=request.clear_existing,
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error("Ingestion failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file for processing."""
    from config.settings import settings

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_dir = Path(settings.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / file.filename

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    logger.info("PDF uploaded: %s (%d bytes)", file.filename, len(content))
    return {"status": "uploaded", "filename": file.filename, "size_bytes": len(content)}


@app.get("/schema")
async def get_schema():
    """Get the current database schema catalog."""
    from storage.schema_manager import schema_manager
    return {"catalog": schema_manager.generate_catalog()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
