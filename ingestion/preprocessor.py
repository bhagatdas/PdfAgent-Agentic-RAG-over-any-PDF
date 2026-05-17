"""
Master preprocessing pipeline — orchestrates the entire offline ingestion flow.

Flow:
  PDF → Text + Tables + Images → Chunk → Contextualize → Embed → Store
  + RAPTOR tree building
  + Schema catalog generation
"""

import logging
import time
from pathlib import Path
from typing import Optional

from config.settings import settings
from ingestion.pdf_processor import extract_text_from_pdf
from ingestion.table_extractor import extract_tables_from_pdf
from ingestion.image_extractor import extract_images_from_pdf
from ingestion.contextualizer import contextualize_chunks_batch
from ingestion.schema_catalog import generate_enhanced_catalog
from retrieval.chunking import create_parent_child_chunks
from retrieval.vector_store import vector_store
from retrieval.bm25_retriever import bm25_index
from retrieval.raptor import raptor_builder
from storage.sql_store import sql_store

logger = logging.getLogger(__name__)


def preprocess_all_pdfs(
    pdf_dir: Optional[str] = None,
    use_vision: bool = True,
    use_contextual: bool = True,
    use_raptor: bool = True,
    clear_existing: bool = False,
) -> dict:
    """
    Run the full preprocessing pipeline on all PDFs in the directory.

    Args:
        pdf_dir: Directory containing PDF files (default: settings.pdf_dir)
        use_vision: Whether to use vision model for image captioning
        use_contextual: Whether to apply contextual retrieval enrichment
        use_raptor: Whether to build RAPTOR hierarchical summary tree
        clear_existing: Whether to clear existing data before processing

    Returns:
        Summary dict with counts and timing
    """
    pdf_dir = Path(pdf_dir or settings.pdf_dir)

    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("PDF directory created (empty): %s", pdf_dir)
        return {"status": "no_pdfs", "message": f"Place PDF files in {pdf_dir}"}

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return {"status": "no_pdfs", "message": f"No PDFs found in {pdf_dir}"}

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("PDFs found: %d", len(pdf_files))
    logger.info("Options: vision=%s, contextual=%s, raptor=%s", use_vision, use_contextual, use_raptor)
    logger.info("=" * 60)

    start_time = time.time()

    # Clear existing data if requested
    if clear_existing:
        logger.info("Clearing existing data...")
        vector_store.clear()
        sql_store.drop_all_tables()
        bm25_index.clear()

    # Collect all chunks across all documents for RAPTOR
    all_chunks_for_raptor = []

    # Track statistics
    stats = {
        "pdfs_processed": 0,
        "total_pages": 0,
        "text_chunks": 0,
        "tables_extracted": 0,
        "images_processed": 0,
        "raptor_nodes": 0,
    }

    for pdf_path in pdf_files:
        logger.info("-" * 40)
        logger.info("Processing: %s", pdf_path.name)
        doc_stats = _process_single_pdf(
            str(pdf_path),
            use_vision=use_vision,
            use_contextual=use_contextual,
        )
        stats["pdfs_processed"] += 1
        stats["total_pages"] += doc_stats.get("pages", 0)
        stats["text_chunks"] += doc_stats.get("text_chunks", 0)
        stats["tables_extracted"] += doc_stats.get("tables", 0)
        stats["images_processed"] += doc_stats.get("images", 0)

        all_chunks_for_raptor.extend(doc_stats.get("raw_chunks", []))

    # Build BM25 index over all stored chunks
    logger.info("Building BM25 index...")
    all_stored = vector_store.get_all_documents()
    if all_stored:
        bm25_index.build_index(
            corpus=[doc["content"] for doc in all_stored],
            doc_ids=[doc["id"] for doc in all_stored],
            metadatas=[doc.get("metadata", {}) for doc in all_stored],
        )

    # Build RAPTOR tree if enabled
    if use_raptor and all_chunks_for_raptor:
        logger.info("Building RAPTOR hierarchical summary tree...")
        raptor_nodes = raptor_builder.build_tree(all_chunks_for_raptor)
        stats["raptor_nodes"] = len(raptor_nodes)
        logger.info("RAPTOR tree built — %d summary nodes created", len(raptor_nodes))

    # Generate enhanced schema catalog
    logger.info("Generating schema catalog...")
    generate_enhanced_catalog()

    elapsed = time.time() - start_time
    stats["total_time_seconds"] = round(elapsed, 1)

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)
    logger.info("=" * 60)

    return stats


def _process_single_pdf(
    pdf_path: str,
    use_vision: bool = True,
    use_contextual: bool = True,
) -> dict:
    """
    Process a single PDF through the full pipeline.

    Returns:
        Dict with processing statistics
    """
    doc_name = Path(pdf_path).stem
    stats = {"pages": 0, "text_chunks": 0, "tables": 0, "images": 0, "raw_chunks": []}

    # ── Step 1: Extract text ──
    logger.info("[1/5] Extracting text...")
    pages = extract_text_from_pdf(pdf_path)
    stats["pages"] = len(pages)

    # ── Step 2: Create parent-child chunks ──
    logger.info("[2/5] Chunking text (parent-child)...")
    chunks = []
    for page in pages:
        page_chunks = create_parent_child_chunks(
            text=page.text,
            document_name=page.document_name,
            page_number=page.page_number,
        )
        chunks.extend(page_chunks)

    # ── Step 3: Extract tables ──
    logger.info("[3/5] Extracting tables...")
    tables = extract_tables_from_pdf(pdf_path)
    stats["tables"] = len(tables)

    # Add table text representations as chunks
    for table in tables:
        if table.text_representation:
            chunks.append({
                "content": table.text_representation,
                "document_name": doc_name,
                "page_number": table.page_number,
                "chunk_type": "table_repr",
                "parent_id": f"table_{table.sql_table_name}",
                "chunk_id": f"table_{table.sql_table_name}_repr",
            })

    # ── Step 4: Extract images ──
    logger.info("[4/5] Extracting and processing images...")
    images = extract_images_from_pdf(pdf_path, use_vision_model=use_vision)
    stats["images"] = len(images)

    # Add image text as chunks
    for img in images:
        if img.combined_text:
            chunks.append({
                "content": img.combined_text,
                "document_name": doc_name,
                "page_number": img.page_number,
                "chunk_type": "image_caption",
                "parent_id": f"image_{img.page_number}_{img.image_index}",
                "chunk_id": f"img_{doc_name}_p{img.page_number}_i{img.image_index}",
            })

    # Save raw chunks for RAPTOR (before contextualization)
    stats["raw_chunks"] = [
        {"content": c["content"], "id": c.get("chunk_id", ""), "metadata": c}
        for c in chunks
        if c.get("chunk_type") != "table_repr"  # Don't include table reprs in RAPTOR
    ]

    # ── Step 5: Contextualize chunks ──
    if use_contextual and chunks:
        logger.info("[5/5] Contextualizing chunks...")
        chunks = contextualize_chunks_batch(chunks, doc_title=doc_name)
    else:
        logger.info("[5/5] Skipping contextualization")

    # ── Step 6: Embed and store in ChromaDB ──
    if chunks:
        logger.info("Storing %d chunks in vector store...", len(chunks))
        vector_store.add_chunks(chunks)
        stats["text_chunks"] = len(chunks)

    return stats
