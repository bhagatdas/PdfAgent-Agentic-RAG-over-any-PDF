"""
ChromaDB vector store abstraction.
Designed to be swappable with FAISS, Qdrant, or Weaviate.
"""

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from utils.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-backed vector store for document chunks.
    Provides semantic search with metadata filtering.
    """

    def __init__(
        self,
        collection_name: str = "sustainability_chunks",
        persist_dir: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._embedding_fn = self._create_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Vector store initialized — collection=%s, docs=%d",
            self.collection_name,
            self._collection.count(),
        )

    def _create_embedding_function(self):
        """Create a ChromaDB-compatible embedding function using Ollama."""
        from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
        from utils.embeddings import embed_texts

        class OllamaEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return embed_texts(list(input))

        return OllamaEmbeddingFunction()

    def add_chunks(self, chunks: list[dict]) -> None:
        """
        Add chunks to the vector store.

        Each chunk dict should have:
        - content: str
        - chunk_id: str
        - document_name: str
        - page_number: int
        - chunk_type: str (child, parent, table_repr, image_caption)
        - parent_id: str
        - raptor_level: int (optional)
        """
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", f"chunk_{len(ids)}")

            # Avoid duplicates
            if chunk_id in ids:
                chunk_id = f"{chunk_id}_{len(ids)}"

            ids.append(chunk_id)
            documents.append(chunk["content"])
            metadatas.append({
                "document_name": chunk.get("document_name", ""),
                "page_number": chunk.get("page_number", 0),
                "chunk_type": chunk.get("chunk_type", "child"),
                "parent_id": chunk.get("parent_id", ""),
                "raptor_level": chunk.get("raptor_level", 0),
            })

        # Add in batches (ChromaDB has limits)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        logger.info("Added %d chunks to vector store", len(ids))

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            where: Metadata filter (e.g., {"chunk_type": "child"})
            where_document: Document content filter

        Returns:
            List of dicts with content, metadata, score, and id
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count() or 1),
        }

        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        try:
            results = self._collection.query(**kwargs)
        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return []

        # Format results
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                    "id": results["ids"][0][i] if results["ids"] else "",
                })

        logger.debug("Vector search — query='%s...', results=%d", query[:50], len(formatted))
        return formatted

    def search_by_raptor_level(
        self,
        query: str,
        min_level: int = 0,
        max_level: int = 3,
        top_k: int = 10,
    ) -> list[dict]:
        """Search only RAPTOR nodes at specific tree levels."""
        return self.search(
            query=query,
            top_k=top_k,
            where={
                "$and": [
                    {"raptor_level": {"$gte": min_level}},
                    {"raptor_level": {"$lte": max_level}},
                ]
            },
        )

    def get_parent_chunk(self, parent_id: str) -> Optional[dict]:
        """Retrieve the parent chunk for context expansion."""
        try:
            results = self._collection.get(
                where={"parent_id": parent_id, "chunk_type": "parent"},
                limit=1,
            )
            if results and results["documents"]:
                return {
                    "content": results["documents"][0],
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                    "id": results["ids"][0] if results["ids"] else "",
                }
        except Exception:
            pass
        return None

    def get_all_documents(self) -> list[dict]:
        """Get all documents in the collection (for BM25 indexing)."""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.get(
            limit=count,
            where={"chunk_type": {"$ne": "parent"}},  # Exclude parent chunks
        )

        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                docs.append({
                    "content": doc,
                    "id": results["ids"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

        return docs

    def clear(self) -> None:
        """Delete all documents from the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error("Failed to clear vector store: %s", e)

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count()


# Singleton instance
vector_store = VectorStore()
