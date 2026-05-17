"""
BM25 sparse retrieval using rank_bm25.
Provides keyword-based search to complement semantic vector search.
"""

import logging
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """
    BM25 keyword-based retrieval index.
    Built during preprocessing, queried during retrieval.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._corpus: list[str] = []
        self._doc_ids: list[str] = []
        self._metadatas: list[dict] = []
        self._is_built = False

    def build_index(
        self,
        corpus: list[str],
        doc_ids: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """
        Build the BM25 index from a corpus of text documents.

        Args:
            corpus: List of document text strings
            doc_ids: Corresponding document IDs
            metadatas: Optional metadata dicts for each document
        """
        if not corpus:
            logger.warning("Empty corpus — BM25 index not built")
            return

        self._corpus = corpus
        self._doc_ids = doc_ids
        self._metadatas = metadatas or [{}] * len(corpus)

        # Tokenize corpus for BM25
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._is_built = True

        logger.info("BM25 index built — documents=%d", len(corpus))

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of dicts with content, metadata, score, and id
        """
        if not self._is_built or self._bm25 is None:
            logger.warning("BM25 index not built — returning empty results")
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "content": self._corpus[idx],
                    "metadata": self._metadatas[idx],
                    "score": float(scores[idx]),
                    "id": self._doc_ids[idx],
                })

        logger.debug("BM25 search — query='%s...', results=%d", query[:50], len(results))
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace tokenizer with lowercasing and cleanup."""
        import re
        # Lowercase, remove special characters, split on whitespace
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        # Remove very short tokens
        return [t for t in tokens if len(t) > 1]

    def clear(self) -> None:
        """Clear the BM25 index."""
        self._bm25 = None
        self._corpus = []
        self._doc_ids = []
        self._metadatas = []
        self._is_built = False
        logger.info("BM25 index cleared")

    @property
    def is_ready(self) -> bool:
        """Check if the index is built and ready for queries."""
        return self._is_built


# Singleton instance
bm25_index = BM25Index()
