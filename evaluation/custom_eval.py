"""
Custom evaluation — semantic similarity scoring, response logging, and timing.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from utils.embeddings import embed_texts

logger = logging.getLogger(__name__)


def semantic_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using embeddings."""
    embeddings = embed_texts([text1, text2])
    a, b = np.array(embeddings[0]), np.array(embeddings[1])
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class ResponseLogger:
    """Logs all queries and responses for analysis and evaluation."""

    def __init__(self, log_path: str = "./data/response_log.jsonl"):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, query: str, result: dict, duration_ms: float) -> None:
        """Log a query-response pair."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence_score", 0.0),
            "citations": result.get("citations", []),
            "query_type": result.get("query_type", ""),
            "query_scope": result.get("query_scope", ""),
            "retrieval_strategy": result.get("retrieval_strategy", ""),
            "chunks_retrieved": len(result.get("retrieved_chunks", [])),
            "sql_used": bool(result.get("generated_sql")),
            "validated": result.get("is_validated", False),
            "duration_ms": duration_ms,
            "execution_trace": result.get("execution_trace", []),
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.debug("Response logged — query='%s...'", query[:50])

    def get_logs(self, limit: int = 100) -> list[dict]:
        """Read recent log entries."""
        if not Path(self.log_path).exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries[-limit:]


# Singleton logger
response_logger = ResponseLogger()
