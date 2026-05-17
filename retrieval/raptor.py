"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.
Builds a hierarchical summary tree during preprocessing to solve the Harry Potter problem.

Levels:
  0 = raw child chunks (leaf nodes)
  1 = cluster summaries (page-level themes)
  2 = section summaries (section-level themes)
  3 = document summary (corpus-level overview)
"""

import logging
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from config.settings import settings
from utils.llm import invoke_llm
from utils.embeddings import embed_texts
from retrieval.vector_store import vector_store

logger = logging.getLogger(__name__)

RAPTOR_SUMMARY_PROMPT = """You are summarizing a cluster of related text chunks from an ESG/sustainability report.
Create a comprehensive summary that captures ALL key facts, data points, and themes from these chunks.
The summary should be self-contained — someone reading only this summary should understand the main points.

TEXT CHUNKS:
{chunks_text}

COMPREHENSIVE SUMMARY:"""


class RaptorTreeBuilder:
    """
    Builds a RAPTOR hierarchical summary tree from raw text chunks.

    Architecture:
      Level 0: Raw chunks (already in vector store)
      Level 1: Summaries of chunk clusters (~10-15 chunks each)
      Level 2: Summaries of Level 1 clusters
      Level 3: Overall document/corpus summary

    For global queries, search Level 2-3.
    For local queries, search Level 0.
    """

    def __init__(
        self,
        cluster_size: Optional[int] = None,
        max_levels: Optional[int] = None,
    ):
        self.cluster_size = cluster_size or settings.raptor_cluster_size
        self.max_levels = max_levels or settings.raptor_max_levels

    def build_tree(self, chunks: list[dict]) -> list[dict]:
        """
        Build the full RAPTOR tree from raw chunks.

        Args:
            chunks: List of chunk dicts with 'content', 'id', 'metadata'

        Returns:
            List of all generated summary nodes (Levels 1+)
        """
        if not chunks or len(chunks) < 3:
            logger.warning("Too few chunks for RAPTOR tree: %d", len(chunks))
            return []

        all_summary_nodes = []
        current_level_texts = [c["content"] for c in chunks]
        current_level_ids = [c.get("id", f"chunk_{i}") for i, c in enumerate(chunks)]

        for level in range(1, self.max_levels + 1):
            if len(current_level_texts) <= 2:
                logger.info("RAPTOR: stopping at level %d (only %d nodes)", level, len(current_level_texts))
                break

            logger.info(
                "RAPTOR: building level %d from %d nodes (cluster_size=%d)",
                level, len(current_level_texts), self.cluster_size,
            )

            # Cluster current level nodes
            clusters = self._cluster_texts(current_level_texts, current_level_ids)

            # Generate summaries for each cluster
            next_level_texts = []
            next_level_ids = []

            for cluster_idx, cluster in enumerate(clusters):
                summary = self._summarize_cluster(cluster["texts"])

                if not summary.strip():
                    continue

                node_id = f"raptor_L{level}_C{cluster_idx}"

                # Store in vector store
                summary_node = {
                    "content": summary,
                    "chunk_id": node_id,
                    "document_name": "raptor_summary",
                    "page_number": 0,
                    "chunk_type": f"raptor_L{level}",
                    "parent_id": node_id,
                    "raptor_level": level,
                }

                vector_store.add_chunks([summary_node])
                all_summary_nodes.append(summary_node)

                next_level_texts.append(summary)
                next_level_ids.append(node_id)

            current_level_texts = next_level_texts
            current_level_ids = next_level_ids

            logger.info(
                "RAPTOR level %d complete — %d summary nodes created",
                level, len(next_level_texts),
            )

        logger.info("RAPTOR tree complete — total summary nodes: %d", len(all_summary_nodes))
        return all_summary_nodes

    def _cluster_texts(
        self,
        texts: list[str],
        ids: list[str],
    ) -> list[dict]:
        """
        Cluster texts by semantic similarity using KMeans.

        Returns:
            List of clusters, each with 'texts', 'ids'
        """
        if len(texts) <= self.cluster_size:
            return [{"texts": texts, "ids": ids}]

        # Embed all texts
        embeddings = embed_texts(texts)
        embeddings_array = np.array(embeddings)

        # Determine number of clusters
        n_clusters = max(2, len(texts) // self.cluster_size)
        n_clusters = min(n_clusters, len(texts) - 1)

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_array)

        # Group texts by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {"texts": [], "ids": []}
            clusters[label]["texts"].append(texts[i])
            clusters[label]["ids"].append(ids[i])

        return list(clusters.values())

    def _summarize_cluster(self, texts: list[str]) -> str:
        """Generate an abstractive summary of a cluster of texts."""
        # Concatenate cluster texts (with truncation to fit context)
        combined = "\n---\n".join(texts)
        if len(combined) > 4000:
            combined = combined[:4000] + "\n... (truncated)"

        try:
            prompt = RAPTOR_SUMMARY_PROMPT.format(chunks_text=combined)
            summary = invoke_llm(prompt, task_type="light")
            return summary.strip()
        except Exception as e:
            logger.error("RAPTOR summarization failed: %s", e)
            return ""


# Singleton instance
raptor_builder = RaptorTreeBuilder()
