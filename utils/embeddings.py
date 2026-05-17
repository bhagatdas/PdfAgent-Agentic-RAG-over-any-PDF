"""
Embedding model wrapper using Ollama.
Provides batch embedding and caching support.
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_ollama import OllamaEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> OllamaEmbeddings:
    """Get the singleton embedding model instance."""
    logger.info("Initializing embedding model — model=%s", settings.ollama_model_embed)
    return OllamaEmbeddings(
        model=settings.ollama_model_embed,
        base_url=settings.ollama_base_url,
    )


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings."""
    if not texts:
        return []
    model = get_embedding_model()
    logger.debug("Embedding batch of %d texts", len(texts))
    return model.embed_documents(texts)
