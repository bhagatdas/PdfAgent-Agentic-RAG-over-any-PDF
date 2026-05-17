"""
In-memory LRU cache for LLM responses.
Avoids redundant calls for identical prompts.
Key: hash of (model_name + prompt). TTL-based expiry.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)


class LLMCache:
    """Thread-safe LRU cache with TTL expiry for LLM responses."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, model: str, prompt: str) -> str:
        """Create a deterministic cache key from model name and prompt."""
        raw = f"{model}::{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, model: str, prompt: str) -> Optional[Any]:
        """
        Retrieve a cached response if available and not expired.

        Returns:
            Cached response or None if miss/expired
        """
        key = self._make_key(model, prompt)

        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache HIT — key=%s...  (hits=%d)", key[:12], self._hits)
                    return value
                else:
                    # Expired — remove
                    del self._cache[key]

        self._misses += 1
        return None

    def put(self, model: str, prompt: str, response: Any) -> None:
        """Store a response in the cache."""
        key = self._make_key(model, prompt)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (response, time.time())

            # Evict oldest if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                logger.debug("Cache eviction — key=%s...", evicted_key[:12])

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
        logger.info("Cache cleared")

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "N/A",
        }


# Singleton cache instance
llm_cache = LLMCache(max_size=500, ttl_seconds=3600)
