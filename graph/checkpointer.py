"""
LangGraph persistence setup — SqliteSaver for short-term state,
InMemoryStore for long-term cross-session memory.
"""

import logging
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

from config.settings import settings

logger = logging.getLogger(__name__)

_checkpointer: Optional[SqliteSaver] = None
_store: Optional[InMemoryStore] = None


def get_checkpointer() -> SqliteSaver:
    """
    Get the SqliteSaver checkpointer for per-thread state persistence.
    Each thread_id = one conversation session.
    State persists across app restarts.
    """
    global _checkpointer
    if _checkpointer is None:
        db_path = settings.checkpoint_db
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _checkpointer = SqliteSaver.from_conn_string(db_path)
        logger.info("Checkpointer initialized — db=%s", db_path)
    return _checkpointer


def get_memory_store() -> InMemoryStore:
    """
    Get the InMemoryStore for long-term cross-session memory.
    Namespaced by user_id. Stores past Q&A summaries.
    """
    global _store
    if _store is None:
        _store = InMemoryStore()
        logger.info("Long-term memory store initialized (InMemoryStore)")
    return _store
