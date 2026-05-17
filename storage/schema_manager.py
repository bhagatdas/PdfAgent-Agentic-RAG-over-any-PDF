"""
Schema manager — generates a Databricks-style schema catalog
that the Table Agent uses to understand available tables and generate SQL.
"""

import json
import logging
from typing import Optional

from storage.sql_store import sql_store

logger = logging.getLogger(__name__)


class SchemaManager:
    """
    Generates and caches a human-readable schema catalog.
    Injected into Table Agent prompts so the LLM knows the database structure.
    """

    def __init__(self):
        self._cached_catalog: Optional[str] = None

    def generate_catalog(self, force_refresh: bool = False) -> str:
        """
        Generate a formatted schema catalog string for LLM consumption.

        Returns:
            Formatted string describing all tables, columns, types, and sample data
        """
        if self._cached_catalog and not force_refresh:
            return self._cached_catalog

        schemas = sql_store.get_all_schemas()

        if not schemas:
            self._cached_catalog = "NO TABLES AVAILABLE — No structured data has been ingested yet."
            return self._cached_catalog

        lines = [
            "DATABASE SCHEMA CATALOG",
            "=" * 60,
            f"Total tables: {len(schemas)}",
            "",
        ]

        for schema in schemas:
            lines.append(f"TABLE: {schema['table_name']}")
            lines.append(f"  Source: {schema['source_document']} (Page {schema['source_page']})")
            lines.append(f"  Description: {schema['description']}")
            lines.append(f"  Rows: {schema['row_count']}")
            lines.append("  Columns:")

            for col in schema["columns"]:
                nullable = " (nullable)" if col.get("nullable") else ""
                lines.append(f"    - {col['name']}: {col['type']}{nullable}")

            if schema.get("sample_data"):
                col_names = [c["name"] for c in schema["columns"]]
                lines.append("  Sample Data (first 3 rows):")
                lines.append(f"    {' | '.join(col_names)}")
                lines.append(f"    {'-' * (len(' | '.join(col_names)))}")
                for row in schema["sample_data"][:3]:
                    lines.append(f"    {' | '.join(str(v) for v in row)}")

            lines.append("")

        self._cached_catalog = "\n".join(lines)
        logger.info("Schema catalog generated — %d tables", len(schemas))
        return self._cached_catalog

    def get_catalog_json(self) -> list[dict]:
        """Get the schema catalog as a structured JSON-serializable list."""
        return sql_store.get_all_schemas()

    def invalidate_cache(self) -> None:
        """Invalidate the cached catalog (call after ingesting new tables)."""
        self._cached_catalog = None
        logger.info("Schema catalog cache invalidated")


# Singleton instance
schema_manager = SchemaManager()
