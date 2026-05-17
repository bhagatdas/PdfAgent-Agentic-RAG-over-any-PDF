"""
Schema catalog generator — creates a Databricks Unity Catalog-style
schema document from SQLite tables with LLM-enhanced descriptions.
"""

import json
import logging
from pathlib import Path

from storage.sql_store import sql_store
from storage.schema_manager import schema_manager
from utils.llm import invoke_llm

logger = logging.getLogger(__name__)


def generate_enhanced_catalog(output_path: str = "./data/schema_catalog.json") -> str:
    """
    Generate an enhanced schema catalog with LLM-generated column descriptions.

    Args:
        output_path: Where to save the JSON catalog

    Returns:
        Formatted catalog string for LLM prompts
    """
    schemas = sql_store.get_all_schemas()

    if not schemas:
        logger.warning("No tables found — schema catalog is empty")
        return "NO TABLES AVAILABLE"

    enhanced_schemas = []
    for schema in schemas:
        # Enhance column descriptions using LLM
        enhanced_cols = _enhance_column_descriptions(schema)
        schema["columns"] = enhanced_cols
        enhanced_schemas.append(schema)

    # Save as JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_schemas, f, indent=2, default=str)

    logger.info("Schema catalog saved — tables=%d, path=%s", len(enhanced_schemas), output_path)

    # Invalidate and regenerate the cached catalog string
    schema_manager.invalidate_cache()
    return schema_manager.generate_catalog(force_refresh=True)


def _enhance_column_descriptions(schema: dict) -> list[dict]:
    """Use LLM to generate meaningful descriptions for table columns."""
    try:
        col_names = [c["name"] for c in schema["columns"]]
        sample = schema.get("sample_data", [])

        prompt = f"""Given this database table, provide a brief description for each column.
Table: {schema['table_name']}
Table description: {schema.get('description', 'N/A')}
Source: {schema.get('source_document', 'N/A')} (Page {schema.get('source_page', 'N/A')})
Columns: {', '.join(col_names)}
Sample data: {json.dumps(sample[:2], default=str)}

For each column, provide a description in this format:
column_name: description

Descriptions:"""

        response = invoke_llm(prompt, task_type="light")

        # Parse descriptions from response
        desc_map = {}
        for line in response.strip().split("\n"):
            if ":" in line:
                parts = line.split(":", 1)
                col = parts[0].strip().lower().replace(" ", "_")
                desc = parts[1].strip()
                desc_map[col] = desc

        # Apply descriptions to columns
        enhanced = []
        for col in schema["columns"]:
            col_copy = dict(col)
            col_copy["description"] = desc_map.get(col["name"], "")
            enhanced.append(col_copy)

        return enhanced

    except Exception as e:
        logger.warning("Failed to enhance column descriptions: %s", e)
        return schema["columns"]
