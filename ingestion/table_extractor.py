"""
Table extraction from PDFs — detects tables, converts to DataFrames,
stores in SQLite, and creates text representations for vector search.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF
import pandas as pd

from storage.sql_store import sql_store
from utils.llm import invoke_llm

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTable:
    """A table extracted from a PDF page."""
    page_number: int
    table_index: int
    document_name: str
    dataframe: pd.DataFrame
    sql_table_name: str = ""
    text_representation: str = ""
    description: str = ""


def extract_tables_from_pdf(pdf_path: str) -> list[ExtractedTable]:
    """
    Extract all tables from a PDF, store in SQLite, and generate text representations.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of ExtractedTable objects with SQL table names and text representations
    """
    path = Path(pdf_path)
    if not path.exists():
        logger.error("PDF not found: %s", pdf_path)
        return []

    doc_name = path.stem
    extracted_tables: list[ExtractedTable] = []

    try:
        doc = fitz.open(str(path))
        logger.info("Scanning for tables — file=%s, pages=%d", doc_name, len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]

            try:
                tables = page.find_tables()
            except Exception as e:
                logger.debug("No tables on page %d: %s", page_num + 1, e)
                continue

            for table_idx, table in enumerate(tables):
                try:
                    # Convert to pandas DataFrame
                    df = table.to_pandas()

                    if df.empty or df.shape[0] < 2:
                        continue

                    # Use first row as header if it looks like headers
                    if _looks_like_header(df):
                        df.columns = df.iloc[0].astype(str)
                        df = df.iloc[1:].reset_index(drop=True)

                    # Clean the DataFrame
                    df = _clean_dataframe(df)

                    if df.empty:
                        continue

                    # Generate table name
                    table_name = f"{doc_name}_p{page_num + 1}_t{table_idx + 1}"

                    # Generate description using LLM
                    description = _generate_table_description(df, doc_name, page_num + 1)

                    # Store in SQLite
                    actual_name = sql_store.create_table(
                        table_name=table_name,
                        df=df,
                        source_document=path.name,
                        source_page=page_num + 1,
                        description=description,
                    )

                    # Create text representation for vector search
                    text_repr = _create_text_representation(df, description, doc_name, page_num + 1)

                    extracted_tables.append(ExtractedTable(
                        page_number=page_num + 1,
                        table_index=table_idx,
                        document_name=doc_name,
                        dataframe=df,
                        sql_table_name=actual_name,
                        text_representation=text_repr,
                        description=description,
                    ))

                    logger.info(
                        "Table extracted — page=%d, index=%d, shape=%s, sql_name=%s",
                        page_num + 1, table_idx, df.shape, actual_name,
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to process table on page %d, index %d: %s",
                        page_num + 1, table_idx, e,
                    )

        doc.close()
        logger.info("Table extraction complete — doc=%s, tables=%d", doc_name, len(extracted_tables))

    except Exception as e:
        logger.error("Failed to extract tables from %s: %s", pdf_path, e)

    return extracted_tables


def _looks_like_header(df: pd.DataFrame) -> bool:
    """Check if the first row looks like column headers rather than data."""
    if df.empty:
        return False
    first_row = df.iloc[0]
    # Headers are usually strings, not numbers
    str_count = sum(1 for v in first_row if isinstance(v, str) and not _is_numeric(str(v)))
    return str_count > len(first_row) * 0.5


def _is_numeric(s: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(s.replace(",", "").replace("%", "").replace("$", ""))
        return True
    except (ValueError, AttributeError):
        return False


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a DataFrame for storage."""
    # Remove completely empty rows/columns
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Try to convert numeric-looking columns
    for col in df.columns:
        try:
            numeric_col = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("%", "").str.replace("$", ""),
                errors="coerce",
            )
            # If more than 50% of values converted successfully, use numeric
            if numeric_col.notna().sum() > len(df) * 0.5:
                df[col] = numeric_col
        except Exception:
            pass

    return df


def _generate_table_description(df: pd.DataFrame, doc_name: str, page: int) -> str:
    """Use LLM to generate a human-readable description of the table."""
    try:
        preview = df.head(3).to_string(index=False)
        columns = ", ".join(df.columns.tolist())

        prompt = f"""Describe this table from '{doc_name}' (page {page}) in one sentence.
Focus on what data it contains and its purpose.

Columns: {columns}
Sample data:
{preview}

Description (one sentence):"""

        description = invoke_llm(prompt, task_type="light")
        return description.strip()[:500]  # Limit length

    except Exception as e:
        logger.warning("Failed to generate table description: %s", e)
        return f"Table from {doc_name}, page {page} with columns: {', '.join(df.columns.tolist()[:10])}"


def _create_text_representation(
    df: pd.DataFrame, description: str, doc_name: str, page: int
) -> str:
    """
    Create a text representation of a table for vector search.
    This allows the retrieval agent to find table-related information
    even when routing to the Table Agent for SQL queries.
    """
    lines = [
        f"[TABLE from {doc_name}, Page {page}]",
        f"Description: {description}",
        f"Columns: {', '.join(df.columns.tolist())}",
        f"Number of rows: {len(df)}",
        "",
        "Data preview:",
        df.head(5).to_string(index=False),
    ]
    return "\n".join(lines)
