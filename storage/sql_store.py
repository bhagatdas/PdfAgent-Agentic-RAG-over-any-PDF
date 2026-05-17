"""
SQLite-based table memory for storing extracted PDF tables.
Supports schema-aware SQL querying with read-only enforcement at query time.
"""

import logging
import sqlite3
import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class SQLTableStore:
    """
    Manages structured table data in SQLite.
    Tables are created during preprocessing; queries are read-only at runtime.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.sqlite_table_db
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database and metadata table."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _table_metadata (
                    table_name TEXT PRIMARY KEY,
                    source_document TEXT,
                    source_page INTEGER,
                    description TEXT,
                    column_info TEXT,
                    row_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        logger.info("SQLite table store initialized — db=%s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _sanitize_table_name(name: str) -> str:
        """Sanitize a table name for safe SQL usage."""
        # Replace non-alphanumeric chars with underscore, lowercase
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "t_" + sanitized
        return sanitized[:64]  # Limit length

    @staticmethod
    def _sanitize_column_name(name: str) -> str:
        """Sanitize a column name for safe SQL usage."""
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", str(name)).lower().strip("_")
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "col_" + sanitized
        return sanitized[:64]

    def create_table(
        self,
        table_name: str,
        df: pd.DataFrame,
        source_document: str = "",
        source_page: int = 0,
        description: str = "",
    ) -> str:
        """
        Create a table from a pandas DataFrame.

        Args:
            table_name: Desired table name (will be sanitized)
            df: Data to store
            source_document: Source PDF filename
            source_page: Page number where table was found
            description: Human-readable description

        Returns:
            Actual sanitized table name used
        """
        safe_name = self._sanitize_table_name(table_name)

        # Sanitize column names
        df = df.copy()
        df.columns = [self._sanitize_column_name(c) for c in df.columns]

        # Remove completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")

        if df.empty:
            logger.warning("Skipping empty table: %s", safe_name)
            return safe_name

        with self._connect() as conn:
            # Store the data
            df.to_sql(safe_name, conn, if_exists="replace", index=False)

            # Store metadata
            col_info = ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])
            conn.execute("""
                INSERT OR REPLACE INTO _table_metadata
                (table_name, source_document, source_page, description, column_info, row_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (safe_name, source_document, source_page, description, col_info, len(df)))
            conn.commit()

        logger.info(
            "Table created — name=%s, rows=%d, cols=%d, source=%s p.%d",
            safe_name, len(df), len(df.columns), source_document, source_page,
        )
        return safe_name

    def execute_query(self, sql: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a DataFrame.

        Args:
            sql: SQL query string (SELECT only)

        Returns:
            Query results as DataFrame

        Raises:
            ValueError: If query is not a SELECT statement
            sqlite3.Error: On SQL execution error
        """
        # Enforce read-only: only SELECT statements allowed
        sql_stripped = sql.strip().upper()
        if not sql_stripped.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed at query time")

        # Block dangerous statements
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
        for keyword in forbidden:
            if keyword in sql_stripped:
                raise ValueError(f"Forbidden SQL keyword detected: {keyword}")

        with self._connect() as conn:
            try:
                df = pd.read_sql_query(sql, conn)
                logger.info("SQL executed — rows=%d, cols=%d", len(df), len(df.columns))
                return df
            except Exception as e:
                logger.error("SQL execution error: %s — query: %s", e, sql)
                raise

    def safe_execute(self, sql: str) -> tuple[bool, Union[pd.DataFrame, str]]:
        """
        Execute a query with error handling.

        Returns:
            Tuple of (success: bool, result_or_error: DataFrame | str)
        """
        try:
            result = self.execute_query(sql)
            return True, result
        except Exception as e:
            return False, str(e)

    def get_table_names(self) -> list[str]:
        """List all user-created tables (excluding metadata table)."""
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE '_%'
                ORDER BY name
            """)
            return [row[0] for row in cursor.fetchall()]

    def get_schema(self, table_name: str) -> dict:
        """
        Get schema information for a single table.

        Returns:
            Dict with table_name, columns (name, type), row_count, metadata
        """
        safe_name = self._sanitize_table_name(table_name)

        with self._connect() as conn:
            # Get column info from PRAGMA
            cursor = conn.execute(f"PRAGMA table_info({safe_name})")
            columns = [
                {"name": row[1], "type": row[2], "nullable": not row[3]}
                for row in cursor.fetchall()
            ]

            # Get metadata
            meta_cursor = conn.execute(
                "SELECT * FROM _table_metadata WHERE table_name = ?",
                (safe_name,),
            )
            meta_row = meta_cursor.fetchone()

            # Get sample data (first 3 rows)
            try:
                sample_df = pd.read_sql_query(
                    f"SELECT * FROM {safe_name} LIMIT 3", conn
                )
                sample_data = sample_df.values.tolist()
            except Exception:
                sample_data = []

        return {
            "table_name": safe_name,
            "columns": columns,
            "row_count": meta_row["row_count"] if meta_row else 0,
            "source_document": meta_row["source_document"] if meta_row else "",
            "source_page": meta_row["source_page"] if meta_row else 0,
            "description": meta_row["description"] if meta_row else "",
            "sample_data": sample_data,
        }

    def get_all_schemas(self) -> list[dict]:
        """Get schema information for all tables."""
        return [self.get_schema(name) for name in self.get_table_names()]

    def drop_all_tables(self) -> None:
        """Drop all user-created tables. Used during re-ingestion."""
        table_names = self.get_table_names()
        with self._connect() as conn:
            for name in table_names:
                conn.execute(f"DROP TABLE IF EXISTS {name}")
            conn.execute("DELETE FROM _table_metadata")
            conn.commit()
        logger.info("Dropped %d tables", len(table_names))


# Singleton instance
sql_store = SQLTableStore()
