"""
macro_db.py
-----------
Tool 1: Query fct_macro_indicators_monthly and streaming tables.

Safety:
  - Read-only connection (READONLY role or SET TRANSACTION READ ONLY)
  - SQL is validated to only allow SELECT statements
  - Results are capped at 200 rows and formatted as Markdown tables
"""

from __future__ import annotations

import os
import re

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text


def _get_engine():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    return create_engine(url, pool_pre_ping=True)


def _validate_sql(sql: str) -> None:
    """Reject anything that isn't a SELECT."""
    stripped = sql.strip().upper()
    if not stripped.startswith("SELECT") and not stripped.startswith("WITH"):
        raise ValueError("Only SELECT queries are permitted.")
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE", "GRANT"]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", stripped):
            raise ValueError(f"Forbidden SQL keyword: {kw}")


def query_macro_data(sql: str, description: str = "") -> str:
    """
    Execute a read-only SELECT against the macro data mart.
    Returns results formatted as a Markdown table (max 200 rows).
    """
    logger.info(f"macro_db: {description} | SQL: {sql[:120]}...")

    try:
        _validate_sql(sql)
        engine = _get_engine()

        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            df = pd.read_sql(text(sql), conn)

        if df.empty:
            return "Query returned no results."

        # Cap at 200 rows
        if len(df) > 200:
            df = df.head(200)
            suffix = f"\n\n_(Showing first 200 of {len(df)} rows)_"
        else:
            suffix = f"\n\n_({len(df)} rows returned)_"

        # Round floats for readability
        for col in df.select_dtypes(include="float"):
            df[col] = df[col].round(4)

        return df.to_markdown(index=False) + suffix

    except Exception as e:
        logger.error(f"macro_db query failed: {e}")
        return f"Query error: {e}\n\nSQL attempted:\n```sql\n{sql}\n```"
