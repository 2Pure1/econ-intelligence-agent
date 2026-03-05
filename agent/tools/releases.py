"""
agent/tools/releases.py
-----------------------
Retrieves recent economic data releases from the streaming.releases_enriched table.
"""

import os
from typing import Any
from sqlalchemy import create_engine, text
from loguru import logger

def get_recent_releases(
    limit: int = 10,
    source: str = "all",
    high_impact_only: bool = False
) -> str:
    """
    Query the database for recent economic releases.
    """
    pg_url = os.environ.get("POSTGRES_URL")
    if not pg_url:
        # Fallback to individual components
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ.get("POSTGRES_DB", "postgres")
        user = os.environ.get("POSTGRES_USER", "postgres")
        pw = os.environ.get("POSTGRES_PASSWORD", "")
        pg_url = f"postgresql://{user}:{pw}@{host}:{port}/{db}"

    try:
        engine = create_engine(pg_url)
        
        query = "SELECT event_name, release_date, actual, prior, surprise_pct, impact FROM streaming.releases_enriched"
        conditions = []
        
        if source != "all":
            conditions.append(f"source = :source")
        if high_impact_only:
            conditions.append("impact = 'HIGH'")
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY release_date DESC LIMIT :limit"
        
        with engine.connect() as conn:
            result = conn.execute(text(query), {"limit": limit, "source": source})
            rows = result.fetchall()
            
            if not rows:
                return "No recent releases found in the streaming database."
            
            # Format as a compact table string
            output = [f"{'Event':<30} | {'Date':<12} | {'Actual':<10} | {'Impact':<10}"]
            output.append("-" * 70)
            for row in rows:
                output.append(f"{str(row[0]):<30} | {str(row[1]):<12} | {str(row[2]):<10} | {str(row[5]):<10}")
            
            return "\n".join(output)

    except Exception as e:
        logger.error(f"Failed to fetch releases: {e}")
        return f"Error fetching releases: {str(e)}"
