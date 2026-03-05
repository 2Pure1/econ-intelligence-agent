"""
rag_search.py
-------------
Tool 3: Semantic search over Fed/IMF/FOMC document corpus.

Supports three vector DB backends, selected via VECTOR_DB env var:
  - "chroma"   — local ChromaDB (default, no infra)
  - "pgvector" — reuses existing PostgreSQL with pgvector extension
  - "pinecone" — managed cloud (set PINECONE_API_KEY)

Documents are embedded with voyage-3 (Anthropic's recommended embedder)
or text-embedding-3-small as fallback.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from loguru import logger

VECTOR_BACKEND = os.environ.get("VECTOR_DB", "chroma").lower()
COLLECTION_NAME = "econ_documents"
EMBED_MODEL = "voyage-3"


# ── Embedding function ────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts using Voyage AI (Anthropic's recommended embedder)."""
    try:
        import voyageai
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        result = client.embed(texts, model=EMBED_MODEL, input_type="query")
        return result.embeddings
    except Exception:
        # Fallback: OpenAI embeddings
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            resp = client.embeddings.create(input=texts, model="text-embedding-3-small")
            return [r.embedding for r in resp.data]
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise


# ── Chroma backend ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_chroma_collection():
    import chromadb
    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _search_chroma(query: str, k: int, source_filter: str) -> list[dict]:
    collection = _get_chroma_collection()
    embeddings = embed_texts([query])

    where = None
    if source_filter and source_filter != "all":
        where = {"source_type": {"$eq": source_filter}}

    results = collection.query(
        query_embeddings=embeddings,
        n_results=min(k, collection.count() or k),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":       doc,
            "source":     meta.get("source", "unknown"),
            "title":      meta.get("title", ""),
            "date":       meta.get("date", ""),
            "source_type":meta.get("source_type", ""),
            "score":      round(1 - dist, 4),   # cosine similarity
        })
    return hits


# ── pgvector backend ───────────────────────────────────────────────────────────
def _search_pgvector(query: str, k: int, source_filter: str) -> list[dict]:
    import psycopg2
    import json

    conn = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )

    embeddings = embed_texts([query])
    vec = embeddings[0]
    vec_str = f"[{','.join(map(str, vec))}]"

    source_clause = ""
    params: list = [vec_str, k]
    if source_filter and source_filter != "all":
        source_clause = "AND source_type = %s"
        params.insert(1, source_filter)

    cur = conn.cursor()
    cur.execute(f"""
        SELECT content, source, title, published_date, source_type,
               1 - (embedding <=> %s::vector) AS score
        FROM rag.document_chunks
        WHERE 1=1 {source_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, [vec_str] + (params[1:-1] if source_filter != "all" else []) + [vec_str, k])

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "text":        r[0],
            "source":      r[1],
            "title":       r[2],
            "date":        str(r[3]) if r[3] else "",
            "source_type": r[4],
            "score":       round(float(r[5]), 4),
        }
        for r in rows
    ]


# ── Pinecone backend ───────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ.get("PINECONE_INDEX", "econ-documents"))


def _search_pinecone(query: str, k: int, source_filter: str) -> list[dict]:
    index = _get_pinecone_index()
    embeddings = embed_texts([query])

    filter_dict = None
    if source_filter and source_filter != "all":
        filter_dict = {"source_type": {"$eq": source_filter}}

    results = index.query(
        vector=embeddings[0],
        top_k=k,
        include_metadata=True,
        filter=filter_dict,
    )

    return [
        {
            "text":        m.metadata.get("text", ""),
            "source":      m.metadata.get("source", ""),
            "title":       m.metadata.get("title", ""),
            "date":        m.metadata.get("date", ""),
            "source_type": m.metadata.get("source_type", ""),
            "score":       round(m.score, 4),
        }
        for m in results.matches
    ]


# ── Main search function ───────────────────────────────────────────────────────
def search_documents(
    query:         str,
    k:             int = 5,
    source_filter: str = "all",
) -> str:
    """
    Semantic search over the Fed/IMF/FOMC document corpus.
    Returns formatted results with source citations.
    """
    k = min(k, 10)
    logger.info(f"rag_search: query='{query[:60]}' k={k} source={source_filter} backend={VECTOR_BACKEND}")

    try:
        if VECTOR_BACKEND == "pgvector":
            hits = _search_pgvector(query, k, source_filter)
        elif VECTOR_BACKEND == "pinecone":
            hits = _search_pinecone(query, k, source_filter)
        else:
            hits = _search_chroma(query, k, source_filter)

        if not hits:
            return f"No documents found for query: '{query}'"

        parts = [f"**Document search results for:** '{query}'\n"]
        for i, h in enumerate(hits, 1):
            parts.append(
                f"---\n**[{i}] {h['title'] or h['source']}** "
                f"({h['source_type'].upper()}, {h['date']})\n"
                f"Score: {h['score']}\n\n"
                f"{h['text'][:800]}{'...' if len(h['text']) > 800 else ''}\n"
            )

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"rag_search failed: {e}")
        return (
            f"Document search unavailable: {e}\n"
            "Ensure documents have been ingested (run: python ingestion/ingest.py)"
        )
