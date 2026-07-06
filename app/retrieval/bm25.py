"""Lexical retrieval via Postgres full-text search (BM25-style).

Uses the generated `tsv` column plus `ts_rank_cd` for ranking. `websearch_to_tsquery`
gives Google-like query syntax and is forgiving of arbitrary user input.

This is the "BM25" arm of the hybrid retriever. Postgres FTS is not literally
Okapi BM25, but it is a lexical, term-frequency-based ranker that plays the same
role and lives in the same store as the vectors — no separate index to maintain.
"""
from __future__ import annotations

from app.db import get_conn
from app.retrieval.types import RetrievedChunk

_BM25_SQL = """
SELECT id, source, source_type, chunk_index, content, metadata,
       ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS rank
FROM chunks
WHERE tsv @@ websearch_to_tsquery('english', %s)
ORDER BY rank DESC
LIMIT %s
"""


def lexical_search(query: str, limit: int = 10) -> list[RetrievedChunk]:
    """Return the `limit` best full-text matches for `query`."""
    results: list[RetrievedChunk] = []
    with get_conn() as conn:
        rows = conn.execute(_BM25_SQL, (query, query, limit)).fetchall()

    for rank, row in enumerate(rows, start=1):
        rid, source, source_type, chunk_index, content, metadata, ts_rank = row
        results.append(
            RetrievedChunk(
                id=rid,
                source=source,
                source_type=source_type,
                chunk_index=chunk_index,
                content=content,
                metadata=metadata or {},
                score=float(ts_rank),
                lexical_rank=rank,
            )
        )
    return results
