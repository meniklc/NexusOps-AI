"""Dense vector retrieval over pgvector (cosine distance)."""
from __future__ import annotations

from app.db import get_conn
from app.llm.factory import get_embedder
from app.retrieval.types import RetrievedChunk

# `<=>` is pgvector's cosine-distance operator (smaller = more similar).
_VECTOR_SQL = """
SELECT id, source, source_type, chunk_index, content, metadata,
       1 - (embedding <=> %s::vector) AS similarity
FROM chunks
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s
"""


def vector_search(query: str, limit: int = 10) -> list[RetrievedChunk]:
    """Embed the query and return the `limit` nearest chunks by cosine similarity."""
    embedder = get_embedder()
    qvec = embedder.embed_query(query)

    results: list[RetrievedChunk] = []
    with get_conn() as conn:
        rows = conn.execute(_VECTOR_SQL, (qvec, qvec, limit)).fetchall()

    for rank, row in enumerate(rows, start=1):
        rid, source, source_type, chunk_index, content, metadata, similarity = row
        results.append(
            RetrievedChunk(
                id=rid,
                source=source,
                source_type=source_type,
                chunk_index=chunk_index,
                content=content,
                metadata=metadata or {},
                score=float(similarity),
                vector_rank=rank,
            )
        )
    return results
