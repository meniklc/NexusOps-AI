"""End-to-end ingestion: load -> split -> embed -> store in pgvector.

    from app.ingestion.pipeline import ingest_path
    ingest_path("sample_docs")
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from psycopg.types.json import Jsonb

from app.config import settings
from app.db import get_conn
from app.ingestion.embeddings import embed_in_batches
from app.ingestion.loaders import Document, load_path
from app.ingestion.splitter import RecursiveCharacterTextSplitter
from app.llm.factory import get_embedder

_INSERT_SQL = """
INSERT INTO chunks (source, source_type, chunk_index, content, metadata, embedding)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (source, chunk_index)
DO UPDATE SET content   = EXCLUDED.content,
              metadata  = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding,
              source_type = EXCLUDED.source_type
"""


@dataclass
class IngestStats:
    documents: int
    chunks: int
    sources: list[str]

    def as_dict(self) -> dict:
        return {"documents": self.documents, "chunks": self.chunks, "sources": self.sources}


def _chunk_document(doc: Document, splitter: RecursiveCharacterTextSplitter) -> list[dict]:
    rows = []
    for idx, text in enumerate(splitter.split_text(doc.content)):
        rows.append(
            {
                "source": doc.source,
                "source_type": doc.source_type,
                "chunk_index": idx,
                "content": text,
                "metadata": {**doc.metadata, "chunk_index": idx},
            }
        )
    return rows


def ingest_path(path: str | os.PathLike) -> IngestStats:
    """Ingest a file or directory into the `chunks` table."""
    docs = load_path(path)
    if not docs:
        return IngestStats(documents=0, chunks=0, sources=[])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    embedder = get_embedder()

    all_rows: list[dict] = []
    for doc in docs:
        all_rows.extend(_chunk_document(doc, splitter))

    vectors = embed_in_batches(embedder, [r["content"] for r in all_rows])

    params = [
        (
            r["source"],
            r["source_type"],
            r["chunk_index"],
            r["content"],
            Jsonb(r["metadata"]),
            vec,
        )
        for r, vec in zip(all_rows, vectors)
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, params)
        conn.commit()

    return IngestStats(
        documents=len(docs),
        chunks=len(all_rows),
        sources=sorted({d.source for d in docs}),
    )
