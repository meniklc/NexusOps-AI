"""CLI to ingest documents into pgvector.

    python -m scripts.ingest sample_docs
    python -m scripts.ingest path/to/file.pdf another_dir
"""
from __future__ import annotations

import sys

from app.ingestion.pipeline import ingest_path


def main(argv: list[str]) -> int:
    paths = argv or ["sample_docs"]
    total_docs = total_chunks = 0
    for path in paths:
        print(f"Ingesting: {path}")
        stats = ingest_path(path)
        total_docs += stats.documents
        total_chunks += stats.chunks
        for src in stats.sources:
            print(f"  - {src}")
    print(f"\nDone. {total_docs} document(s), {total_chunks} chunk(s) ingested.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
