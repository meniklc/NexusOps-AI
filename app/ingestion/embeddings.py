"""Batch embedding helper over the configured EmbeddingProvider."""
from __future__ import annotations

from typing import Sequence

from app.llm.base import EmbeddingProvider


def embed_in_batches(
    embedder: EmbeddingProvider, texts: Sequence[str], batch_size: int = 64
) -> list[list[float]]:
    """Embed `texts` in batches to bound request size / memory."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        vectors.extend(embedder.embed(batch))
    return vectors
