"""Shared retrieval data types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    id: int
    source: str
    source_type: str
    chunk_index: int
    content: str
    metadata: dict = field(default_factory=dict)
    # Fusion / ranking diagnostics (populated by the hybrid retriever).
    score: float = 0.0
    vector_rank: int | None = None
    lexical_rank: int | None = None

    def citation(self) -> dict:
        """Compact citation payload for the API / UI."""
        return {
            "id": self.id,
            "source": self.source,
            "source_type": self.source_type,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 6),
        }
