"""Hybrid retriever: dense vectors + lexical FTS fused with RRF.

    from app.retrieval.hybrid import HybridRetriever
    chunks = HybridRetriever().retrieve("what is the refund policy?")

Each returned `RetrievedChunk` carries its source metadata so the graph can
produce grounded, citeable answers.
"""
from __future__ import annotations

from app.config import settings
from app.retrieval.bm25 import lexical_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk
from app.retrieval.vector_store import vector_search


class HybridRetriever:
    def __init__(
        self,
        top_k: int | None = None,
        rrf_k: int | None = None,
        candidate_multiplier: int = 4,
    ) -> None:
        self.top_k = top_k or settings.retrieval_top_k
        self.rrf_k = rrf_k or settings.rrf_k
        # Pull more candidates per arm than we finally return, so fusion has
        # room to promote items that rank well in either modality.
        self.candidate_k = self.top_k * candidate_multiplier

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        vector_hits = vector_search(query, limit=self.candidate_k)
        lexical_hits = lexical_search(query, limit=self.candidate_k)

        by_id: dict[int, RetrievedChunk] = {}
        for hit in vector_hits:
            by_id[hit.id] = hit
        for hit in lexical_hits:
            if hit.id in by_id:
                # Merge the lexical rank onto the existing (vector) record.
                by_id[hit.id].lexical_rank = hit.lexical_rank
            else:
                by_id[hit.id] = hit

        fused = reciprocal_rank_fusion(
            {
                "vector": [h.id for h in vector_hits],
                "lexical": [h.id for h in lexical_hits],
            },
            k=self.rrf_k,
            top_n=self.top_k,
        )

        results: list[RetrievedChunk] = []
        for fr in fused:
            chunk = by_id[fr.key]
            chunk.score = fr.score
            chunk.vector_rank = fr.ranks.get("vector")
            chunk.lexical_rank = fr.ranks.get("lexical")
            results.append(chunk)
        return results
