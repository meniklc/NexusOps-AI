"""Reciprocal Rank Fusion (RRF).

Combines several ranked result lists (here: lexical BM25/FTS and dense vector)
into one ranking without needing comparable score scales. Each item gets
``sum over lists of 1 / (k + rank)`` where ``rank`` is 1-based. Larger ``k``
flattens the contribution of top ranks. Dependency-free and unit-tested.

Reference: Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass
class FusionResult:
    key: Hashable
    score: float
    ranks: dict[str, int]  # list-name -> 1-based rank it appeared at


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[Hashable]],
    *,
    k: int = 60,
    top_n: int | None = None,
) -> list[FusionResult]:
    """Fuse named ranked lists of keys into a single ranking.

    Args:
        ranked_lists: mapping of list-name -> ordered keys (best first).
        k: RRF constant (typical default 60).
        top_n: optionally truncate the fused output.
    """
    scores: dict[Hashable, float] = {}
    ranks: dict[Hashable, dict[str, int]] = {}

    for list_name, keys in ranked_lists.items():
        for position, key in enumerate(keys):
            rank = position + 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(key, {})[list_name] = rank

    fused = [FusionResult(key=key, score=score, ranks=ranks[key]) for key, score in scores.items()]
    # Deterministic tie-break: higher score, then keys seen in more lists, then key repr.
    fused.sort(key=lambda r: (-r.score, -len(r.ranks), str(r.key)))
    return fused[:top_n] if top_n is not None else fused
