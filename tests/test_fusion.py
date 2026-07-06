"""Offline tests for Reciprocal Rank Fusion."""
from __future__ import annotations

from app.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_ranks_first():
    fused = reciprocal_rank_fusion(
        {"vector": ["a", "b", "c"], "lexical": ["b", "d", "a"]},
        k=60,
    )
    # 'a' and 'b' appear in both lists; they should outrank singletons.
    top_keys = [r.key for r in fused[:2]]
    assert set(top_keys) == {"a", "b"}
    assert set(fused[0].ranks) == {"vector", "lexical"}


def test_scores_follow_rrf_formula():
    fused = reciprocal_rank_fusion({"only": ["x", "y"]}, k=10)
    by_key = {r.key: r.score for r in fused}
    assert abs(by_key["x"] - 1 / 11) < 1e-12
    assert abs(by_key["y"] - 1 / 12) < 1e-12
    assert by_key["x"] > by_key["y"]


def test_top_n_truncation():
    fused = reciprocal_rank_fusion(
        {"a": ["1", "2", "3", "4", "5"]}, k=60, top_n=3
    )
    assert len(fused) == 3


def test_deterministic_order():
    lists = {"v": ["a", "b", "c"], "l": ["c", "b", "a"]}
    first = [r.key for r in reciprocal_rank_fusion(lists)]
    second = [r.key for r in reciprocal_rank_fusion(lists)]
    assert first == second


def test_empty_input():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"v": []}) == []
