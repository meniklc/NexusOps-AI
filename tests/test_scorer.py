"""Offline tests for the eval scorers + an end-to-end score using the Fake LLM."""
from __future__ import annotations

from app.eval.scorer import (
    answer_relevance,
    citation_rate,
    context_recall,
    faithfulness,
    score_item,
)
from app.graph.prompts import SYNTHESIS_SYSTEM, synthesis_user_prompt
from app.llm.base import ChatMessage
from app.llm.fake_provider import FakeLLMProvider


def test_context_recall():
    ctx = ["Employees get 20 days of paid vacation per year."]
    assert context_recall(ctx, ["20", "vacation"]) == 1.0
    assert context_recall(ctx, ["20", "sick"]) == 0.5
    assert context_recall(ctx, []) == 1.0


def test_faithfulness_bounds():
    ctx = ["The Widget Pro costs 19.99 dollars."]
    grounded = faithfulness("Widget Pro costs 19.99", ctx)
    hallucinated = faithfulness("The moon is made of cheese entirely", ctx)
    assert grounded > hallucinated
    assert 0.0 <= grounded <= 1.0


def test_answer_relevance_and_citation():
    assert answer_relevance("Vacation is 20 days", ["20", "vacation"]) == 1.0
    assert citation_rate("Answer grounded in sources [1].") == 1.0
    assert citation_rate("No citations here.") == 0.0


def test_end_to_end_score_with_fake_llm():
    # Simulate what the graph does for a Q&A turn, minus the DB retrieval.
    retrieved = [
        {"source": "handbook.md", "content": "Full-time employees accrue 20 days of paid vacation per year."},
        {"source": "handbook.md", "content": "Up to 5 unused vacation days may be carried over."},
    ]
    llm = FakeLLMProvider()
    answer = llm.chat(
        [
            ChatMessage("system", SYNTHESIS_SYSTEM),
            ChatMessage("user", synthesis_user_prompt("How many vacation days?", retrieved)),
        ]
    )
    scores = score_item(answer, [c["content"] for c in retrieved], ["20", "vacation"])
    # Fake synthesis echoes grounded context with a [1] citation.
    assert scores.citation_rate == 1.0
    assert scores.context_recall == 1.0
    assert scores.faithfulness > 0.5
    assert 0.0 <= scores.mean() <= 1.0
