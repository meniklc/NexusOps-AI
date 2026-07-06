"""Offline tests for the Fake provider and NL->SQL heuristic (no services)."""
from __future__ import annotations

import json

from app.llm.base import ChatMessage
from app.llm.fake_provider import FakeEmbeddingProvider, FakeLLMProvider
from app.tools.nl_sql import heuristic_nl_to_sql


def test_embeddings_deterministic_and_unit_norm():
    emb = FakeEmbeddingProvider(dim=256)
    a1 = emb.embed_query("the quick brown fox")
    a2 = emb.embed_query("the quick brown fox")
    assert a1 == a2, "same text must embed identically"
    assert len(a1) == 256
    norm = sum(x * x for x in a1) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embeddings_similarity_signal():
    emb = FakeEmbeddingProvider(dim=1024)
    q = emb.embed_query("refund policy for hardware orders")
    near = emb.embed_query("what is the refund policy on hardware")
    far = emb.embed_query("bananas are yellow tropical fruit")

    def cos(u, v):
        return sum(a * b for a, b in zip(u, v))

    assert cos(q, near) > cos(q, far)


def test_sql_task_returns_select():
    llm = FakeLLMProvider()
    msgs = [
        ChatMessage("system", "LLM-TASK: sql\nYou translate questions to SQL."),
        ChatMessage("user", "Question: how many delivered orders in EMEA?"),
    ]
    sql = llm.chat(msgs)
    assert sql.strip().upper().startswith("SELECT")
    assert "status = 'delivered'" in sql
    assert "region = 'EMEA'" in sql


def test_critique_task_returns_json_verdict():
    llm = FakeLLMProvider()
    msgs = [
        ChatMessage("system", "LLM-TASK: critique\nCheck the draft."),
        ChatMessage("user", "Draft: ...\nContext: ..."),
    ]
    verdict = json.loads(llm.chat(msgs))
    assert set(verdict) == {"supported", "issues", "revised_answer"}


def test_synthesis_cites_context():
    llm = FakeLLMProvider()
    user = (
        "Question: what is the vacation policy?\n\n"
        "Context:\n"
        "[1] Employees receive 20 days of paid vacation per year.\n"
        "[2] Vacation must be approved by a manager in advance."
    )
    msgs = [
        ChatMessage("system", "LLM-TASK: synthesize\nAnswer with citations."),
        ChatMessage("user", user),
    ]
    answer = llm.chat(msgs)
    assert "[1]" in answer


def test_stream_reassembles_to_full_text():
    llm = FakeLLMProvider()
    msgs = [
        ChatMessage("system", "LLM-TASK: synthesize\n"),
        ChatMessage("user", "Question: x\n\nContext:\n[1] Hello world facts here."),
    ]
    full = llm.chat(msgs)
    streamed = "".join(llm.stream(msgs))
    assert streamed == full


def test_nl_sql_group_by_region_revenue():
    sql = heuristic_nl_to_sql("what is total revenue by region?")
    assert "GROUP BY region" in sql
    assert "SUM(quantity * unit_price)" in sql


def test_nl_sql_is_read_only_default():
    sql = heuristic_nl_to_sql("show me something random")
    assert sql.strip().upper().startswith("SELECT")
