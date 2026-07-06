"""Offline tests for intent classification and prompt formatting."""
from __future__ import annotations

from app.graph.prompts import format_context, synthesis_user_prompt
from app.graph.router import classify_intent


def test_action_intent():
    for q in [
        "Please create a ticket for a broken printer",
        "Open an incident: the VPN is down",
        "I need to file a ticket about billing",
    ]:
        assert classify_intent(q) == "action", q


def test_sql_intent():
    for q in [
        "How many orders were delivered in EMEA?",
        "What is total revenue by region?",
        "count of refunded orders",
        "Which product had the most orders?",
        "What is the average order quantity per region?",
    ]:
        assert classify_intent(q) == "sql", q


def test_qa_intent_default():
    for q in [
        "What is the refund policy for hardware?",
        "How does parental leave work?",
        "Tell me about Support Gold",
    ]:
        assert classify_intent(q) == "qa", q


def test_aggregation_words_do_not_hijack_document_questions():
    """Regression: policy/HR questions phrased with "how many" must stay in qa,
    not get misrouted to the SQL agent (which only knows the orders table)."""
    for q in [
        "How many vacation days do full-time employees get, and how many can they carry over?",
        "What's the parental leave policy for a secondary caregiver?",
        "What's the remote work policy for part-time employees?",
        "How many days do I have to return a defective Widget Pro?",
        "What is the total warranty period on the Widget Pro?",
    ]:
        assert classify_intent(q) == "qa", q


def test_format_context_numbering_and_citation_prompt():
    retrieved = [
        {"source": "faq.md", "content": "Widget Pro costs $19.99."},
        {"source": "hb.md", "content": "20 days vacation."},
    ]
    ctx = format_context(retrieved)
    assert "[1] (source: faq.md)" in ctx
    assert "[2] (source: hb.md)" in ctx
    prompt = synthesis_user_prompt("How much is Widget Pro?", retrieved)
    assert "Question: How much is Widget Pro?" in prompt
    assert "Context:" in prompt
