"""Offline tests for tools: SQL guard, ticket tool, web search stub."""
from __future__ import annotations

import os
import tempfile

from app.tools.sql_guard import UnsafeSQLError, ensure_limit, is_safe_select, validate_select
from app.tools.web_search import web_search


# --- SQL guard -------------------------------------------------------------
def test_guard_accepts_plain_select():
    assert is_safe_select("SELECT * FROM orders WHERE status = 'delivered'")
    assert is_safe_select("WITH x AS (SELECT 1) SELECT * FROM x")


def test_guard_rejects_mutations():
    for bad in [
        "DELETE FROM orders",
        "UPDATE orders SET status='x'",
        "DROP TABLE orders",
        "INSERT INTO orders VALUES (1)",
        "SELECT 1; DROP TABLE orders",
        "SELECT * FROM orders -- comment",
        "TRUNCATE orders",
    ]:
        assert not is_safe_select(bad), bad


def test_guard_raises_typed_error():
    try:
        validate_select("DELETE FROM orders")
    except UnsafeSQLError:
        return
    raise AssertionError("expected UnsafeSQLError")


def test_ensure_limit_adds_when_missing():
    assert ensure_limit("SELECT * FROM orders", 50).endswith("LIMIT 50")
    assert ensure_limit("SELECT * FROM orders LIMIT 3", 50).endswith("LIMIT 3")


# --- Ticket tool -----------------------------------------------------------
def test_ticket_create_and_list(monkeypatch=None):
    with tempfile.TemporaryDirectory() as d:
        os.environ["TICKET_DB_PATH"] = os.path.join(d, "tickets.json")
        # Import inside so the patched env path is picked up.
        import importlib

        import app.tools.ticket_tool as tt
        importlib.reload(tt)

        t1 = tt.create_ticket(tt.TicketRequest(title="Printer broken", priority="HIGH"))
        t2 = tt.create_ticket(tt.TicketRequest(title="VPN down"))
        assert t1["id"] == "TCK-0001"
        assert t1["priority"] == "high"
        assert t1["status"] == "open"
        assert t2["id"] == "TCK-0002"
        assert "_seq" not in t1
        assert len(tt.list_tickets()) == 2
    os.environ.pop("TICKET_DB_PATH", None)


def test_ticket_requires_title():
    with tempfile.TemporaryDirectory() as d:
        os.environ["TICKET_DB_PATH"] = os.path.join(d, "tickets.json")
        import importlib

        import app.tools.ticket_tool as tt
        importlib.reload(tt)
        try:
            tt.create_ticket(tt.TicketRequest(title="   "))
        except ValueError:
            return
        finally:
            os.environ.pop("TICKET_DB_PATH", None)
    raise AssertionError("expected ValueError for empty title")


# --- Web search stub -------------------------------------------------------
def test_web_search_stub_deterministic():
    a = web_search("langgraph interrupts", k=3)
    b = web_search("langgraph interrupts", k=3)
    assert len(a) == 3
    assert [r.as_dict() for r in a] == [r.as_dict() for r in b]
