"""Turn-level service wrapping the compiled graph.

Provides the operations the API needs:
  - run_turn         : start a turn; may complete or pause for approval
  - resume_turn      : resume a paused turn with an approval decision
  - stream_turn      : same as run_turn but yields SSE-friendly events
  - stream_resume    : streaming resume

Interrupt detection reads the checkpointer snapshot: if the graph has a pending
task carrying an `interrupt` value, the turn is awaiting human approval.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Iterator

from langgraph.types import Command

from app.graph.build import get_graph


def new_thread_id() -> str:
    return uuid.uuid4().hex


# --- thread registry -------------------------------------------------------
# Lightweight in-memory index of conversations, powering the frontend's history
# sidebar. It intentionally shares the checkpointer's lifetime semantics: the
# graph uses an in-memory MemorySaver (see build.py), so threads live for the
# life of the process. No DB / schema change is involved.
_THREADS: dict[str, dict] = {}


def _touch_thread(thread_id: str, query: str) -> None:
    """Record/refresh a thread on each user turn (first message = preview)."""
    now = time.time()
    entry = _THREADS.get(thread_id)
    if entry is None:
        preview = " ".join((query or "").split())
        _THREADS[thread_id] = {
            "thread_id": thread_id,
            "preview": preview or "(empty)",
            "created_at": now,
            "updated_at": now,
            "turns": 1,
        }
    else:
        entry["updated_at"] = now
        entry["turns"] = entry.get("turns", 0) + 1


def _bump_thread(thread_id: str) -> None:
    """Refresh a thread's activity timestamp without counting a new turn."""
    entry = _THREADS.get(thread_id)
    if entry is not None:
        entry["updated_at"] = time.time()


def list_threads() -> list[dict]:
    """All known threads, most-recently-active first."""
    return sorted(_THREADS.values(), key=lambda e: e["updated_at"], reverse=True)


def get_history(thread_id: str) -> list[dict]:
    """Return a thread's messages in the frontend's chat schema.

    History is stored as `{role, content}` in the checkpointer state. Per-turn
    sources/metadata aren't persisted per message (state is last-write-wins), so
    historical assistant turns reload as text; the final assistant turn is
    enriched with the thread's latest citations/metadata from the snapshot.
    """
    snap = get_graph().get_state(_config(thread_id))
    values: dict = snap.values or {}
    history = values.get("history", []) or []

    messages: list[dict] = [
        {
            "role": m.get("role", "assistant"),
            "content": m.get("content", ""),
            "sources": [],
            "meta": {},
        }
        for m in history
    ]

    for msg in reversed(messages):
        if msg["role"] == "assistant":
            msg["sources"] = values.get("citations", []) or []
            msg["meta"] = {
                "intent": values.get("intent"),
                "security": values.get("security"),
                "critique": values.get("critique"),
                "sql_result": values.get("sql_result") or {},
            }
            break

    return messages


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _pending_interrupt(snapshot) -> Any | None:
    """Return the interrupt payload if the graph is paused, else None.

    If the graph has a pending task we treat the turn as awaiting approval even
    when the payload can't be introspected (defensive across LangGraph
    versions), falling back to a generic marker.
    """
    if not getattr(snapshot, "next", None):
        return None
    for task in getattr(snapshot, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            return getattr(intr, "value", intr)
    return {"type": "approval_request", "action": "pending"}


def _collect(thread_id: str) -> dict:
    graph = get_graph()
    snap = graph.get_state(_config(thread_id))
    values: dict = snap.values or {}
    interrupt_payload = _pending_interrupt(snap)

    if interrupt_payload is not None:
        status = "pending_approval"
    elif values.get("blocked"):
        status = "blocked"
    else:
        status = "complete"

    return {
        "thread_id": thread_id,
        "status": status,
        "intent": values.get("intent"),
        "answer": values.get("answer", ""),
        "citations": values.get("citations", []),
        "retrieved": values.get("retrieved", []),
        "sql_result": values.get("sql_result"),
        "security": values.get("security"),
        "critique": values.get("critique"),
        "action_result": values.get("action_result"),
        "interrupt": interrupt_payload,
    }


def _initial_state(query: str, thread_id: str) -> dict:
    return {
        "query": query,
        "thread_id": thread_id,
        "history": [{"role": "user", "content": query}],
    }


# --- non-streaming ---------------------------------------------------------
def run_turn(query: str, thread_id: str | None = None) -> dict:
    thread_id = thread_id or new_thread_id()
    _touch_thread(thread_id, query)
    get_graph().invoke(_initial_state(query, thread_id), _config(thread_id))
    return _collect(thread_id)


def resume_turn(thread_id: str, approved: bool, edits: dict | None = None) -> dict:
    _bump_thread(thread_id)
    get_graph().invoke(
        Command(resume={"approved": approved, "edits": edits}),
        _config(thread_id),
    )
    return _collect(thread_id)


# --- streaming -------------------------------------------------------------
def _stream(inp: Any, thread_id: str) -> Iterator[dict]:
    """Run the graph, yielding node-progress events then answer tokens/interrupt."""
    graph = get_graph()
    config = _config(thread_id)
    captured_interrupt: Any | None = None

    for chunk in graph.stream(inp, config, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                # updates-mode surfaces interrupts as a tuple of Interrupt objects.
                try:
                    captured_interrupt = update[0].value
                except (IndexError, AttributeError, TypeError):
                    captured_interrupt = update
                continue
            event: dict = {"event": "status", "node": node_name}
            if isinstance(update, dict):
                if "security" in update:
                    event["security"] = update["security"]
                if "intent" in update:
                    event["intent"] = update["intent"]
                if "citations" in update:
                    event["citations"] = update["citations"]
            yield event

    result = _collect(thread_id)

    if result["status"] == "pending_approval" or captured_interrupt is not None:
        payload = captured_interrupt if captured_interrupt is not None else result["interrupt"]
        yield {"event": "interrupt", "data": payload, "thread_id": thread_id}
        return

    if result["citations"]:
        yield {"event": "sources", "data": result["citations"]}

    for token in re.findall(r"\S+\s*", result["answer"] or ""):
        yield {"event": "token", "data": token}

    yield {"event": "final", "data": result}


def stream_turn(query: str, thread_id: str | None = None) -> Iterator[dict]:
    thread_id = thread_id or new_thread_id()
    _touch_thread(thread_id, query)
    yield {"event": "thread", "thread_id": thread_id}
    yield from _stream(_initial_state(query, thread_id), thread_id)


def stream_resume(thread_id: str, approved: bool, edits: dict | None = None) -> Iterator[dict]:
    _bump_thread(thread_id)
    yield {"event": "thread", "thread_id": thread_id}
    yield from _stream(Command(resume={"approved": approved, "edits": edits}), thread_id)
