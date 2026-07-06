"""FastAPI application exposing the LangGraph assistant.

Endpoints
---------
GET  /health          liveness + provider/db status
POST /chat            run a turn (non-streaming) -> TurnResponse
POST /chat/stream     run a turn, streaming SSE events (status/sources/token/interrupt/final)
POST /approve         resume a paused turn (non-streaming)
POST /approve/stream  resume a paused turn, streaming SSE
GET  /tickets         list mock tickets created via the action tool
GET  /threads         list past conversation threads (preview + timestamps)
GET  /threads/{id}/messages   full message history for a thread (chat schema)

Human-in-the-loop: when /chat or /chat/stream returns status
`pending_approval` (the graph hit the `create_ticket` interrupt), call
/approve (or /approve/stream) with the same thread_id and approved=true/false.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.api.schemas import ApprovalRequest, ChatRequest, HealthResponse, TurnResponse
from app.config import settings
from app.db import healthcheck
from app.graph import service
from app.tools.ticket_tool import list_tickets

app = FastAPI(title="Enterprise RAG + Agentic Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Export LangSmith env so LangChain/LangGraph auto-trace runs when enabled.
    settings.apply_langsmith_env()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        database="up" if healthcheck() else "down",
    )


@app.post("/chat", response_model=TurnResponse)
def chat(req: ChatRequest) -> TurnResponse:
    result = service.run_turn(req.message, req.thread_id)
    return TurnResponse(**result)


@app.post("/approve", response_model=TurnResponse)
def approve(req: ApprovalRequest) -> TurnResponse:
    result = service.resume_turn(req.thread_id, req.approved, req.edits)
    return TurnResponse(**result)


@app.get("/tickets")
def tickets() -> dict:
    return {"tickets": list_tickets()}


@app.get("/threads")
def threads() -> dict:
    """List past conversation threads, most-recently-active first."""
    return {"threads": service.list_threads()}


@app.get("/threads/{thread_id}/messages")
def thread_messages(thread_id: str) -> dict:
    """Full message history for a thread, in the frontend's chat schema."""
    return {"thread_id": thread_id, "messages": service.get_history(thread_id)}


# --- SSE helpers -----------------------------------------------------------
def _to_sse(event: dict) -> dict:
    """Map a service event dict to an SSE frame (name + data)."""
    name = event.get("event", "message")
    if name == "token":
        return {"event": "token", "data": event.get("data", "")}
    payload = {k: v for k, v in event.items() if k != "event"}
    return {"event": name, "data": json.dumps(payload)}


async def _sse_stream(events) -> AsyncIterator[dict]:
    # The service generators are synchronous; iterate and adapt to SSE frames.
    for event in events:
        yield _to_sse(event)
    yield {"event": "done", "data": "{}"}


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    events = service.stream_turn(req.message, req.thread_id)
    return EventSourceResponse(_sse_stream(events))


@app.post("/approve/stream")
async def approve_stream(req: ApprovalRequest):
    events = service.stream_resume(req.thread_id, req.approved, req.edits)
    return EventSourceResponse(_sse_stream(events))
