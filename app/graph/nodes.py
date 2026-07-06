"""LangGraph node implementations.

Nodes are plain functions: they receive the current `AgentState` and return a
partial dict of updates. Provider-agnostic (they call `get_llm()` /
`HybridRetriever`), so the same graph runs offline (fake) or on OpenAI.
"""
from __future__ import annotations

import re

from langgraph.types import interrupt

from app.graph.critique import parse_critique
from app.graph.prompts import (
    CRITIQUE_SYSTEM,
    SYNTHESIS_SYSTEM,
    critique_user_prompt,
    synthesis_user_prompt,
)
from app.graph.router import classify_intent
from app.graph.state import AgentState
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.retrieval.hybrid import HybridRetriever
from app.security.injection import detect_injection
from app.tools.sql_tool import run_sql_agent
from app.tools.ticket_tool import TicketRequest, create_ticket

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


# --- guardrail -------------------------------------------------------------
def guard(state: AgentState) -> dict:
    verdict = detect_injection(state.get("query", ""))
    update: dict = {"security": verdict.as_dict()}
    if verdict.is_suspicious:
        update["blocked"] = True
        update["answer"] = (
            "I can't help with that request — it looks like an attempt to override "
            "my instructions. If this was a genuine question, please rephrase it."
        )
        update["history"] = [{"role": "assistant", "content": update["answer"]}]
    else:
        update["blocked"] = False
    return update


def route_after_guard(state: AgentState) -> str:
    return "blocked" if state.get("blocked") else "classify_intent"


# --- classification --------------------------------------------------------
def classify_intent_node(state: AgentState) -> dict:
    return {"intent": classify_intent(state.get("query", ""))}


# --- Q&A retrieval ---------------------------------------------------------
def retrieve_node(state: AgentState) -> dict:
    chunks = _get_retriever().retrieve(state.get("query", ""))
    retrieved = [
        {
            "id": c.id,
            "source": c.source,
            "source_type": c.source_type,
            "chunk_index": c.chunk_index,
            "content": c.content,
            "score": round(c.score, 6),
        }
        for c in chunks
    ]
    citations = [c.citation() for c in chunks]
    return {"retrieved": retrieved, "citations": citations}


# --- SQL agent -------------------------------------------------------------
def sql_agent_node(state: AgentState) -> dict:
    result = run_sql_agent(state.get("query", ""))
    return {"sql_result": result.as_dict()}


# --- action (human-in-the-loop) -------------------------------------------
_PRIORITY_RE = re.compile(r"\b(urgent|critical|emergency|asap|high|low)\b", re.I)


def parse_ticket_request(query: str, requester: str = "assistant-user") -> TicketRequest:
    priority = "normal"
    m = _PRIORITY_RE.search(query)
    if m:
        word = m.group(1).lower()
        priority = {"urgent": "urgent", "critical": "urgent", "emergency": "urgent",
                    "asap": "high", "high": "high", "low": "low"}.get(word, "normal")
    # Title: strip leading "create a ticket to/for/about ..." boilerplate.
    title = re.sub(
        r"^\s*(please\s+)?(create|open|file|raise|submit|log)\s+(a\s+|an\s+)?"
        r"(ticket|issue|case|incident|request)\s*(to|for|about|:)?\s*",
        "",
        query,
        flags=re.I,
    ).strip()
    title = (title or query).strip()
    if len(title) > 120:
        title = title[:117] + "..."
    return TicketRequest(title=title, description=query.strip(), priority=priority,
                         requester=requester)


def action_tool_node(state: AgentState) -> dict:
    """Prepare a ticket, PAUSE for human approval, then execute on resume.

    The `interrupt()` call checkpoints the graph and surfaces the approval
    payload to the caller. Execution resumes here when the API replays the
    thread with `Command(resume={"approved": ...})`.
    """
    request = parse_ticket_request(state.get("query", ""))
    request_dict = {
        "title": request.title,
        "description": request.description,
        "priority": request.priority,
        "requester": request.requester,
        "tags": request.tags,
    }

    # >>> pause here until a human approves/rejects <<<
    decision = interrupt(
        {
            "type": "approval_request",
            "action": "create_ticket",
            "request": request_dict,
            "message": f"Approve creating a '{request.priority}' ticket: \"{request.title}\"?",
        }
    )

    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    edits = decision.get("edits") if isinstance(decision, dict) else None

    if not approved:
        answer = "Okay — I did not create the ticket. Let me know if you'd like to change anything."
        return {
            "approved": False,
            "action_request": request_dict,
            "answer": answer,
            "history": [{"role": "assistant", "content": answer}],
        }

    # Apply any human edits to the request before executing.
    if isinstance(edits, dict):
        request_dict.update({k: v for k, v in edits.items() if k in request_dict})

    ticket = create_ticket(
        TicketRequest(
            title=request_dict["title"],
            description=request_dict.get("description", ""),
            priority=request_dict.get("priority", "normal"),
            requester=request_dict.get("requester", "assistant-user"),
            tags=request_dict.get("tags", []),
        )
    )
    answer = (
        f"Done — I created ticket {ticket['id']} "
        f"(priority: {ticket['priority']}): \"{ticket['title']}\"."
    )
    return {
        "approved": True,
        "action_request": request_dict,
        "action_result": ticket,
        "answer": answer,
        "history": [{"role": "assistant", "content": answer}],
    }


# --- response synthesis ----------------------------------------------------
def generate_response_node(state: AgentState) -> dict:
    intent = state.get("intent", "qa")
    llm = get_llm()

    if intent == "sql":
        sql = state.get("sql_result", {}) or {}
        draft = sql.get("summary", "I couldn't produce a result.")
        if sql.get("sql"):
            draft += f"\n\n_Query:_ `{sql['sql']}`"
        # History is finalized in self_critique (single place) to avoid dupes.
        return {"draft": draft, "answer": draft}

    # Q&A synthesis grounded in retrieved context.
    retrieved = state.get("retrieved", []) or []
    draft = llm.chat(
        [
            ChatMessage("system", SYNTHESIS_SYSTEM),
            ChatMessage("user", synthesis_user_prompt(state.get("query", ""), retrieved)),
        ]
    )
    return {"draft": draft, "answer": draft}


# --- self critique ---------------------------------------------------------
def self_critique_node(state: AgentState) -> dict:
    retrieved = state.get("retrieved", []) or []
    draft = state.get("draft", "") or state.get("answer", "")

    # Only meaningful for grounded Q&A; skip when there's no context.
    if not retrieved or not draft:
        return {
            "critique": {"skipped": True, "supported": True},
            "history": [{"role": "assistant", "content": state.get("answer", "")}],
        }

    llm = get_llm()
    raw = llm.chat(
        [
            ChatMessage("system", CRITIQUE_SYSTEM),
            ChatMessage("user", critique_user_prompt(draft, retrieved)),
        ]
    )
    verdict = parse_critique(raw)

    answer = state.get("answer", draft)
    if not verdict.get("supported", True) and verdict.get("revised_answer"):
        answer = verdict["revised_answer"]

    return {
        "critique": verdict,
        "answer": answer,
        "history": [{"role": "assistant", "content": answer}],
    }
