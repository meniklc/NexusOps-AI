"""Shared LangGraph state.

`history` uses an additive reducer so conversation turns accumulate across
invocations under the same checkpointer thread_id (demonstrating persistent
multi-turn state). All other keys are last-write-wins per turn.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Intent = Literal["qa", "sql", "action"]


class AgentState(TypedDict, total=False):
    # --- input ---
    query: str
    thread_id: str

    # --- security guardrail ---
    security: dict[str, Any]
    blocked: bool

    # --- routing ---
    intent: Intent

    # --- Q&A path ---
    retrieved: list[dict[str, Any]]     # {id, source, source_type, chunk_index, content, score}
    citations: list[dict[str, Any]]

    # --- SQL path ---
    sql_result: dict[str, Any]

    # --- action path ---
    action_request: dict[str, Any]
    action_result: dict[str, Any]
    approved: bool

    # --- output ---
    draft: str
    answer: str
    critique: dict[str, Any]

    # --- persistent conversation history (additive) ---
    history: Annotated[list[dict[str, str]], operator.add]
