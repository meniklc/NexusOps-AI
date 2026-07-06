"""API request/response models."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    thread_id: Optional[str] = Field(
        None, description="Conversation thread id; omit to start a new conversation."
    )


class ApprovalRequest(BaseModel):
    thread_id: str = Field(..., description="Thread id of the paused turn")
    approved: bool = Field(..., description="Whether the human approves the pending action")
    edits: Optional[dict[str, Any]] = Field(
        None, description="Optional edits to apply to the action before executing."
    )


class TurnResponse(BaseModel):
    thread_id: str
    status: str                      # 'complete' | 'pending_approval' | 'blocked'
    intent: Optional[str] = None
    answer: str = ""
    citations: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    sql_result: Optional[dict[str, Any]] = None
    security: Optional[dict[str, Any]] = None
    critique: Optional[dict[str, Any]] = None
    action_result: Optional[dict[str, Any]] = None
    interrupt: Optional[dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    database: str
