"""HTTP request/response models for the chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chatbot.core.types import AnswerStatus, Citation


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: str | None = Field(default=None, max_length=128)
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    status: AnswerStatus
    citations: list[Citation] = Field(default_factory=list)
    route: str = ""
    trace_id: str = ""
    latency_ms: dict[str, float] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str | None = None
    rating: int = Field(ge=-1, le=1)  # -1 thumbs down, 1 thumbs up
    reason: str | None = Field(default=None, max_length=64)
    comment: str | None = Field(default=None, max_length=2000)


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    incremental: bool = True


class IngestResponse(BaseModel):
    added: int
    updated: int
    deleted: int
    skipped: int
    chunks: int


class MemoryItem(BaseModel):
    id: str
    text: str
    created_at: str | None = None
    updated_at: str | None = None
