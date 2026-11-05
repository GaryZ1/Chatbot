"""HTTP request/response models for the chat API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from chatbot.core.types import AnswerStatus, Citation
from chatbot.ops.feedback import Rating, ReasonCode


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
    trace_id: str = Field(min_length=1, max_length=64)
    rating: Rating
    reasons: list[ReasonCode] = Field(default_factory=list, max_length=10)
    comment: str = Field(default="", max_length=4000)
    correction: str = Field(default="", max_length=4000)


class IngestRequest(BaseModel):
    paths: list[str] = Field(min_length=1)
    incremental: bool = True


class IngestResponse(BaseModel):
    added: int
    updated: int
    deleted: int
    skipped: int
    chunks: int
    quarantined: list[str] = Field(default_factory=list)
    failed: dict[str, str] = Field(default_factory=dict)
