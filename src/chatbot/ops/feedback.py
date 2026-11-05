"""User feedback and the feedback -> golden-set pipeline (docs/14-production-llmops.md §3, §6).

Thumbs up/down with reason codes are stored per trace. Negative feedback is
converted into *golden-set candidates* (question, answer, retrieved ids, reason,
suggested failure code) for the weekly error review: a human triages each one into
a regression test, a content fix, or noise (docs/14 §6, "bad answer reported ->
trace -> root cause -> regression test added").
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from chatbot.core.failures import FailureCode
from chatbot.core.types import utcnow

logger = logging.getLogger(__name__)


class Rating(StrEnum):
    UP = "up"
    DOWN = "down"


class ReasonCode(StrEnum):
    WRONG = "wrong"
    INCOMPLETE = "incomplete"
    OUTDATED = "outdated"
    NOT_RELEVANT = "not_relevant"
    HALLUCINATION = "hallucination"
    BAD_CITATION = "bad_citation"
    MISSING_INFO = "missing_info"  # "the bot said it doesn't know but the docs have it"
    UNAUTHORIZED = "unauthorized"  # user saw something they should not
    UNSAFE = "unsafe"
    FORMAT = "format"
    TOO_LONG = "too_long"
    MISUNDERSTOOD = "misunderstood"  # follow-up resolved wrongly
    OTHER = "other"


#: First guess at the failure stage for triage (docs/00 §3 taxonomy).
REASON_TO_FAILURE: dict[ReasonCode, FailureCode] = {
    ReasonCode.OUTDATED: FailureCode.F12_STALE_CONFLICTING,
    ReasonCode.INCOMPLETE: FailureCode.F11_INCOMPLETE,
    ReasonCode.MISSING_INFO: FailureCode.F5_MISSED_TOP_K,
    ReasonCode.NOT_RELEVANT: FailureCode.F5_MISSED_TOP_K,
    ReasonCode.HALLUCINATION: FailureCode.F8_NOT_EXTRACTED,
    ReasonCode.WRONG: FailureCode.F8_NOT_EXTRACTED,
    ReasonCode.FORMAT: FailureCode.F9_WRONG_FORMAT,
    ReasonCode.TOO_LONG: FailureCode.F9_WRONG_FORMAT,
    ReasonCode.UNAUTHORIZED: FailureCode.F13_LEAKAGE,
    ReasonCode.UNSAFE: FailureCode.F14_INJECTION,
    ReasonCode.MISUNDERSTOOD: FailureCode.F15_CONVERSATION_DRIFT,
}

#: Reasons that must page someone rather than wait for the weekly review.
URGENT_REASONS: frozenset[ReasonCode] = frozenset({ReasonCode.UNAUTHORIZED, ReasonCode.UNSAFE})


class Feedback(BaseModel):
    trace_id: str
    user_id: str = ""
    rating: Rating
    reasons: list[ReasonCode] = Field(default_factory=list)
    comment: str = Field(default="", max_length=4000)
    #: Optional user-supplied correct answer / pointer to the right document.
    correction: str = Field(default="", max_length=4000)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def negative(self) -> bool:
        return self.rating is Rating.DOWN

    @property
    def urgent(self) -> bool:
        return bool(URGENT_REASONS.intersection(self.reasons))


@runtime_checkable
class FeedbackStore(Protocol):
    async def add(self, feedback: Feedback) -> None: ...

    async def list(self, *, since: datetime | None = None, negative_only: bool = False) -> list[Feedback]: ...


class InMemoryFeedbackStore:
    def __init__(self) -> None:
        self._items: list[Feedback] = []

    async def add(self, feedback: Feedback) -> None:
        self._items.append(feedback)

    async def list(self, *, since: datetime | None = None, negative_only: bool = False) -> list[Feedback]:
        return [
            f
            for f in self._items
            if (since is None or f.created_at >= since) and (not negative_only or f.negative)
        ]


class JsonlFeedbackStore:
    """Append-only JSONL feedback log."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _write(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _read(self) -> list[Feedback]:
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    out.append(Feedback.model_validate_json(line))
                except ValidationError as exc:
                    logger.warning("skipping bad feedback line %s:%d: %s", self.path, n, exc)
        return out

    async def add(self, feedback: Feedback) -> None:
        async with self._lock:
            await asyncio.to_thread(self._write, feedback.model_dump_json())
        if feedback.urgent:
            logger.error(
                "URGENT feedback on trace %s: %s", feedback.trace_id, [r.value for r in feedback.reasons]
            )

    async def list(self, *, since: datetime | None = None, negative_only: bool = False) -> list[Feedback]:
        items = await asyncio.to_thread(self._read)
        return [
            f
            for f in items
            if (since is None or f.created_at >= since) and (not negative_only or f.negative)
        ]


# --------------------------------------------------------------------------- golden-set candidates


class GoldenCandidate(BaseModel):
    """A record for triage into the golden evaluation set (docs/12)."""

    trace_id: str
    question: str
    condensed_question: str = ""
    answer: str = ""
    retrieved_ids: list[str] = Field(default_factory=list)
    context_ids: list[str] = Field(default_factory=list)
    reasons: list[ReasonCode] = Field(default_factory=list)
    comment: str = ""
    correction: str = ""
    suggested_failure: FailureCode | None = None
    urgent: bool = False
    route: str = ""
    status: str = "triage"  # triage -> accepted | rejected | content_fix
    created_at: datetime = Field(default_factory=utcnow)


@runtime_checkable
class TraceLookup(Protocol):
    """Anything that can fetch a stored trace (``chatbot.obs.trace_store.TraceStore``)."""

    async def get(self, trace_id: str) -> Any: ...


def _attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _retrieved_ids(trace: Any) -> list[str]:
    retrieval = _attr(trace, "retrieval", {}) or {}
    seen: dict[str, None] = {}
    for items in retrieval.values():
        for it in items:
            cid = _attr(it, "chunk_id", None)
            if cid:
                seen.setdefault(cid, None)
    return list(seen)


async def to_golden_candidates(
    feedback: Iterable[Feedback], traces: TraceLookup
) -> list[GoldenCandidate]:
    """Convert negative feedback into triage candidates (deduplicated per trace)."""
    out: dict[str, GoldenCandidate] = {}
    for fb in feedback:
        if not fb.negative:
            continue
        trace = await traces.get(fb.trace_id)
        if trace is None:
            logger.warning("feedback for unknown trace %s; skipping", fb.trace_id)
            continue
        suggested = next((REASON_TO_FAILURE[r] for r in fb.reasons if r in REASON_TO_FAILURE), None)
        prev = out.get(fb.trace_id)
        if prev is not None:
            prev.reasons = list(dict.fromkeys([*prev.reasons, *fb.reasons]))
            prev.comment = "\n".join(c for c in (prev.comment, fb.comment) if c)
            prev.urgent = prev.urgent or fb.urgent
            continue
        out[fb.trace_id] = GoldenCandidate(
            trace_id=fb.trace_id,
            question=_attr(trace, "query", ""),
            condensed_question=_attr(trace, "condensed_query", ""),
            answer=_attr(trace, "answer", ""),
            retrieved_ids=_retrieved_ids(trace),
            context_ids=list(_attr(trace, "context_ids", []) or []),
            reasons=list(fb.reasons),
            comment=fb.comment,
            correction=fb.correction,
            suggested_failure=suggested,
            urgent=fb.urgent,
            route=_attr(trace, "route", ""),
        )
    # Urgent first, then oldest first.
    return sorted(out.values(), key=lambda c: (not c.urgent, c.created_at))


def export_candidates(candidates: Sequence[GoldenCandidate], path: str | Path) -> int:
    """Append candidates to a JSONL triage file; returns the number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for c in candidates:
            fh.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return len(candidates)


def feedback_summary(feedback: Iterable[Feedback]) -> dict[str, Any]:
    """Thumbs-up rate and reason histogram for the dashboard."""
    items = list(feedback)
    up = sum(f.rating is Rating.UP for f in items)
    reasons: dict[str, int] = {}
    for f in items:
        for r in f.reasons:
            reasons[r.value] = reasons.get(r.value, 0) + 1
    return {
        "count": len(items),
        "thumbs_up_rate": up / len(items) if items else 0.0,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
