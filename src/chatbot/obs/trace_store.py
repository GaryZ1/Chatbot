"""Trace persistence and replay (docs/14-production-llmops.md §3, "Replay").

Stored traces are already PII-redacted by :class:`~chatbot.obs.tracing.Tracer`.
:func:`replay` re-runs stored traces through a new pipeline version and reports how
the answers changed, the basis for regression testing against production traffic.

Replaying safely: run the candidate pipeline with a *trace-free / write-free*
configuration (no trace store, no cache writes, no tools with side effects) and
with the original principal's permission scope; the ``pipeline`` callable receives
the full record so it can rebuild that scope.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from chatbot.core.types import Answer
from chatbot.obs.tracing import TraceRecord

logger = logging.getLogger(__name__)


@runtime_checkable
class TraceStore(Protocol):
    async def append(self, record: TraceRecord) -> None: ...

    async def get(self, trace_id: str) -> TraceRecord | None: ...

    async def list(
        self, *, limit: int | None = None, since: datetime | None = None, route: str | None = None
    ) -> list[TraceRecord]: ...

    async def attach_feedback(self, trace_id: str, feedback: dict[str, object]) -> bool: ...


def _match(r: TraceRecord, since: datetime | None, route: str | None) -> bool:
    if since is not None and r.created_at < since:
        return False
    return route is None or r.route == route


class InMemoryTraceStore:
    """Bounded in-memory store (tests, local development)."""

    def __init__(self, max_records: int = 10_000) -> None:
        self.max_records = max_records
        self._records: dict[str, TraceRecord] = {}

    async def append(self, record: TraceRecord) -> None:
        self._records[record.trace_id] = record
        while len(self._records) > self.max_records:
            self._records.pop(next(iter(self._records)))

    async def get(self, trace_id: str) -> TraceRecord | None:
        return self._records.get(trace_id)

    async def list(
        self, *, limit: int | None = None, since: datetime | None = None, route: str | None = None
    ) -> list[TraceRecord]:
        out = [r for r in self._records.values() if _match(r, since, route)]
        return out[-limit:] if limit else out

    async def attach_feedback(self, trace_id: str, feedback: dict[str, object]) -> bool:
        r = self._records.get(trace_id)
        if r is None:
            return False
        r.feedback = dict(feedback)
        return True

    def __len__(self) -> int:
        return len(self._records)


class JsonlTraceStore:
    """Append-only JSONL file (one record per line). Feedback is appended as a patch line.

    Suitable for a single process or as a sink shipped to a log pipeline; use a real
    database or an observability backend for fleet-wide querying.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _append_line(self, line: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    async def append(self, record: TraceRecord) -> None:
        line = record.to_json()
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)

    def _load(self) -> dict[str, TraceRecord]:
        records: dict[str, TraceRecord] = {}
        if not self.path.exists():
            return records
        with self.path.open(encoding="utf-8") as fh:
            for n, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "_feedback_for" in obj:
                        rec = records.get(obj["_feedback_for"])
                        if rec is not None:
                            rec.feedback = obj["feedback"]
                        continue
                    rec = TraceRecord.model_validate(obj)
                except (json.JSONDecodeError, ValidationError, KeyError) as exc:
                    logger.warning("skipping corrupt trace line %s:%d: %s", self.path, n, exc)
                    continue
                records[rec.trace_id] = rec
        return records

    async def get(self, trace_id: str) -> TraceRecord | None:
        return (await asyncio.to_thread(self._load)).get(trace_id)

    async def list(
        self, *, limit: int | None = None, since: datetime | None = None, route: str | None = None
    ) -> list[TraceRecord]:
        recs = [
            r for r in (await asyncio.to_thread(self._load)).values() if _match(r, since, route)
        ]
        return recs[-limit:] if limit else recs

    async def attach_feedback(self, trace_id: str, feedback: dict[str, object]) -> bool:
        if await self.get(trace_id) is None:
            return False
        line = json.dumps({"_feedback_for": trace_id, "feedback": feedback}, default=str)
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)
        return True


# --------------------------------------------------------------------------- replay

_TOK_RE = re.compile(r"\w+")


def answer_similarity(a: str, b: str) -> float:
    """Token Jaccard similarity (cheap default; plug in an embedding or judge for semantics)."""
    ta, tb = set(_TOK_RE.findall(a.lower())), set(_TOK_RE.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


class ReplayResult(BaseModel):
    trace_id: str
    query: str
    old_answer: str
    new_answer: str = ""
    old_status: str = ""
    new_status: str = ""
    similarity: float = 0.0
    changed: bool = False
    old_context_ids: list[str] = Field(default_factory=list)
    new_context_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class ReplayReport(BaseModel):
    results: list[ReplayResult]

    @property
    def changed(self) -> list[ReplayResult]:
        return [r for r in self.results if r.changed]

    @property
    def errors(self) -> list[ReplayResult]:
        return [r for r in self.results if r.error]

    @property
    def change_rate(self) -> float:
        return len(self.changed) / len(self.results) if self.results else 0.0

    @property
    def mean_similarity(self) -> float:
        ok = [r.similarity for r in self.results if not r.error]
        return sum(ok) / len(ok) if ok else 0.0


ReplayPipeline = Callable[[TraceRecord], Awaitable[Answer | str]]


async def replay(
    traces: Iterable[TraceRecord],
    pipeline: ReplayPipeline,
    *,
    similarity: Callable[[str, str], float] = answer_similarity,
    changed_below: float = 0.8,
    concurrency: int = 4,
) -> ReplayReport:
    """Re-run each trace's query through ``pipeline`` and compare answers.

    An answer counts as *changed* if similarity < ``changed_below`` or the status
    (answered / abstained / blocked ...) differs.
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(rec: TraceRecord) -> ReplayResult:
        res = ReplayResult(
            trace_id=rec.trace_id,
            query=rec.query,
            old_answer=rec.answer,
            old_status=rec.status,
            old_context_ids=list(rec.context_ids),
        )
        try:
            async with sem:
                out = await pipeline(rec)
        except Exception as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            res.changed = True
            return res
        if isinstance(out, Answer):
            res.new_answer = out.text
            res.new_status = out.status.value
            res.new_context_ids = list(out.context_chunk_ids)
        else:
            res.new_answer = out
            res.new_status = rec.status
        res.similarity = round(similarity(res.old_answer, res.new_answer), 4)
        res.changed = res.similarity < changed_below or res.new_status != res.old_status
        return res

    items: Sequence[TraceRecord] = list(traces)
    results = await asyncio.gather(*(one(r) for r in items))
    return ReplayReport(results=list(results))


async def replay_store(
    store: TraceStore,
    pipeline: ReplayPipeline,
    *,
    limit: int | None = 200,
    since: datetime | None = None,
    route: str | None = None,
    **kw: object,
) -> ReplayReport:
    """Load traces from ``store`` and :func:`replay` them (errored traces are skipped)."""
    traces = [t for t in await store.list(limit=limit, since=since, route=route) if not t.error]
    return await replay(traces, pipeline, **kw)  # type: ignore[arg-type]
