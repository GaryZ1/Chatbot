"""Per-request tracing (docs/14-production-llmops.md §3).

One :class:`TraceRecord` per chat request holds everything §3 asks to log: query,
condensed query, route, filters, retrieved ids + scores for every stage, reranked
ids, final context ids, prompt version, model, tokens, latency per stage, answer,
citations, guardrail verdicts (and, later, user feedback). Stages are timed with
spans (context manager or the :func:`traced` decorator); a span's duration is also
accumulated into ``latency_ms[stage]``.

On finish the record is PII-redacted (:mod:`chatbot.guard.pii`), written as one
JSON log line to the ``chatbot.trace`` logger, appended to a
:class:`~chatbot.obs.trace_store.TraceStore` (for replay) and optionally exported
to OpenTelemetry using the GenAI semantic-convention attribute names.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import random
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from pydantic import BaseModel, Field

from chatbot.core.types import Answer, Chunk, Citation, ScoredChunk, utcnow
from chatbot.guard.pii import PIIDetector, default_detector
from chatbot.llm.base import Usage

if TYPE_CHECKING:
    from chatbot.obs.trace_store import TraceStore

logger = logging.getLogger(__name__)
trace_logger = logging.getLogger("chatbot.trace")

P = ParamSpec("P")
R = TypeVar("R")


# --------------------------------------------------------------------------- record model


class RetrievedItem(BaseModel):
    chunk_id: str
    doc_id: str = ""
    score: float = 0.0


class SpanRecord(BaseModel):
    name: str
    span_id: str
    parent_id: str | None = None
    start_unix_ns: int
    latency_ms: float
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceRecord(BaseModel):
    """Everything needed to debug, replay and evaluate one request."""

    trace_id: str
    created_at: datetime = Field(default_factory=utcnow)
    pipeline_version: str = ""
    prompt_version: str = ""
    tenant_id: str = ""
    #: Permission scope (hash), never the raw group list; enough to replay safely.
    scope_key: str = ""
    user_id: str = ""
    conversation_id: str = ""
    query: str = ""
    condensed_query: str = ""
    route: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    #: stage name (dense, sparse, fused, reranked ...) -> ranked ids + scores
    retrieval: dict[str, list[RetrievedItem]] = Field(default_factory=dict)
    context_ids: list[str] = Field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    cost_usd: float = 0.0
    latency_ms: dict[str, float] = Field(default_factory=dict)
    answer: str = ""
    status: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    guardrails: list[dict[str, Any]] = Field(default_factory=list)
    feedback: dict[str, Any] | None = None
    cache_hit: bool = False
    error: str | None = None
    spans: list[SpanRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return self.latency_ms.get("total", sum(self.latency_ms.values()))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------- redaction


def _redact_value(v: Any, red: PIIDetector) -> Any:
    if isinstance(v, str):
        return red.redact_text(v)
    if isinstance(v, list):
        return [_redact_value(x, red) for x in v]
    if isinstance(v, dict):
        return {k: _redact_value(x, red) for k, x in v.items()}
    return v


def redact_record(record: TraceRecord, redactor: PIIDetector | None = None) -> TraceRecord:
    """Copy of ``record`` with PII/secrets replaced by placeholders in all free text."""
    red = redactor or default_detector()
    r = record.model_copy(deep=True)
    r.query = red.redact_text(r.query)
    r.condensed_query = red.redact_text(r.condensed_query)
    r.answer = red.redact_text(r.answer)
    r.citations = [_redact_value(c, red) for c in r.citations]
    r.filters = _redact_value(r.filters, red)
    r.metadata = _redact_value(r.metadata, red)
    r.feedback = _redact_value(r.feedback, red) if r.feedback else r.feedback
    r.guardrails = [_redact_value(g, red) for g in r.guardrails]
    if r.error:
        r.error = red.redact_text(r.error)
    for s in r.spans:
        s.attributes = _redact_value(s.attributes, red)
        if s.error:
            s.error = red.redact_text(s.error)
    return r


# --------------------------------------------------------------------------- context

_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "chatbot_current_trace", default=None
)
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "chatbot_current_span", default=None
)


def current_trace() -> TraceContext | None:
    """The trace of the request being handled in this task, if any."""
    return _current_trace.get()


def _new_id(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


class Span:
    """A timed stage; attributes are free-form (strings are redacted on export)."""

    def __init__(self, name: str, parent_id: str | None) -> None:
        self.name = name
        self.span_id = _new_id(16)
        self.parent_id = parent_id
        self.attributes: dict[str, Any] = {}
        self.start_unix_ns = time.time_ns()
        self.status = "ok"
        self.error: str | None = None
        self.latency_ms = 0.0

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)


class TraceContext:
    """Mutable builder for one request's :class:`TraceRecord`."""

    def __init__(self, tracer: Tracer, record: TraceRecord, sampled: bool) -> None:
        self.tracer = tracer
        self.record = record
        self.sampled = sampled
        self._t0 = tracer.clock()
        self._token: contextvars.Token[TraceContext | None] | None = None
        self._finished = False

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    # ------------------------------------------------------------------ spans

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        sp = Span(name, _current_span.get())
        sp.set(**attributes)
        token = _current_span.set(sp.span_id)
        t0 = self.tracer.clock()
        try:
            yield sp
        except BaseException as exc:
            sp.status = "error"
            sp.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _current_span.reset(token)
            sp.latency_ms = round((self.tracer.clock() - t0) * 1000.0, 3)
            self.record.latency_ms[name] = round(
                self.record.latency_ms.get(name, 0.0) + sp.latency_ms, 3
            )
            self.record.spans.append(
                SpanRecord(
                    name=name,
                    span_id=sp.span_id,
                    parent_id=sp.parent_id,
                    start_unix_ns=sp.start_unix_ns,
                    latency_ms=sp.latency_ms,
                    status=sp.status,
                    error=sp.error,
                    attributes=dict(sp.attributes),
                )
            )

    # ------------------------------------------------------------------ fields

    def set(self, **fields: Any) -> None:
        """Set top-level record fields (route, condensed_query, filters, prompt_version ...)."""
        for k, v in fields.items():
            if k not in TraceRecord.model_fields:
                self.record.metadata[k] = v
            else:
                setattr(self.record, k, v)

    def record_retrieval(self, stage: str, items: Sequence[ScoredChunk | Chunk]) -> None:
        self.record.retrieval[stage] = [
            RetrievedItem(chunk_id=it.chunk.chunk_id, doc_id=it.chunk.doc_id, score=float(it.score))
            if isinstance(it, ScoredChunk)
            else RetrievedItem(chunk_id=it.chunk_id, doc_id=it.doc_id)
            for it in items
        ]

    def record_context(self, items: Sequence[ScoredChunk | Chunk | str]) -> None:
        ids: list[str] = []
        for it in items:
            if isinstance(it, ScoredChunk):
                ids.append(it.chunk.chunk_id)
            elif isinstance(it, Chunk):
                ids.append(it.chunk_id)
            else:
                ids.append(it)
        self.record.context_ids = ids

    def add_verdict(self, verdict: Any) -> None:
        """Accepts a guard verdict (anything with ``to_log()``) or a plain dict."""
        if hasattr(verdict, "to_log"):
            self.record.guardrails.append(dict(verdict.to_log()))
        elif isinstance(verdict, Mapping):
            self.record.guardrails.append(dict(verdict))
        else:
            raise TypeError(f"unsupported verdict type {type(verdict).__name__}")

    def record_llm(self, model: str, usage: Usage, *, cost_usd: float | None = None) -> None:
        """Accumulate one LLM call's usage (the last model is recorded as ``model``)."""
        self.record.model = model or self.record.model
        self.record.llm_calls += 1
        for k, v in usage.model_dump().items():
            self.record.usage[k] = self.record.usage.get(k, 0) + int(v)
        if cost_usd is None and self.tracer.prices is not None:
            cost_usd = self.tracer.prices.cost(model, usage)
        self.record.cost_usd = round(self.record.cost_usd + (cost_usd or 0.0), 8)

    def set_answer(self, answer: Answer | str) -> None:
        if isinstance(answer, str):
            self.record.answer = answer
            return
        r = self.record
        r.answer = answer.text
        r.status = answer.status.value
        r.route = answer.route or r.route
        r.condensed_query = answer.standalone_query or r.condensed_query
        if answer.context_chunk_ids:
            r.context_ids = list(answer.context_chunk_ids)
        r.citations = [_citation_log(c) for c in answer.citations]
        for flag in answer.guardrail_flags:
            r.guardrails.append({"stage": "answer", "flag": flag})
        if not r.usage and answer.usage:
            r.usage = dict(answer.usage)

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> TraceContext:
        self._token = _current_trace.set(self)
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._close(ev)

    async def __aenter__(self) -> TraceContext:
        return self.__enter__()

    async def __aexit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        record = self._close(ev)
        if record is not None and self.tracer.store is not None:
            try:
                await self.tracer.store.append(record)
            except Exception:  # never fail a user request because tracing failed
                logger.exception("failed to persist trace %s", record.trace_id)

    def _close(self, error: BaseException | None) -> TraceRecord | None:
        if self._token is not None:
            _current_trace.reset(self._token)
            self._token = None
        if self._finished:
            return None
        self._finished = True
        if error is not None and self.record.error is None:
            self.record.error = f"{type(error).__name__}: {error}"
        self.record.latency_ms["total"] = round((self.tracer.clock() - self._t0) * 1000.0, 3)
        if not self.sampled:
            return None
        return self.tracer.emit(self.record)


def _citation_log(c: Citation) -> dict[str, Any]:
    return {
        "index": c.index,
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "page": c.page,
        "supported": c.supported,
        "snippet": c.snippet[:300],
    }


# --------------------------------------------------------------------------- tracer


class Tracer:
    """Creates traces; owns redaction, logging, persistence and export."""

    def __init__(
        self,
        store: TraceStore | None = None,
        *,
        redact: bool = True,
        redactor: PIIDetector | None = None,
        exporter: OTelExporter | None = None,
        pipeline_version: str = "",
        sample_rate: float = 1.0,
        log_records: bool = True,
        prices: Any | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be within [0, 1]")
        self.store = store
        self.redact = redact
        self.redactor = redactor or default_detector()
        self.exporter = exporter
        self.pipeline_version = pipeline_version
        self.sample_rate = sample_rate
        self.log_records = log_records
        #: Optional :class:`chatbot.obs.metrics.PriceTable` for automatic cost.
        self.prices = prices
        self.clock = clock

    def trace(self, query: str = "", *, trace_id: str | None = None, **fields: Any) -> TraceContext:
        """Start a trace: ``async with tracer.trace(query=q, route=...) as t:``."""
        principal = fields.pop("principal", None)
        record = TraceRecord(
            trace_id=trace_id or uuid.uuid4().hex,
            pipeline_version=self.pipeline_version,
            query=query,
        )
        if principal is not None:
            record.user_id = principal.user_id
            record.tenant_id = principal.tenant_id
            record.scope_key = principal.scope_key()
        ctx = TraceContext(self, record, sampled=random.random() < self.sample_rate)
        ctx.set(**fields)
        return ctx

    def emit(self, record: TraceRecord) -> TraceRecord:
        """Redact, log and export a finished record; returns the persisted form."""
        out = redact_record(record, self.redactor) if self.redact else record
        if self.log_records:
            trace_logger.info(out.to_json())
        if self.exporter is not None:
            try:
                self.exporter.export(out)
            except Exception:
                logger.exception("OpenTelemetry export failed for trace %s", out.trace_id)
        return out


def traced(name: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: time a (sync or async) function as a span of the current trace.

    A no-op when no trace is active, so library code can be decorated freely.
    """

    def deco(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or fn.__name__
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                t = current_trace()
                if t is None:
                    return await fn(*args, **kwargs)
                with t.span(span_name):
                    return await fn(*args, **kwargs)

            return cast(Callable[P, R], awrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t = current_trace()
            if t is None:
                return fn(*args, **kwargs)
            with t.span(span_name):
                return fn(*args, **kwargs)

        return wrapper

    return deco


# --------------------------------------------------------------------------- OpenTelemetry


def genai_attributes(record: TraceRecord, provider: str = "anthropic") -> dict[str, Any]:
    """Root-span attributes following the OpenTelemetry GenAI semantic conventions.

    Standard ``gen_ai.*`` names are used where they exist; RAG-specific data uses a
    ``rag.*`` namespace. Free text (query / answer) is only included redacted and
    only if the record itself is redacted upstream.
    """
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": provider,
        "gen_ai.system": provider,  # pre-1.37 name, still read by many backends
        "gen_ai.request.model": record.model,
        "gen_ai.response.model": record.model,
        "gen_ai.usage.input_tokens": int(record.usage.get("input_tokens", 0)),
        "gen_ai.usage.output_tokens": int(record.usage.get("output_tokens", 0)),
        "gen_ai.usage.cache_read.input_tokens": int(record.usage.get("cache_read_input_tokens", 0)),
        "gen_ai.usage.cache_creation.input_tokens": int(
            record.usage.get("cache_creation_input_tokens", 0)
        ),
        "rag.trace_id": record.trace_id,
        "rag.route": record.route,
        "rag.prompt_version": record.prompt_version,
        "rag.pipeline_version": record.pipeline_version,
        "rag.context_ids": list(record.context_ids),
        "rag.status": record.status,
        "rag.cost_usd": record.cost_usd,
        "rag.cache_hit": record.cache_hit,
        "rag.guardrail_actions": [str(g.get("action", g.get("flag", ""))) for g in record.guardrails],
    }
    if record.conversation_id:
        attrs["gen_ai.conversation.id"] = record.conversation_id
    for stage, items in record.retrieval.items():
        attrs[f"rag.retrieval.{stage}.ids"] = [i.chunk_id for i in items]
        attrs[f"rag.retrieval.{stage}.scores"] = [i.score for i in items]
    return {k: v for k, v in attrs.items() if v not in ("", None)}


class OTelExporter:
    """Exports finished records as OpenTelemetry spans (``pip install chatbot[otel]``).

    The OTel SDK must be configured by the application (exporter / collector); this
    class only creates spans with the record's real timestamps.
    """

    def __init__(self, tracer_name: str = "chatbot.rag", provider: str = "anthropic") -> None:
        from opentelemetry import trace as otel_trace  # optional dependency

        self._otel = otel_trace
        self._tracer = otel_trace.get_tracer(tracer_name)
        self.provider = provider

    def export(self, record: TraceRecord) -> None:
        start_ns = int(record.created_at.timestamp() * 1e9)
        end_ns = start_ns + int(record.total_latency_ms * 1e6)
        root = self._tracer.start_span(
            f"chat {record.route or 'rag'}",
            start_time=start_ns,
            attributes=genai_attributes(record, self.provider),
        )
        ctx = self._otel.set_span_in_context(root)
        by_id: dict[str, Any] = {}
        for s in sorted(record.spans, key=lambda s: s.start_unix_ns):
            parent = by_id.get(s.parent_id or "")
            pctx = self._otel.set_span_in_context(parent) if parent is not None else ctx
            attrs = {
                k: v if isinstance(v, str | int | float | bool) else json.dumps(v, default=str)
                for k, v in s.attributes.items()
            }
            span = self._tracer.start_span(
                s.name, context=pctx, start_time=s.start_unix_ns, attributes=attrs
            )
            if s.status == "error":
                span.set_status(self._otel.Status(self._otel.StatusCode.ERROR, s.error or ""))
            span.end(end_time=s.start_unix_ns + int(s.latency_ms * 1e6))
            by_id[s.span_id] = span
        if record.error:
            root.set_status(self._otel.Status(self._otel.StatusCode.ERROR, record.error))
        root.end(end_time=end_ns)
