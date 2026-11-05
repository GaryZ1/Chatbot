"""Intent → route decision, adaptive retrieval decision and routing logs
(docs/06-query-understanding.md §1 "Intent classification & routing", "Adaptive retrieval
decision", §4 "Routing accuracy (a confusion matrix over a labeled set)").

Routing is deterministic given the structured plan, so the LLM never names a route
directly and plan → route stays testable. Every decision can be emitted as a
:class:`RoutingLogRecord`; sampling and labelling those records feeds the confusion
matrix that detects silent misroutes ("8% of analytics questions go to the text KB").
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime

from pydantic import BaseModel, Field

from chatbot.core.types import utcnow
from chatbot.query.plan import Complexity, Intent, PlanSource, QueryPlan, RetrievalMode, Route

logger = logging.getLogger(__name__)


class RouterConfig(BaseModel):
    sql_enabled: bool = True
    agent_enabled: bool = True
    clarify_enabled: bool = True
    escalate_enabled: bool = True
    #: Minimum number of sub-queries for the decomposed route.
    min_sub_queries: int = 2


class RoutingDecision(BaseModel):
    route: Route
    retrieval_mode: RetrievalMode
    reason: str


def decide_route(
    *,
    intent: Intent,
    sub_queries: list[str],
    complexity: Complexity,
    needs_retrieval: bool,
    clarify_question: str | None,
    config: RouterConfig | None = None,
) -> RoutingDecision:
    """Pure routing function used by both :class:`QueryRouter` and the understanding step."""
    cfg = config or RouterConfig()
    R, M = Route, RetrievalMode
    if intent in (Intent.SMALLTALK, Intent.OUT_OF_SCOPE):
        return RoutingDecision(route=R.SMALLTALK, retrieval_mode=M.NONE, reason=intent.value)
    if intent is Intent.ESCALATE:
        if cfg.escalate_enabled:
            return RoutingDecision(route=R.ESCALATE, retrieval_mode=M.NONE, reason="escalate")
        intent = Intent.KB_QA
    if clarify_question and clarify_question.strip() and cfg.clarify_enabled:
        return RoutingDecision(route=R.CLARIFY, retrieval_mode=M.NONE, reason="ambiguous")
    if intent is Intent.ANALYTICS_SQL:
        if cfg.sql_enabled:
            return RoutingDecision(route=R.SQL_TOOL, retrieval_mode=M.NONE, reason="analytics")
        logger.info("SQL route disabled; falling back to RAG for analytics intent")
    if intent is Intent.ACTION:
        if cfg.agent_enabled:
            mode = M.SINGLE if needs_retrieval else M.NONE
            return RoutingDecision(route=R.AGENT, retrieval_mode=mode, reason="action")
        if cfg.escalate_enabled:
            return RoutingDecision(route=R.ESCALATE, retrieval_mode=M.NONE, reason="action_no_agent")
    # Knowledge-base QA (and fall-throughs).
    if len(sub_queries) >= cfg.min_sub_queries:
        mode = M.ITERATIVE if complexity is Complexity.MULTI_HOP else M.MULTI
        if cfg.agent_enabled and complexity is Complexity.MULTI_HOP and len(sub_queries) > 4:
            return RoutingDecision(route=R.AGENT, retrieval_mode=M.ITERATIVE, reason="deep_multi_hop")
        return RoutingDecision(route=R.RAG_DECOMPOSED, retrieval_mode=mode, reason="decomposed")
    if not needs_retrieval:
        return RoutingDecision(route=R.RAG_SIMPLE, retrieval_mode=M.NONE, reason="no_retrieval")
    return RoutingDecision(route=R.RAG_SIMPLE, retrieval_mode=M.SINGLE, reason="kb_qa")


class RoutingLogRecord(BaseModel):
    """One routing decision; add ``gold_route`` when a labeller reviews it."""

    timestamp: datetime = Field(default_factory=utcnow)
    trace_id: str = ""
    query: str
    standalone_query: str
    intent: Intent
    route: Route
    retrieval_mode: RetrievalMode
    reason: str = ""
    plan_source: PlanSource = PlanSource.LLM
    latency_ms: float = 0.0
    gold_route: Route | None = None


class QueryRouter:
    """Routes plans and emits routing log records to an optional sink."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        *,
        log_sink: Callable[[RoutingLogRecord], None] | None = None,
    ) -> None:
        self.config = config or RouterConfig()
        self.log_sink = log_sink

    def decide(self, plan: QueryPlan) -> RoutingDecision:
        return decide_route(
            intent=plan.intent,
            sub_queries=plan.sub_queries,
            complexity=plan.complexity,
            needs_retrieval=plan.needs_retrieval,
            clarify_question=plan.clarify_question,
            config=self.config,
        )

    def route(self, plan: QueryPlan, *, trace_id: str = "") -> QueryPlan:
        """Return a copy of ``plan`` with route + retrieval mode (re)decided, and log it."""
        d = self.decide(plan)
        routed = plan.model_copy(update={"route": d.route, "retrieval_mode": d.retrieval_mode})
        record = RoutingLogRecord(
            trace_id=trace_id,
            query=plan.original_query,
            standalone_query=plan.standalone_query,
            intent=plan.intent,
            route=d.route,
            retrieval_mode=d.retrieval_mode,
            reason=d.reason,
            plan_source=plan.source,
            latency_ms=plan.latency_ms,
        )
        logger.info(
            "route intent=%s route=%s mode=%s reason=%s source=%s",
            record.intent.value,
            record.route.value,
            record.retrieval_mode.value,
            record.reason,
            record.plan_source.value,
        )
        if self.log_sink is not None:
            try:
                self.log_sink(record)
            except Exception:  # a broken log sink must never fail the request
                logger.exception("routing log sink failed")
        return routed


# --------------------------------------------------------------------------- evaluation


class RoutingReport(BaseModel):
    labels: list[str]
    matrix: dict[str, dict[str, int]]  # gold → predicted → count
    accuracy: float
    recall: dict[str, float]
    precision: dict[str, float]
    support: dict[str, int]
    n: int


def routing_report(pairs: Iterable[tuple[str, str]]) -> RoutingReport:
    """Confusion matrix + per-class precision/recall from ``(gold, predicted)`` pairs.

    Off-diagonal cells are the silent misroutes, e.g. ``matrix["SQL_TOOL"]["RAG_SIMPLE"]``.
    """
    items = [(str(g), str(p)) for g, p in pairs]
    labels = sorted({x for pair in items for x in pair})
    matrix = {g: dict.fromkeys(labels, 0) for g in labels}
    for g, p in items:
        matrix[g][p] += 1
    gold_counts = Counter(g for g, _ in items)
    pred_counts = Counter(p for _, p in items)
    correct = sum(matrix[label][label] for label in labels)
    return RoutingReport(
        labels=labels,
        matrix=matrix,
        accuracy=correct / len(items) if items else 0.0,
        recall={
            label: matrix[label][label] / gold_counts[label] if gold_counts[label] else 0.0
            for label in labels
        },
        precision={
            label: matrix[label][label] / pred_counts[label] if pred_counts[label] else 0.0
            for label in labels
        },
        support=dict(gold_counts),
        n=len(items),
    )


def report_from_records(records: Iterable[RoutingLogRecord]) -> RoutingReport:
    """Routing report over labelled log records (records without ``gold_route`` are skipped)."""
    return routing_report(
        (r.gold_route.value, r.route.value) for r in records if r.gold_route is not None
    )
