"""Data model of the query-understanding output (docs/06-query-understanding.md §3).

Kept separate from ``understanding.py`` so the router can import it without cycles;
``understanding`` re-exports everything here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from chatbot.core.filters import Filter


class Intent(StrEnum):
    SMALLTALK = "smalltalk"
    KB_QA = "kb_qa"
    ANALYTICS_SQL = "analytics_sql"
    ACTION = "action"
    OUT_OF_SCOPE = "out_of_scope"
    ESCALATE = "escalate"


class Route(StrEnum):
    SMALLTALK = "SMALLTALK"
    RAG_SIMPLE = "RAG_SIMPLE"
    RAG_DECOMPOSED = "RAG_DECOMPOSED"
    SQL_TOOL = "SQL_TOOL"
    AGENT = "AGENT"
    ESCALATE = "ESCALATE"
    CLARIFY = "CLARIFY"


class Complexity(StrEnum):
    SIMPLE = "simple"  # single fact / lookup
    MULTI_PART = "multi_part"  # several independent sub-questions (comparisons, lists)
    MULTI_HOP = "multi_hop"  # later steps depend on earlier answers


class RetrievalMode(StrEnum):
    """Adaptive retrieval decision (Self-RAG / Adaptive-RAG, docs/06 §1)."""

    NONE = "none"
    SINGLE = "single"
    MULTI = "multi"  # parallel sub-queries, fused
    ITERATIVE = "iterative"  # sequential multi-hop retrieval


class PlanSource(StrEnum):
    FAST_PATH = "fast_path"
    LLM = "llm"
    CACHE = "cache"
    FALLBACK = "fallback"


class QueryPlan(BaseModel):
    intent: Intent
    route: Route
    standalone_query: str
    sub_queries: list[str] = Field(default_factory=list)
    #: Validated (accepted) raw filters as proposed by the model, for logging/debugging.
    filters: dict[str, Any] = Field(default_factory=dict)
    #: The validated filter to AND with the ACL filter at retrieval time.
    filter: Filter | None = None
    dropped_filters: list[str] = Field(default_factory=list)
    needs_retrieval: bool = True
    language: str = "en"
    clarify_question: str | None = None
    complexity: Complexity = Complexity.SIMPLE
    retrieval_mode: RetrievalMode = RetrievalMode.SINGLE
    original_query: str = ""
    normalized_query: str = ""
    source: PlanSource = PlanSource.LLM
    latency_ms: float = 0.0

    @property
    def retrieval_queries(self) -> list[str]:
        """Queries to send to retrieval: sub-queries when decomposed, else the standalone query."""
        if self.route is Route.RAG_DECOMPOSED and self.sub_queries:
            return list(self.sub_queries)
        return [self.standalone_query]
