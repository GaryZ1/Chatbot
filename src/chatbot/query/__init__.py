"""Query understanding & transformation (docs/06-query-understanding.md)."""

from chatbot.query.cache import TTLCache
from chatbot.query.condense import (
    CONDENSE_SYSTEM,
    QueryCondenser,
    enforce_entities,
    extract_entities,
    format_history,
)
from chatbot.query.decompose import Decomposition, QueryDecomposer, looks_multi_part
from chatbot.query.filters import (
    DroppedFilter,
    FieldSpec,
    FieldType,
    FilterSchema,
    FilterValidation,
    SelfQueryExtractor,
    SelfQueryResult,
)
from chatbot.query.hyde import HyDEGenerator, HyDEResult
from chatbot.query.multi_query import MultiQueryGenerator, dedupe_queries
from chatbot.query.normalize import (
    QueryNormalizer,
    clean_text,
    contains_identifier,
    find_identifiers,
    looks_like_identifier,
)
from chatbot.query.plan import Complexity, Intent, PlanSource, QueryPlan, RetrievalMode, Route
from chatbot.query.rewrite import QueryRewriter
from chatbot.query.router import (
    QueryRouter,
    RouterConfig,
    RoutingDecision,
    RoutingLogRecord,
    RoutingReport,
    decide_route,
    report_from_records,
    routing_report,
)
from chatbot.query.step_back import StepBackGenerator
from chatbot.query.understanding import QueryUnderstanding, detect_smalltalk, history_hash

__all__ = [
    "CONDENSE_SYSTEM",
    "Complexity",
    "Decomposition",
    "DroppedFilter",
    "FieldSpec",
    "FieldType",
    "FilterSchema",
    "FilterValidation",
    "HyDEGenerator",
    "HyDEResult",
    "Intent",
    "MultiQueryGenerator",
    "PlanSource",
    "QueryCondenser",
    "QueryDecomposer",
    "QueryNormalizer",
    "QueryPlan",
    "QueryRewriter",
    "QueryRouter",
    "QueryUnderstanding",
    "RetrievalMode",
    "Route",
    "RouterConfig",
    "RoutingDecision",
    "RoutingLogRecord",
    "RoutingReport",
    "SelfQueryExtractor",
    "SelfQueryResult",
    "StepBackGenerator",
    "TTLCache",
    "clean_text",
    "contains_identifier",
    "decide_route",
    "dedupe_queries",
    "detect_smalltalk",
    "enforce_entities",
    "extract_entities",
    "find_identifiers",
    "format_history",
    "history_hash",
    "looks_like_identifier",
    "looks_multi_part",
    "report_from_records",
    "routing_report",
]
