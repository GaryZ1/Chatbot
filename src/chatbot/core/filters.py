"""A small, store-agnostic metadata filter language.

Every retriever applies the *same* ACL + metadata pre-filter before scoring and
top-K (docs/07-retrieval.md §2, docs/13-guardrails-security.md §3). The in-memory
stores evaluate filters with :func:`Filter.matches`; external stores translate the
same structure into their native query language.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from chatbot.core.types import Chunk, Principal


class Op(StrEnum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    NIN = "nin"
    GTE = "gte"
    LTE = "lte"
    GT = "gt"
    LT = "lt"
    CONTAINS = "contains"  # list field contains value
    EXISTS = "exists"


class Condition(BaseModel):
    field: str
    op: Op = Op.EQ
    value: Any = None


#: Top-level chunk attributes that may be filtered directly; anything else is read
#: from ``chunk.metadata``.
_TOP_LEVEL = {"doc_id", "tenant_id", "kind", "page", "doc_version", "parent_id", "title"}


def _get(chunk: Chunk, field: str) -> Any:
    if field in _TOP_LEVEL:
        val = getattr(chunk, field)
        return val.value if isinstance(val, StrEnum) else val
    return chunk.metadata.get(field)


def _cmp_value(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return v


def _eval(cond: Condition, chunk: Chunk) -> bool:
    actual = _get(chunk, cond.field)
    v = cond.value
    match cond.op:
        case Op.EXISTS:
            return (actual is not None) == bool(v if v is not None else True)
        case Op.EQ:
            return actual == v
        case Op.NE:
            return actual != v
        case Op.IN:
            if isinstance(actual, list):
                return bool(set(actual) & set(v))
            return actual in v
        case Op.NIN:
            if isinstance(actual, list):
                return not (set(actual) & set(v))
            return actual not in v
        case Op.CONTAINS:
            return isinstance(actual, list) and v in actual
    if actual is None:
        return False
    a, b = _cmp_value(actual), _cmp_value(v)
    try:
        match cond.op:
            case Op.GTE:
                return bool(a >= b)
            case Op.LTE:
                return bool(a <= b)
            case Op.GT:
                return bool(a > b)
            case Op.LT:
                return bool(a < b)
    except TypeError:
        return False
    return False


class Filter(BaseModel):
    """Conjunction of ``must`` conditions, minus any ``must_not`` match."""

    must: list[Condition] = Field(default_factory=list)
    must_not: list[Condition] = Field(default_factory=list)

    @classmethod
    def eq(cls, **fields: Any) -> Filter:
        return cls(must=[Condition(field=k, op=Op.EQ, value=v) for k, v in fields.items()])

    def and_(self, other: Filter | None) -> Filter:
        if other is None:
            return self
        return Filter(must=[*self.must, *other.must], must_not=[*self.must_not, *other.must_not])

    def matches(self, chunk: Chunk) -> bool:
        return all(_eval(c, chunk) for c in self.must) and not any(
            _eval(c, chunk) for c in self.must_not
        )

    def is_empty(self) -> bool:
        return not self.must and not self.must_not


def visible(chunk: Chunk, principal: Principal | None, flt: Filter | None = None) -> bool:
    """Single source of truth for pre-filtering: tenant isolation, ACL, then metadata.

    ``principal=None`` is reserved for trusted offline jobs (evaluation, maintenance)
    and bypasses ACLs. Request handlers must always pass a principal.
    """
    if principal is not None:
        if chunk.tenant_id != principal.tenant_id:
            return False
        if not chunk.acl.permits(principal):
            return False
    return flt is None or flt.matches(chunk)
