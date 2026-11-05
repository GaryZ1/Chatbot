"""ACL-aware retrieval helpers and permission-matrix testing (docs/13-guardrails-security.md §3).

* Group memberships are resolved at query time from the IdP and cached with a short
  TTL (:class:`StaticGroupResolver` is the in-process reference implementation;
  production resolvers implement :class:`PrincipalResolver`).
* Filtering itself is done *inside* every store via :func:`chatbot.core.filters.visible`
  before top-K; the helpers here re-use it so there is a single source of truth.
* :func:`assert_no_leak` and :class:`AclMatrix` prove that no unauthorised chunk
  *ever* appears in the retrieved set (not just in the answer).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

from chatbot.core.filters import Filter, visible
from chatbot.core.types import ACL, Chunk, Document, Principal, ScoredChunk

logger = logging.getLogger(__name__)

Clock = Callable[[], float]


@runtime_checkable
class PrincipalResolver(Protocol):
    """Resolves a user's current groups/roles (IdP, SCIM, LDAP ...)."""

    async def resolve(self, user_id: str, tenant_id: str = "default") -> Principal: ...


class StaticGroupResolver:
    """Group memberships from a mapping, with nested groups and a TTL cache.

    ``memberships`` maps ``user_id -> groups``; ``group_parents`` maps a group to the
    groups it is a member of (resolved transitively, e.g. ``eng-backend -> eng``).
    Updates via :meth:`set_memberships` model IdP changes: they become visible after
    the TTL expires (the permission-propagation SLA) or immediately after
    :meth:`invalidate`.
    """

    def __init__(
        self,
        memberships: Mapping[str, Iterable[str]] | None = None,
        *,
        group_parents: Mapping[str, Iterable[str]] | None = None,
        roles: Mapping[str, Iterable[str]] | None = None,
        ttl_s: float = 300.0,
        clock: Clock = time.monotonic,
    ) -> None:
        if ttl_s < 0:
            raise ValueError("ttl_s must be >= 0")
        self._members = {u: set(g) for u, g in (memberships or {}).items()}
        self._parents = {g: set(p) for g, p in (group_parents or {}).items()}
        self._roles = {u: frozenset(r) for u, r in (roles or {}).items()}
        self.ttl_s = ttl_s
        self._clock = clock
        self._cache: dict[tuple[str, str], tuple[float, Principal]] = {}
        self._lock = asyncio.Lock()
        self.lookups = 0  # number of source lookups (cache misses)

    def set_memberships(self, user_id: str, groups: Iterable[str]) -> None:
        self._members[user_id] = set(groups)

    def invalidate(self, user_id: str | None = None) -> None:
        if user_id is None:
            self._cache.clear()
        else:
            for key in [k for k in self._cache if k[0] == user_id]:
                del self._cache[key]

    def _expand(self, groups: set[str]) -> frozenset[str]:
        seen: set[str] = set()
        stack = list(groups)
        while stack:
            g = stack.pop()
            if g in seen:
                continue
            seen.add(g)
            stack.extend(self._parents.get(g, ()))
        return frozenset(seen)

    async def resolve(self, user_id: str, tenant_id: str = "default") -> Principal:
        key = (user_id, tenant_id)
        now = self._clock()
        async with self._lock:
            hit = self._cache.get(key)
            if hit is not None and now - hit[0] < self.ttl_s:
                return hit[1]
            self.lookups += 1
            principal = Principal(
                user_id=user_id,
                tenant_id=tenant_id,
                groups=self._expand(self._members.get(user_id, set())),
                roles=self._roles.get(user_id, frozenset()),
            )
            self._cache[key] = (now, principal)
            return principal


# --------------------------------------------------------------------------- filtering helpers


def _chunk(item: ScoredChunk | Chunk) -> Chunk:
    return item.chunk if isinstance(item, ScoredChunk) else item


T = TypeVar("T", ScoredChunk, Chunk)


def acl_filter(
    items: Iterable[T], principal: Principal | None, flt: Filter | None = None
) -> list[T]:
    """Keep only items visible to ``principal`` (defence-in-depth post-filter).

    Stores must already pre-filter; this is the belt-and-braces check that runs
    again right before prompt assembly.
    """
    out: list[T] = []
    for it in items:
        if visible(_chunk(it), principal, flt):
            out.append(it)
        else:
            logger.error(
                "ACL post-filter removed chunk %s for user %s: store pre-filter is leaking",
                _chunk(it).chunk_id,
                principal.user_id if principal else "-",
            )
    return out


def inherit_acl(*acls: ACL) -> ACL:
    """Merge ACLs along an inheritance chain (space -> folder -> document).

    Allows and denies are unioned; deny still wins at evaluation time.
    """
    out = ACL()
    for a in acls:
        out.public = out.public or a.public
        out.allow_users = sorted({*out.allow_users, *a.allow_users})
        out.allow_groups = sorted({*out.allow_groups, *a.allow_groups})
        out.deny_users = sorted({*out.deny_users, *a.deny_users})
        out.deny_groups = sorted({*out.deny_groups, *a.deny_groups})
    return out


class LeakError(AssertionError):
    """An unauthorised chunk reached the retrieved set (failure code F13)."""

    def __init__(self, principal: Principal, chunk_ids: Sequence[str]) -> None:
        super().__init__(
            f"user {principal.user_id!r} (tenant {principal.tenant_id!r}) retrieved "
            f"{len(chunk_ids)} unauthorised chunk(s): {list(chunk_ids)}"
        )
        self.principal = principal
        self.chunk_ids = list(chunk_ids)


def find_leaks(retrieved: Iterable[ScoredChunk | Chunk], principal: Principal) -> list[str]:
    """Chunk ids in ``retrieved`` that ``principal`` must not see (tenant or ACL)."""
    return [_chunk(it).chunk_id for it in retrieved if not visible(_chunk(it), principal)]


def assert_no_leak(retrieved: Iterable[ScoredChunk | Chunk], principal: Principal) -> None:
    """Raise :class:`LeakError` if any retrieved chunk is not visible to ``principal``."""
    leaks = find_leaks(retrieved, principal)
    if leaks:
        raise LeakError(principal, leaks)


# --------------------------------------------------------------------------- permission matrix


@dataclass(frozen=True, slots=True)
class _Resource:
    doc_id: str
    tenant_id: str
    acl: ACL

    def permits(self, principal: Principal) -> bool:
        return self.tenant_id == principal.tenant_id and self.acl.permits(principal)


@dataclass(slots=True)
class MatrixViolation:
    user_id: str
    doc_id: str
    kind: str  # "leak" (unauthorised doc retrieved) | "missing" (authorised doc not retrievable)
    chunk_ids: list[str] = field(default_factory=list)


RetrieveFn = Callable[[Principal], Awaitable[Iterable[ScoredChunk | Chunk]]]


class AclMatrix:
    """Users x documents permission matrix for ACL tests (docs/13 §3, "Tests").

    Build it from principals and documents/chunks, read :meth:`expected` for the
    ground truth, then :meth:`check` a retrieval function that returns *everything*
    a user can retrieve (e.g. a wildcard query with a large top-K). Leaks are always
    violations; with ``require_complete=True`` so are authorised documents that
    never come back (over-filtering).
    """

    def __init__(
        self, principals: Iterable[Principal], resources: Iterable[Document | Chunk]
    ) -> None:
        self.principals = list(principals)
        res: dict[str, _Resource] = {}
        for r in resources:
            res.setdefault(r.doc_id, _Resource(r.doc_id, r.tenant_id, r.acl))
        self.resources = res

    def expected(self) -> dict[tuple[str, str], bool]:
        """``(user_id, doc_id) -> allowed`` for every cell of the matrix."""
        return {
            (p.user_id, d): r.permits(p) for p in self.principals for d, r in self.resources.items()
        }

    def allowed_docs(self, principal: Principal) -> set[str]:
        return {d for d, r in self.resources.items() if r.permits(principal)}

    def render(self) -> str:
        """Human-readable matrix (``X`` = allowed) for test failure messages."""
        docs = sorted(self.resources)
        width = max([len(p.user_id) for p in self.principals] + [4])
        lines = [" " * width + " | " + " ".join(docs)]
        for p in self.principals:
            allowed = self.allowed_docs(p)
            cells = " ".join(("X" if d in allowed else ".").center(len(d)) for d in docs)
            lines.append(p.user_id.ljust(width) + " | " + cells)
        return "\n".join(lines)

    async def check(
        self, retrieve: RetrieveFn, *, require_complete: bool = False
    ) -> list[MatrixViolation]:
        violations: list[MatrixViolation] = []
        for p in self.principals:
            got = list(await retrieve(p))
            allowed = self.allowed_docs(p)
            by_doc: dict[str, list[str]] = {}
            for it in got:
                ch = _chunk(it)
                by_doc.setdefault(ch.doc_id, []).append(ch.chunk_id)
            # A chunk-level check catches per-chunk ACL drift even inside allowed docs.
            chunk_leaks = set(find_leaks(got, p))
            for doc_id, ids in by_doc.items():
                leaked = [i for i in ids if i in chunk_leaks]
                if doc_id not in allowed or leaked:
                    violations.append(MatrixViolation(p.user_id, doc_id, "leak", leaked or ids))
            if require_complete:
                for doc_id in sorted(allowed - set(by_doc)):
                    violations.append(MatrixViolation(p.user_id, doc_id, "missing"))
        return violations

    async def assert_ok(self, retrieve: RetrieveFn, *, require_complete: bool = False) -> None:
        violations = await self.check(retrieve, require_complete=require_complete)
        if violations:
            detail = "\n".join(f"  {v.kind}: {v.user_id} -> {v.doc_id} {v.chunk_ids}" for v in violations)
            raise AssertionError(f"ACL matrix violations:\n{detail}\nexpected:\n{self.render()}")
