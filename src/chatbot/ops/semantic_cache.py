"""Permission-scoped semantic answer cache (docs/14-production-llmops.md §2; docs/13 §3).

Where semantic caching becomes a security bug: an answer generated from documents
user A may read is served to user B who may not. This cache therefore

* partitions entries by ``Principal.scope_key()`` (tenant + user + groups) **and** the
  corpus/index version, and never looks outside the caller's partition;
* records which documents each answer used, so a document update or deletion
  invalidates every answer built from it (:meth:`SemanticCache.invalidate_doc`);
* uses a high similarity threshold (0.95) for the embedding-based hit, after an exact
  normalised-query hit, plus a TTL;
* only stores fully answered responses (never abstentions, blocks or errors).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from chatbot.core.interfaces import Embedder
from chatbot.core.types import Answer, AnswerStatus, Principal

logger = logging.getLogger(__name__)

Clock = Callable[[], float]

_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_query(q: str) -> str:
    """Exact-hit key: NFKC, case-folded, punctuation-free, whitespace-collapsed."""
    t = unicodedata.normalize("NFKC", q).casefold()
    t = _PUNCT_RE.sub(" ", t)
    return " ".join(t.split())


@dataclass(slots=True)
class CacheEntry:
    key: str
    scope_key: str
    index_version: str
    query: str
    norm_query: str
    answer: Answer
    doc_ids: frozenset[str]
    created_at: float
    embedding: np.ndarray | None = None
    hits: int = 0


@dataclass(slots=True)
class CacheHit:
    answer: Answer
    kind: str  # "exact" | "semantic"
    similarity: float
    cached_query: str


@dataclass(slots=True)
class CacheStats:
    hits_exact: int = 0
    hits_semantic: int = 0
    misses: int = 0
    stores: int = 0
    invalidations: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        total = self.hits_exact + self.hits_semantic + self.misses
        return (self.hits_exact + self.hits_semantic) / total if total else 0.0


class SemanticCache:
    """In-process semantic cache; ``embedder=None`` gives an exact-match-only cache."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        threshold: float = 0.95,
        ttl_s: float = 24 * 3600.0,
        max_entries: int = 10_000,
        clock: Clock = time.monotonic,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if ttl_s <= 0 or max_entries <= 0:
            raise ValueError("ttl_s and max_entries must be positive")
        self.embedder = embedder
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        #: (scope, index_version) -> {norm_query: entry key}
        self._exact: dict[tuple[str, str], dict[str, str]] = {}
        self._by_doc: dict[str, set[str]] = {}
        self._lock = asyncio.Lock()
        self.stats = CacheStats()

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _partition(principal: Principal, index_version: str) -> tuple[str, str]:
        return principal.scope_key(), index_version

    def _remove(self, key: str) -> None:
        e = self._entries.pop(key, None)
        if e is None:
            return
        part = self._exact.get((e.scope_key, e.index_version))
        if part is not None and part.get(e.norm_query) == key:
            del part[e.norm_query]
            if not part:
                del self._exact[(e.scope_key, e.index_version)]
        for d in e.doc_ids:
            keys = self._by_doc.get(d)
            if keys is not None:
                keys.discard(key)
                if not keys:
                    del self._by_doc[d]

    def _expired(self, e: CacheEntry, now: float) -> bool:
        return now - e.created_at >= self.ttl_s

    async def _embed(self, query: str) -> np.ndarray | None:
        if self.embedder is None:
            return None
        try:
            v = np.asarray(await self.embedder.embed_query(query), dtype=np.float32).reshape(-1)
        except Exception:
            logger.exception("semantic cache: query embedding failed; exact match only")
            return None
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    # ------------------------------------------------------------------ API

    async def get(self, query: str, principal: Principal, index_version: str) -> CacheHit | None:
        """Look up within the caller's permission scope and the current index version only."""
        now = self._clock()
        scope, ver = self._partition(principal, index_version)
        norm = normalize_query(query)
        async with self._lock:
            key = self._exact.get((scope, ver), {}).get(norm)
            if key is not None:
                e = self._entries[key]
                if self._expired(e, now):
                    self._remove(key)
                else:
                    e.hits += 1
                    self._entries.move_to_end(key)
                    self.stats.hits_exact += 1
                    return CacheHit(e.answer.model_copy(deep=True), "exact", 1.0, e.query)
        vec = await self._embed(query)
        async with self._lock:
            if vec is not None:
                best: tuple[float, CacheEntry] | None = None
                # Only the caller's partition is ever scanned: no cross-scope candidates.
                for key in list(self._exact.get((scope, ver), {}).values()):
                    e = self._entries[key]
                    if e.embedding is None:
                        continue
                    if self._expired(e, now):
                        self._remove(key)
                        continue
                    if e.embedding.shape != vec.shape:
                        continue
                    sim = float(np.dot(vec, e.embedding))
                    if best is None or sim > best[0]:
                        best = (sim, e)
                if best is not None and best[0] >= self.threshold:
                    sim, e = best
                    e.hits += 1
                    self._entries.move_to_end(e.key)
                    self.stats.hits_semantic += 1
                    return CacheHit(e.answer.model_copy(deep=True), "semantic", sim, e.query)
            self.stats.misses += 1
            return None

    def _reject(self, reason: str) -> bool:
        self.stats.rejected[reason] = self.stats.rejected.get(reason, 0) + 1
        return False

    async def put(
        self,
        query: str,
        principal: Principal,
        index_version: str,
        answer: Answer,
        *,
        doc_ids: Iterable[str] | None = None,
    ) -> bool:
        """Store an answer. Returns False when the answer is not cacheable.

        ``doc_ids`` are the documents the answer was built from (all context docs, not
        only cited ones); defaults to the cited documents.
        """
        if answer.status is not AnswerStatus.ANSWERED:
            return self._reject(f"status:{answer.status.value}")
        if answer.guardrail_flags:
            return self._reject("guardrail_flags")
        docs = frozenset(doc_ids) if doc_ids is not None else frozenset(c.doc_id for c in answer.citations)
        if not docs:
            return self._reject("no_documents")
        vec = await self._embed(query)
        scope, ver = self._partition(principal, index_version)
        norm = normalize_query(query)
        key = f"{scope}:{ver}:{norm}"
        async with self._lock:
            self._remove(key)
            e = CacheEntry(
                key=key,
                scope_key=scope,
                index_version=ver,
                query=query,
                norm_query=norm,
                answer=answer.model_copy(deep=True),
                doc_ids=docs,
                created_at=self._clock(),
                embedding=vec,
            )
            self._entries[key] = e
            self._exact.setdefault((scope, ver), {})[norm] = key
            for d in docs:
                self._by_doc.setdefault(d, set()).add(key)
            while len(self._entries) > self.max_entries:
                self._remove(next(iter(self._entries)))
            self.stats.stores += 1
        return True

    async def invalidate_doc(self, doc_id: str) -> int:
        """Drop every cached answer built from ``doc_id`` (call on update / delete / ACL change)."""
        async with self._lock:
            keys = list(self._by_doc.get(doc_id, ()))
            for k in keys:
                self._remove(k)
            self.stats.invalidations += len(keys)
        if keys:
            logger.info("semantic cache: invalidated %d entries for doc %s", len(keys), doc_id)
        return len(keys)

    async def invalidate_index_version(self, version: str | None = None, *, keep: str | None = None) -> int:
        """Drop entries of index ``version``, or of every version except ``keep``."""
        if (version is None) == (keep is None):
            raise ValueError("pass exactly one of version / keep")
        async with self._lock:
            keys = [
                k
                for k, e in self._entries.items()
                if (version is not None and e.index_version == version)
                or (keep is not None and e.index_version != keep)
            ]
            for k in keys:
                self._remove(k)
            self.stats.invalidations += len(keys)
        return len(keys)

    async def invalidate_scope(self, principal: Principal) -> int:
        """Drop a principal's entries (e.g. after their group memberships changed)."""
        scope = principal.scope_key()
        async with self._lock:
            keys = [k for k, e in self._entries.items() if e.scope_key == scope]
            for k in keys:
                self._remove(k)
        return len(keys)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._exact.clear()
            self._by_doc.clear()

    def __len__(self) -> int:
        return len(self._entries)
