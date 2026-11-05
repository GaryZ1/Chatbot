"""Hybrid retrieval (docs/07-retrieval.md §2 "Hybrid search, done right").

1. Run every leg (BM25, dense, optional question / title index) **in parallel**
   with ``asyncio.gather`` and the **same principal and filter** — each leg
   pre-filters before its own top-K.
2. Fuse with RRF (k=60), weighted RRF, or a normalised convex combination.
3. Dedup by ``chunk_id`` and content hash, return the fused top 50–150 for the
   reranker.

A failing leg is logged and skipped (degraded but available); if all legs fail
the first error is raised. :class:`ParentMappingRetriever` turns a
hypothetical-question index into a leg by mapping each question hit to the chunk
it points to (``parent_id``) — re-checking visibility on the target.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence

from chatbot.core.filters import Filter, visible
from chatbot.core.interfaces import DocStore, Retriever
from chatbot.core.types import ChunkKind, Principal, ScoredChunk
from chatbot.retrieve.fusion import FusionConfig, dedup, fuse

log = logging.getLogger(__name__)


class ParentMappingRetriever:
    """Wraps a retriever over pointer chunks (QUESTION, PROPOSITION, ...) and returns targets.

    Hits whose kind is in ``kinds`` are replaced by ``docstore.get(parent_id)``; the
    best score per target is kept. Other hits pass through unchanged.
    """

    def __init__(
        self,
        inner: Retriever,
        docstore: DocStore,
        *,
        kinds: frozenset[ChunkKind] = frozenset({ChunkKind.QUESTION}),
        name: str | None = None,
        oversample: int = 2,
    ) -> None:
        self.inner = inner
        self.docstore = docstore
        self.kinds = kinds
        self.oversample = max(1, oversample)
        self.name = name or f"{inner.name}->parent"

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0:
            return []
        hits = await self.inner.retrieve(query, k * self.oversample, principal, flt)
        target_ids = [h.chunk.parent_id for h in hits if h.chunk.kind in self.kinds and h.chunk.parent_id]
        targets = {c.chunk_id: c for c in await self.docstore.get(list(dict.fromkeys(target_ids)))}
        best: dict[str, ScoredChunk] = {}
        for h in hits:
            if h.chunk.kind in self.kinds:
                target = targets.get(h.chunk.parent_id or "")
                # The target must be visible in its own right (ACL on the parent).
                if target is None or not visible(target, principal, flt):
                    continue
                cand = ScoredChunk(
                    chunk=target,
                    score=h.score,
                    source=self.name,
                    stage_scores={**h.stage_scores, self.name: h.score},
                )
            else:
                cand = h.with_score(h.score, self.name)
            cur = best.get(cand.chunk.chunk_id)
            if cur is None or cand.score > cur.score:
                best[cand.chunk.chunk_id] = cand
        return sorted(best.values(), key=lambda s: -s.score)[:k]


class HybridRetriever:
    """Parallel multi-leg retrieval + fusion. Itself a :class:`Retriever`."""

    def __init__(
        self,
        legs: Sequence[Retriever],
        *,
        fusion: FusionConfig | None = None,
        leg_k: int = 100,
        dedup_content: bool = True,
        name: str = "hybrid",
        leg_timeout_s: float | None = None,
    ) -> None:
        if not legs:
            raise ValueError("HybridRetriever needs at least one leg")
        names = [leg.name for leg in legs]
        if len(set(names)) != len(names):
            raise ValueError(f"leg names must be unique: {names}")
        if leg_k <= 0:
            raise ValueError("leg_k must be positive")
        self.legs = list(legs)
        self.fusion = fusion or FusionConfig()
        self.leg_k = leg_k
        self.dedup_content = dedup_content
        self.name = name
        self.leg_timeout_s = leg_timeout_s
        self.last_leg_latency_ms: dict[str, float] = {}

    async def _run_leg(
        self, leg: Retriever, query: str, k: int, principal: Principal | None, flt: Filter | None
    ) -> list[ScoredChunk]:
        t0 = time.perf_counter()
        try:
            coro = leg.retrieve(query, k, principal, flt)
            if self.leg_timeout_s is not None:
                return await asyncio.wait_for(coro, self.leg_timeout_s)
            return await coro
        finally:
            self.last_leg_latency_ms[leg.name] = (time.perf_counter() - t0) * 1000

    async def retrieve_legs(
        self, query: str, principal: Principal | None, flt: Filter | None = None, *, leg_k: int | None = None
    ) -> dict[str, list[ScoredChunk]]:
        """Run all legs concurrently; failed legs are omitted from the result."""
        k = leg_k or self.leg_k
        results = await asyncio.gather(
            *(self._run_leg(leg, query, k, principal, flt) for leg in self.legs),
            return_exceptions=True,
        )
        out: dict[str, list[ScoredChunk]] = {}
        errors: list[BaseException] = []
        for leg, res in zip(self.legs, results, strict=True):
            if isinstance(res, BaseException):
                if isinstance(res, asyncio.CancelledError):
                    raise res
                log.warning("retrieval leg %s failed: %r", leg.name, res)
                errors.append(res)
            else:
                out[leg.name] = res
        if not out and errors:
            raise errors[0]
        return out

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not query.strip():
            return []
        legs = await self.retrieve_legs(query, principal, flt, leg_k=max(self.leg_k, k))
        names = list(legs)
        fused = fuse([legs[n] for n in names], self.fusion, names=names)
        fused = dedup(fused, by_content=self.dedup_content)
        return [h.with_score(h.score, self.name) for h in fused[:k]]
