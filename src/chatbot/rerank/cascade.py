"""Reranker cascade (docs/08-reranking.md §3 rule 4).

Cheap reranker on N candidates → expensive reranker on the top M only, optionally
gated by a router predicate ("hard query"). If the expensive stage fails, the cheap
ranking is returned (graceful degradation).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from chatbot.core.interfaces import Reranker
from chatbot.core.types import ScoredChunk

log = logging.getLogger(__name__)


class CascadeReranker:
    def __init__(
        self,
        cheap: Reranker,
        expensive: Reranker,
        *,
        expensive_top_m: int = 20,
        use_expensive: Callable[[str, Sequence[ScoredChunk]], bool] | None = None,
        name: str = "cascade",
    ) -> None:
        if expensive_top_m <= 0:
            raise ValueError("expensive_top_m must be positive")
        self.cheap = cheap
        self.expensive = expensive
        self.expensive_top_m = expensive_top_m
        self.use_expensive = use_expensive
        self.name = name

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if top_k <= 0 or not candidates:
            return []
        first = await self.cheap.rerank(query, candidates, max(top_k, self.expensive_top_m))
        if self.use_expensive is not None and not self.use_expensive(query, first):
            return first[:top_k]
        head, tail = first[: self.expensive_top_m], first[self.expensive_top_m :]
        try:
            second = await self.expensive.rerank(query, head, len(head))
        except Exception as exc:
            log.warning(
                "expensive reranker %s failed, using %s ranking: %r",
                self.expensive.name,
                self.cheap.name,
                exc,
            )
            return first[:top_k]
        # Expensive-stage scores are not comparable with cheap ones: the tail stays below.
        return [*second, *tail][:top_k]
