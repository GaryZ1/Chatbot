"""Sparse lexical retrieval leg over BM25 (docs/07-retrieval.md §1)."""

from __future__ import annotations

from chatbot.core.filters import Filter
from chatbot.core.interfaces import SparseIndex
from chatbot.core.types import Principal, ScoredChunk


class SparseRetriever:
    def __init__(self, index: SparseIndex, *, name: str = "bm25") -> None:
        self.index = index
        self.name = name

    async def retrieve(
        self, query: str, k: int, principal: Principal | None, flt: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not query.strip():
            return []
        hits = await self.index.search(query, k, principal, flt)
        return [h.with_score(h.score, self.name) for h in hits]
