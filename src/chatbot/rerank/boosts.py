"""Business priors after semantic reranking (docs/08-reranking.md §3 rule 6).

``final = ce + β·recency + γ·authority (+ δ·is_latest)`` — applied *after* the
semantic reranker so priors only reorder near-ties. Uses the same freshness and
authority functions as retrieval boosting (docs/07 §3).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from chatbot.core.interfaces import Reranker
from chatbot.core.types import ScoredChunk
from chatbot.retrieve.expand import apply_boosts


class BusinessPriorReranker:
    """Wraps a semantic reranker and adds priors to its scores."""

    def __init__(
        self,
        inner: Reranker,
        *,
        recency_weight: float = 0.05,
        authority_weight: float = 0.05,
        latest_weight: float = 0.0,
        half_life_days: float = 180.0,
        authority_by_doc_type: Mapping[str, float] | None = None,
        prior_pool: int = 3,
        now: datetime | None = None,
        name: str = "priors",
    ) -> None:
        self.inner = inner
        self.recency_weight = recency_weight
        self.authority_weight = authority_weight
        self.latest_weight = latest_weight
        self.half_life_days = half_life_days
        self.authority_by_doc_type = dict(authority_by_doc_type or {})
        self.prior_pool = max(1, prior_pool)
        self.now = now
        self.name = name

    def boost(self, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        return apply_boosts(
            hits,
            freshness_weight=self.recency_weight,
            authority_weight=self.authority_weight,
            latest_weight=self.latest_weight,
            now=self.now,
            half_life_days=self.half_life_days,
            authority_by_doc_type=self.authority_by_doc_type,
            stage=self.name,
        )

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if top_k <= 0:
            return []
        # Rerank a slightly larger pool so priors can promote a hit just below the cut.
        ranked = await self.inner.rerank(query, candidates, top_k * self.prior_pool)
        return self.boost(ranked)[:top_k]
