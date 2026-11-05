"""Maximal Marginal Relevance (docs/07-retrieval.md §3).

``MMR(d) = λ · sim(q, d) − (1 − λ) · max_{s ∈ selected} sim(d, s)``

λ = 1 is pure relevance, λ = 0 pure diversity; 0.5–0.8 is typical. Diversity can
*hurt* when the answer genuinely needs several near-identical passages (e.g. a
table split into chunks), so keep λ high for factoid routes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chatbot.core.interfaces import Embedder
from chatbot.core.types import ScoredChunk
from chatbot.embed.base import l2_normalize


def mmr_indices(
    query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, *, lambda_: float = 0.7
) -> list[int]:
    """Greedy MMR selection over unit vectors; returns indices into ``doc_vecs``."""
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")
    n = doc_vecs.shape[0]
    if n == 0 or k <= 0:
        return []
    d = l2_normalize(np.asarray(doc_vecs, dtype=np.float32))
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32))
    rel = d @ q
    sim = d @ d.T
    selected: list[int] = []
    max_sim = np.full(n, -np.inf, dtype=np.float32)
    remaining = np.ones(n, dtype=bool)
    for _ in range(min(k, n)):
        redundancy = np.where(np.isfinite(max_sim), max_sim, 0.0)
        scores = lambda_ * rel - (1.0 - lambda_) * redundancy
        scores = np.where(remaining, scores, -np.inf)
        best = int(np.argmax(scores))
        selected.append(best)
        remaining[best] = False
        max_sim = np.maximum(max_sim, sim[best])
    return selected


class MMRSelector:
    """Re-selects candidates with MMR using an embedder for query and candidate vectors."""

    def __init__(self, embedder: Embedder, *, lambda_: float = 0.7, name: str = "mmr") -> None:
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        self.embedder = embedder
        self.lambda_ = lambda_
        self.name = name

    async def select(
        self,
        query: str,
        candidates: Sequence[ScoredChunk],
        k: int,
        *,
        vectors: np.ndarray | None = None,
    ) -> list[ScoredChunk]:
        """Pick ``k`` diverse candidates; pass precomputed ``vectors`` to skip re-embedding."""
        if not candidates or k <= 0:
            return []
        q = await self.embedder.embed_query(query)
        if vectors is None:
            vectors = await self.embedder.embed_documents([c.chunk.index_text for c in candidates])
        if vectors.shape[0] != len(candidates):
            raise ValueError("vectors and candidates differ in length")
        order = mmr_indices(q, vectors, k, lambda_=self.lambda_)
        n = len(order)
        # Score encodes MMR rank so downstream sorting preserves the diverse order.
        return [
            candidates[i].with_score(float(n - rank) / n, self.name) for rank, i in enumerate(order)
        ]

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Reranker-protocol adapter."""
        return await self.select(query, candidates, top_k)
