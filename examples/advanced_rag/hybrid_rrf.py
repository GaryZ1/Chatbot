"""Hybrid retrieval: BM25 (exact terms, IDs, rare words) + dense (paraphrase, semantics),
fused with Reciprocal Rank Fusion (Cormack et al., 2009) or a convex score combination."""

import numpy as np
from rag_kit import BM25, DenseIndex, chunk_docs, rrf, show


def minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span else np.zeros_like(x)


def hybrid_search(query: str, k: int = 4, alpha: float = 0.5, method: str = "rrf"):
    chunks = chunk_docs()
    texts = [c.text for c in chunks]
    sparse, dense = BM25(texts).scores(query), DenseIndex(texts).scores(query)
    if method == "rrf":                                   # rank-based: no score calibration
        ranks = [[chunks[i].id for i in np.argsort(-s)[:20]] for s in (sparse, dense)]
        return rrf(ranks, k=60)[:k]
    fused = alpha * minmax(dense) + (1 - alpha) * minmax(sparse)  # convex combination
    return [(chunks[i].id, round(float(fused[i]), 3)) for i in np.argsort(-fused)[:k]]


def demo() -> None:
    q = "ST-2 power consumption"
    show("Hybrid (RRF)", hybrid_search(q))
    show("Hybrid (convex, alpha=0.3)", hybrid_search(q, alpha=0.3, method="convex"))


if __name__ == "__main__":
    demo()
