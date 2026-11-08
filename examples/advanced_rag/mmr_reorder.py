"""Context selection: Maximal Marginal Relevance (Carbonell & Goldstein, 1998) removes
near-duplicate chunks, and "lost-in-the-middle" reordering (Liu et al., 2023) puts the
strongest evidence at the start and end of the context, where models attend best."""

import numpy as np
from rag_kit import Chunk, HashingEmbedder, Retriever, show


def mmr(query: str, cands: list[Chunk], k: int = 4, lam: float = 0.7) -> list[Chunk]:
    emb = HashingEmbedder()
    q, d = emb.embed_one(query), emb.embed(c.text for c in cands)
    rel, chosen = d @ q, []
    while len(chosen) < min(k, len(cands)):
        redundancy = (d @ d[chosen].T).max(1) if chosen else np.zeros(len(cands))
        score = lam * rel - (1 - lam) * redundancy
        score[chosen] = -np.inf
        chosen.append(int(score.argmax()))
    return [cands[i] for i in chosen]


def lost_in_the_middle_order(ranked: list[Chunk]) -> list[Chunk]:
    """[1,2,3,4,5] -> [1,3,5,4,2]: best first, second-best last, weakest in the middle."""
    return ranked[::2] + ranked[1::2][::-1]


def demo() -> None:
    q = "Skyline product features"
    pool = Retriever().search(q, k=10)
    picked = mmr(q, pool)
    show("MMR selection", [c.id for c in picked])
    show("Reordered for the prompt", [c.id for c in lost_in_the_middle_order(picked)])


if __name__ == "__main__":
    demo()
