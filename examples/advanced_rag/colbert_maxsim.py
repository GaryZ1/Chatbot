"""Late interaction (ColBERT, Khattab & Zaharia 2020; ColBERTv2; Jina-ColBERT; PyLate).
Keep one vector per token; score = sum over query tokens of the max similarity to any
document token (MaxSim). Better at fine-grained matching than one pooled vector."""

import numpy as np
from rag_kit import HashingEmbedder, chunk_docs, show

emb = HashingEmbedder(dim=256)


def maxsim(q_tokens: np.ndarray, d_tokens: np.ndarray) -> float:
    sim = q_tokens @ d_tokens.T            # [n_query_tokens, n_doc_tokens]
    return float(sim.max(axis=1).sum())    # best doc token for each query token, summed


def colbert_search(query: str, k: int = 3):
    chunks = chunk_docs()
    doc_vectors = [emb.token_vectors(c.text) for c in chunks]  # offline: store all tokens
    q = emb.token_vectors(query)
    scored = sorted(((maxsim(q, d), c.id) for d, c in zip(doc_vectors, chunks, strict=True)),
                    reverse=True)
    return [(cid, round(s, 2)) for s, cid in scored[:k]]


def demo() -> None:
    # Production: from pylate import models; models.ColBERT("lightonai/GTE-ModernColBERT-v1")
    show("ColBERT MaxSim", colbert_search("star tracker weight grams"))


if __name__ == "__main__":
    demo()
