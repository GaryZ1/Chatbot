"""HyDE, Hypothetical Document Embeddings (Gao et al., 2022): ask the LLM to write a fake
answer passage, embed THAT instead of the question (answer-to-answer similarity). Strong
zero-shot; risky when the LLM confidently hallucinates domain-specific details."""

import numpy as np
from rag_kit import DenseIndex, HashingEmbedder, chunk_docs, get_llm, show

PROMPT = """<question>{q}</question>
Write a short passage that plausibly answers the question, in the style of a company
knowledge-base article. Answer only with the passage."""


def hyde_search(question: str, k: int = 3, n: int = 1, include_query: bool = True):
    llm, emb, chunks = get_llm(), HashingEmbedder(), chunk_docs()
    fakes = [llm(PROMPT.format(q=question), task="hyde") for _ in range(n)]
    vecs = emb.embed(fakes + ([question] if include_query else []))
    q = vecs.mean(0)
    q /= np.linalg.norm(q)                       # average of hypothetical docs (+ query)
    scores = DenseIndex([c.text for c in chunks], emb).matrix @ q
    return fakes[0], [chunks[i].id for i in scores.argsort()[::-1][:k]]


def demo() -> None:
    fake, ids = hyde_search("What happened in the 2022 security incident?")
    show("Hypothetical document", fake)
    show("HyDE hits", ids)


if __name__ == "__main__":
    demo()
