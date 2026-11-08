"""REPLUG (Shi et al., 2023): treat the LM as a black box; run it once per retrieved
document and ENSEMBLE the outputs weighted by retrieval scores (softmax over similarity).
The paper ensembles next-token probabilities; with an API you ensemble candidate answers.
REPLUG-LSR additionally trains the retriever with the LM's preferences."""

from collections import defaultdict

import numpy as np
from rag_kit import DenseIndex, chunk_docs, generate_answer, get_llm, show


def replug(question: str, k: int = 4, temperature: float = 0.1) -> dict:
    llm, chunks = get_llm(), chunk_docs()
    scores = DenseIndex([c.text for c in chunks]).scores(question)
    top = scores.argsort()[::-1][:k]
    w = np.exp(scores[top] / temperature)
    w /= w.sum()                                              # retrieval-score weights
    tally: dict[str, float] = defaultdict(float)
    for i, weight in zip(top, w, strict=True):
        answer = generate_answer(llm, question, [chunks[i]])  # one pass per document
        tally[answer.split(" [")[0].strip()] += float(weight)
    best = max(tally, key=tally.get)
    return {"answer": best, "distribution": {a: round(p, 3) for a, p in tally.items()}}


def demo() -> None:
    show("REPLUG", replug("Where is Orbital Systems headquartered?"))


if __name__ == "__main__":
    demo()
