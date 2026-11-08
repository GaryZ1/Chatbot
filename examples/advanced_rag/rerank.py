"""Second-stage reranking: retrieve a wide pool cheaply, then reorder it with a stronger
model. Two flavours: a cross-encoder (Nogueira & Cho, 2019) and an LLM listwise reranker
with a sliding window (RankGPT, Sun et al., 2023)."""

import json

from rag_kit import Chunk, Retriever, get_llm, overlap, parse_json, show


def cross_encoder_rerank(query: str, cands: list[Chunk], top_k: int = 3) -> list[Chunk]:
    try:  # pip install sentence-transformers
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("BAAI/bge-reranker-v2-m3")
        scores = model.predict([(query, c.text) for c in cands])
    except ImportError:  # offline stand-in: lexical overlap
        scores = [overlap(query, c.text) for c in cands]
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    return [cands[i] for i in order[:top_k]]


def llm_listwise_rerank(query: str, cands: list[Chunk], window: int = 6, step: int = 3,
                        top_k: int = 3) -> list[Chunk]:
    """RankGPT: ask the LLM to permute a window of passages; slide the window from the
    bottom of the list to the top so strong passages bubble up."""
    llm, ranked = get_llm(), list(cands)
    start = max(len(ranked) - window, 0)
    while True:
        win = ranked[start : start + window]
        passages = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(win))
        prompt = (f"<question>{query}</question>\n<passages>\n{passages}\n</passages>\n"
                  "Rank the passages by relevance to the question. Return only a JSON list "
                  "of passage numbers, most relevant first.")
        default = sorted(range(len(win)), key=lambda i: -overlap(query, win[i].text))
        perm = parse_json(llm(prompt, task="rank"), default)
        if not isinstance(perm, list):
            perm = default
        perm = [i for i in perm if isinstance(i, int) and 0 <= i < len(win)]
        perm += [i for i in range(len(win)) if i not in perm]  # never drop a passage
        ranked[start : start + window] = [win[i] for i in perm]
        if start == 0:
            return ranked[:top_k]
        start = max(start - step, 0)


def demo() -> None:
    q = "Who leads Orbital Systems?"
    pool = Retriever().search(q, k=10)
    show("Cross-encoder", [c.id for c in cross_encoder_rerank(q, pool)])
    show("LLM listwise", json.dumps([c.id for c in llm_listwise_rerank(q, pool)]))


if __name__ == "__main__":
    demo()
