"""MemoRAG (Qian et al., 2024): a light "memory" model first forms a global, compressed
memory of the whole corpus. For a question, it recalls DRAFT CLUES (likely answer
fragments, sub-questions) from that memory; the clues, not just the raw question, drive
retrieval. Helps with implicit or vague information needs that a query alone cannot
express. A strong generator then answers from the retrieved evidence."""

from rag_kit import CORPUS, Retriever, generate_answer, get_llm, parse_json, rrf, show


def build_memory() -> str:
    """Global memory: a compressed view of the corpus (the paper uses a long-context
    model's KV cache; a running summary is the API-friendly approximation)."""
    docs = "\n".join(f"<document>{d.title}: {d.text}</document>" for d in CORPUS)
    return get_llm()(f"{docs}\nCompress these documents into a dense memory of the key "
                     "entities, events and facts.", task="summarize")


def memorag(question: str, memory: str, k: int = 3) -> dict:
    llm, r = get_llm(), Retriever()
    clues = parse_json(llm(f"<memory>{memory}</memory>\n<question>{question}</question>\n"
                           "From your memory of the corpus, write 2-4 clues (draft answer "
                           "fragments or specific search queries) that would locate the "
                           "evidence. Return a JSON list of strings.", task="memory_clues"),
                       [question])
    ranking = rrf([[c.id for c in r.search(clue, k=5)] for clue in [question, *clues]])
    evidence = [r.by_id[cid] for cid, _ in ranking[:k]]
    return {"clues": clues, "evidence": [c.id for c in evidence],
            "answer": generate_answer(llm, question, evidence)}


def demo() -> None:
    memory = build_memory()
    show("Memory", memory[:200] + "...")
    show("MemoRAG", memorag("What progress did the company make on energy use?", memory))


if __name__ == "__main__":
    demo()
