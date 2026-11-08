"""Multi-query retrieval + RAG-Fusion (Raudaschl, 2023): generate several reformulations,
retrieve for each in parallel, and fuse the ranked lists with RRF. Improves recall on
ambiguous or under-specified queries."""

from concurrent.futures import ThreadPoolExecutor

from rag_kit import Retriever, generate_answer, get_llm, parse_json, rrf, show

PROMPT = """<question>{q}</question>
Generate 3 different search queries that together cover the likely intents of the question
(synonyms, more specific, more general). Return a JSON list of strings."""


def rag_fusion(question: str, k: int = 4) -> dict:
    llm, r = get_llm(), Retriever()
    queries = parse_json(llm(PROMPT.format(q=question), task="queries"), [])
    queries = [question, *[q for q in queries if isinstance(q, str) and q != question]]
    with ThreadPoolExecutor() as pool:
        rankings = list(pool.map(lambda q: [c.id for c in r.search(q, k=10)], queries))
    fused = [r.by_id[cid] for cid, _ in rrf(rankings)[:k]]
    return {"queries": queries, "hits": [c.id for c in fused],
            "answer": generate_answer(llm, question, fused)}


def demo() -> None:
    show("RAG-Fusion", rag_fusion("Skyline energy and environmental impact"))


if __name__ == "__main__":
    demo()
