"""CRAG, Corrective RAG (Yan et al., 2024): a lightweight evaluator grades the retrieved
documents. CORRECT -> refine them (decompose into knowledge strips, keep relevant strips);
INCORRECT -> discard and fall back to another source (web search in the paper; broader
internal search, a clarifying question or abstention in enterprise); AMBIGUOUS -> both."""

from rag_kit import Chunk, Retriever, generate_answer, get_llm, parse_json, show, split_sentences

GRADE = """<question>{q}</question><document>{doc}</document>
How relevant is the document for answering the question?
Return JSON {{"relevant": bool, "score": 0-1}}."""
UPPER, LOWER = 0.6, 0.3


def grade(llm, q: str, text: str) -> float:
    return float(parse_json(llm(GRADE.format(q=q, doc=text), task="grade"), {}).get("score", 0))


def refine(llm, q: str, docs: list[Chunk]) -> list[Chunk]:
    """Decompose-then-recompose: split into sentence strips, keep those that grade well."""
    strips = [Chunk(f"{d.id}~{i}", d.doc_id, d.title, s) for d in docs
              for i, s in enumerate(split_sentences(d.text))]
    return [s for s in strips if grade(llm, q, s.text) >= LOWER] or docs


def fallback_search(llm, q: str) -> list[Chunk]:
    """Stand-in for web search: rewrite the query and search a wider pool."""
    query = llm(f"<question>{q}</question>\nRewrite as a web search query.", task="rewrite")
    return Retriever().search(query, k=6, mode="sparse")


def crag(question: str, k: int = 3) -> dict:
    llm = get_llm()
    docs = Retriever().search(question, k=k)
    best = max((grade(llm, question, d.text) for d in docs), default=0.0)
    if best >= UPPER:
        action, knowledge = "CORRECT", refine(llm, question, docs)
    elif best < LOWER:
        action, knowledge = "INCORRECT", fallback_search(llm, question)
    else:
        action = "AMBIGUOUS"
        knowledge = refine(llm, question, docs) + fallback_search(llm, question)
    return {"action": action, "best_grade": best,
            "answer": generate_answer(llm, question, knowledge)}


def demo() -> None:
    for q in ["How long do customers have to request a refund?", "What is the dress code?"]:
        show(q, crag(q))


if __name__ == "__main__":
    demo()
