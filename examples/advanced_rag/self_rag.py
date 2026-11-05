"""Self-RAG (Asai et al., 2023): the model decides WHETHER to retrieve, judges each passage
(ISREL), generates a candidate per passage, and critiques it (ISSUP: supported?; ISUSE:
useful?). The best-scoring candidate wins. The paper trains reflection tokens into a model;
here the same control flow is driven by prompted judges, the common production approach."""

from rag_kit import Retriever, format_context, generate_answer, get_llm, overlap, parse_json, show

RETRIEVE = """<question>{q}</question>
Does answering this need facts from the company knowledge base? Return JSON
{{"label": "C"}} if yes, {{"label": "A"}} if not."""
ISREL = """<question>{q}</question><document>{doc}</document>
Is the document relevant to the question? Return JSON {{"relevant": bool, "score": 0-1}}."""
ISSUP = """<context>{ctx}</context><claim>{claim}</claim>
Is the claim fully supported by the context? Return JSON {{"verdict": "yes"|"no",
"score": 0-1}}."""


def self_rag(question: str, k: int = 4) -> dict:
    llm = get_llm()
    if parse_json(llm(RETRIEVE.format(q=question), task="classify"), {}).get("label") == "A":
        return {"retrieved": False, "answer": llm(question, task="direct")}
    candidates = []
    for c in Retriever().search(question, k=k):
        rel = parse_json(llm(ISREL.format(q=question, doc=c.text), task="grade"), {})
        if not rel.get("relevant"):
            continue                                                     # ISREL = irrelevant
        answer = generate_answer(llm, question, [c])
        sup = parse_json(llm(ISSUP.format(ctx=format_context([c]), claim=answer),
                             task="judge"), {})
        useful = overlap(question, answer)              # ISUSE (cheap proxy; or a 1-5 judge)
        score = rel.get("score", 0) + 2 * sup.get("score", 0) + 0.5 * useful
        candidates.append({"chunk": c.id, "answer": answer, "score": round(score, 2)})
    if not candidates:
        return {"retrieved": True, "answer": "I don't know.", "candidates": []}
    best = max(candidates, key=lambda x: x["score"])
    return {"retrieved": True, "answer": best["answer"], "candidates": candidates}


def demo() -> None:
    show("Self-RAG", self_rag("How many operators use Skyline?"))


if __name__ == "__main__":
    demo()
