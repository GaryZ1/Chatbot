"""Sufficient-context gating (Joren et al., Google, ICLR 2025, "Sufficient Context"): an
autorater decides whether the retrieved context is ENOUGH to answer; combine that signal
with the model's self-rated confidence to abstain selectively. RAG often makes models
answer confidently from insufficient context; this gate cuts those hallucinations."""

from rag_kit import Retriever, format_context, generate_answer, get_llm, parse_json, show

RATER = """<context>{ctx}</context>
<question>{q}</question>
Does the context contain enough information to definitively answer the question? Think
about what an answer would require, then return JSON {{"verdict": "yes"|"no",
"score": 0-1}}."""


def answer_or_abstain(question: str, threshold: float = 0.5) -> dict:
    llm = get_llm()
    hits = Retriever().search(question, k=3)
    rating = parse_json(llm(RATER.format(ctx=format_context(hits), q=question), task="sufficient"),
                        {})
    sufficient = rating.get("verdict") == "yes" and rating.get("score", 0) >= threshold
    if not sufficient:
        return {"sufficient": False, "answer": "I don't have enough information to answer."}
    return {"sufficient": True, "answer": generate_answer(llm, question, hits)}


def demo() -> None:
    for q in ["How many operators use Skyline?", "What is the dress code at Orbital Systems?"]:
        show(q, answer_or_abstain(q))


if __name__ == "__main__":
    demo()
