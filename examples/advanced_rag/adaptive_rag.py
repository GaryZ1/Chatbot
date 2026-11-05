"""Adaptive-RAG (Jeong et al., 2024): a cheap classifier predicts query complexity and
routes it: A = answer directly (no retrieval), B = single-step RAG, C = multi-step
iterative RAG. Spend expensive compute only where it pays."""

from ircot import ircot
from rag_kit import Retriever, generate_answer, get_llm, parse_json, show

PROMPT = """Classify the question's complexity:
A = answerable without any documents (greetings, general knowledge, arithmetic);
B = needs one lookup in the company knowledge base;
C = needs several chained lookups (multi-hop reasoning).
<question>{q}</question>
Return JSON {{"label": "A" | "B" | "C"}}."""


def adaptive_rag(question: str) -> dict:
    llm = get_llm()
    label = parse_json(llm(PROMPT.format(q=question), task="classify"), {}).get("label", "B")
    if label == "A":
        answer = llm(f"<question>{question}</question>", task="direct")
    elif label == "C":
        answer = ircot(question)["answer"]
    else:
        answer = generate_answer(llm, question, Retriever().search(question, k=3))
    return {"route": label, "answer": answer}


def demo() -> None:
    for q in ["hello there",
              "How long do refunds take?",
              "Where did the CTO of the company that bought Lumen Labs work before?"]:
        show(q, adaptive_rag(q))


if __name__ == "__main__":
    demo()
