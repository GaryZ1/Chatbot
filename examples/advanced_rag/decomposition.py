"""Query decomposition (sub-question / least-to-most, Zhou et al., 2022; LlamaIndex
SubQuestionQueryEngine): split a compound question into independent sub-questions, answer
each with its own retrieval, then synthesise."""

from rag_kit import Retriever, generate_answer, get_llm, parse_json, show

PROMPT = """<question>{q}</question>
Break the question into the minimal list of independent sub-questions that must be answered
to answer it. Return a JSON list of strings."""


def decompose_and_answer(question: str) -> dict:
    llm, r = get_llm(), Retriever()
    subs = parse_json(llm(PROMPT.format(q=question), task="decompose"), [question])
    partials, evidence = [], {}
    for sq in subs:
        hits = r.search(sq, k=2)
        evidence.update({c.id: c for c in hits})
        partials.append(f"<sub_answer question='{sq}'>{generate_answer(llm, sq, hits)}"
                        "</sub_answer>")
    final = generate_answer(llm, question, list(evidence.values()), extra="\n".join(partials))
    return {"sub_questions": subs, "answer": final}


def demo() -> None:
    show("Decomposition",
         decompose_and_answer("Where is Orbital Systems headquartered and who is its CEO?"))


if __name__ == "__main__":
    demo()
