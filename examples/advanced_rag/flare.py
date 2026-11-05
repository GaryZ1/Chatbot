"""FLARE, Forward-Looking Active REtrieval (Jiang et al., 2023): generate the answer one
sentence at a time; when the upcoming sentence is low-confidence, use it as a query to
retrieve fresh evidence and regenerate it. The paper reads token probabilities; with APIs
that do not expose them, ask for a calibrated self-check (as here) or check support against
the current context. DRAGIN (Su et al., 2024) refines *when* and *what* to retrieve."""

from rag_kit import Retriever, format_context, get_llm, parse_json, show

NEXT = """<context>
{context}
</context>
<question>{q}</question>
<answer_so_far>{so_far}</answer_so_far>
Write the next sentence of the answer, or END if the answer is complete."""
CHECK = """<context>
{context}
</context>
<sentence>{sentence}</sentence>
Is every factual claim in the sentence directly supported by the context?
Return JSON {{"confident": true | false}}."""


def flare(question: str, max_sentences: int = 4) -> dict:
    llm, r = get_llm(), Retriever()
    context, answer, retrievals = r.search(question, k=2), [], 1
    for _ in range(max_sentences):
        ask = NEXT.format(context=format_context(context), q=question, so_far=" ".join(answer))
        sentence = llm(ask, task="next_sentence").strip()
        if sentence == "END" or not sentence:
            break
        check = CHECK.format(context=format_context(context), sentence=sentence)
        if not parse_json(llm(check, task="confidence"), {}).get("confident", False):
            context = r.search(f"{question} {sentence}", k=2)        # look-ahead query
            retrievals += 1
            ask = NEXT.format(context=format_context(context), q=question,
                              so_far=" ".join(answer))
            sentence = llm(ask, task="next_sentence").strip()
            if sentence == "END":
                break
        answer.append(sentence)
    return {"answer": " ".join(answer), "retrievals": retrievals}


def demo() -> None:
    show("FLARE", flare("Describe the Lumen ST-2 star tracker."))


if __name__ == "__main__":
    demo()
