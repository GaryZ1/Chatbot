"""Self-Ask (Press et al., 2022): the model asks itself explicit follow-up questions, each
answered by retrieval ("intermediate answer"), until it can state the final answer.
Compositional questions become a chain of simple lookups."""

from rag_kit import Retriever, format_context, generate_answer, get_llm, parse_json, show

PROMPT = """<question>{q}</question>
<context>
{context}
</context>
{steps}
Are follow-up questions needed? If yes, return JSON {{"done": false, "query": "<one simple
follow-up question>"}}. If the intermediate answers suffice, return JSON
{{"done": true, "answer": "<final answer>"}}."""


def self_ask(question: str, max_hops: int = 4) -> dict:
    llm, r = get_llm(), Retriever()
    trail: list[tuple[str, str]] = []
    context = r.search(question, k=2)
    for _ in range(max_hops):
        steps = "\n".join(f"<step>{a}</step>" for _, a in trail)
        out = parse_json(llm(PROMPT.format(q=question, context=format_context(context),
                                           steps=steps), task="next_step"), {"done": True})
        if out.get("done"):
            return {"follow_ups": trail, "answer": out.get("answer", "")}
        follow_up = out["query"]
        hits = r.search(follow_up, k=2)
        intermediate = generate_answer(llm, follow_up, hits)        # "Intermediate answer:"
        trail.append((follow_up, intermediate))
        context = list({c.id: c for c in context + hits}.values())
    return {"follow_ups": trail, "answer": trail[-1][1] if trail else ""}


def demo() -> None:
    show("Self-Ask", self_ask("In which city is the star tracker of the company acquired by "
                              "Orbital Systems built?"))


if __name__ == "__main__":
    demo()
