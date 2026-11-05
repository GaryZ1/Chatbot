"""IRCoT, Interleaving Retrieval with Chain-of-Thought (Trivedi et al., 2023): alternate
"reason one step" and "retrieve with that step as the query". Each new CoT sentence names
the bridge entity the next hop needs, so multi-hop evidence is found step by step."""

from rag_kit import Retriever, format_context, get_llm, parse_json, show

PROMPT = """<context>
{context}
</context>
<question>{q}</question>
{steps}
Continue the chain of thought with ONE new sentence that states a fact from the context
and moves toward the answer. Return JSON {{"done": false, "thought": "..."}}, or
{{"done": true, "answer": "..."}} once the answer is known."""


def ircot(question: str, max_steps: int = 4, k: int = 2) -> dict:
    llm, r = get_llm(), Retriever()
    evidence = {c.id: c for c in r.search(question, k=k)}
    thoughts: list[str] = []
    for _ in range(max_steps):
        steps = "\n".join(f"<step>{t}</step>" for t in thoughts)
        out = parse_json(llm(PROMPT.format(context=format_context(list(evidence.values())),
                                           q=question, steps=steps), task="next_step"),
                         {"done": True, "answer": ""})
        if out.get("done"):
            return {"thoughts": thoughts, "answer": out.get("answer", ""),
                    "evidence": list(evidence)}
        thoughts.append(out["thought"])
        for c in r.search(out["thought"], k=k):     # the CoT sentence IS the next query
            evidence.setdefault(c.id, c)
    return {"thoughts": thoughts, "answer": thoughts[-1] if thoughts else "",
            "evidence": list(evidence)}


def demo() -> None:
    show("IRCoT", ircot("Where did the CTO of Orbital Systems work before?"))


if __name__ == "__main__":
    demo()
