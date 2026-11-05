"""ITER-RETGEN, iterative retrieval-generation synergy (Shao et al., 2023): retrieve with
question + the previous draft answer, regenerate, repeat for T rounds. The draft surfaces
terms (bridge entities) the original question lacked."""

from rag_kit import Retriever, generate_answer, get_llm, show


def iter_retgen(question: str, rounds: int = 3, k: int = 3) -> list[dict]:
    llm, r, draft, history = get_llm(), Retriever(), "", []
    for t in range(rounds):
        hits = r.search(f"{question} {draft}".strip(), k=k)
        draft = generate_answer(llm, question, hits)
        history.append({"round": t + 1, "hits": [c.id for c in hits], "draft": draft})
    return history


def demo() -> None:
    show("ITER-RETGEN", iter_retgen("Who is the CEO, and which company did they work for "
                                    "before Orbital Systems?"))


if __name__ == "__main__":
    demo()
