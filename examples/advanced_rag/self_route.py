"""RAG vs long context. Self-Route (Li et al., Google DeepMind, 2024): try cheap RAG first
and let the model declare the question unanswerable from the retrieved chunks; only those
queries go to the expensive long-context call over everything. LongRAG (Jiang et al.,
2024): use LONG retrieval units (whole documents or groups of related documents, ~4k+
tokens) so a strong long-context reader gets complete evidence from few units."""

from collections import defaultdict

from rag_kit import CORPUS, Chunk, DenseIndex, Retriever, format_context, get_llm, show

SELF_ROUTE = """<context>
{ctx}
</context>
<question>{q}</question>
Answer from the context. If the context does not contain the answer, reply exactly
UNANSWERABLE."""


def self_route(question: str) -> dict:
    llm = get_llm()
    hits = Retriever().search(question, k=2)
    out = llm(SELF_ROUTE.format(ctx=format_context(hits), q=question), task="answer")
    if "UNANSWERABLE" not in out and "don't know" not in out.lower():
        return {"route": "rag", "answer": out}
    everything = [Chunk(d.id, d.id, d.title, d.text) for d in CORPUS]   # long-context path
    return {"route": "long_context",
            "answer": llm(SELF_ROUTE.format(ctx=format_context(everything), q=question),
                          task="answer")}


def long_retrieval_units(max_chars: int = 600) -> list[Chunk]:
    """LongRAG grouping: pack documents that share entities/type into long units."""
    groups = defaultdict(list)
    for d in CORPUS:
        groups[d.meta["type"]].append(d)
    units = []
    for _kind, docs in groups.items():
        buf = []
        for d in docs:
            if buf and sum(len(x.text) for x in buf) + len(d.text) > max_chars:
                units.append(buf)
                buf = []
            buf.append(d)
        units.append(buf)
    return [Chunk(f"unit{i}", "+".join(d.id for d in u), " / ".join(d.title for d in u),
                  " ".join(d.text for d in u)) for i, u in enumerate(units)]


def longrag(question: str, k: int = 2) -> dict:
    units = long_retrieval_units()
    top = [units[i] for i in DenseIndex([u.text for u in units]).scores(question)
           .argsort()[::-1][:k]]
    return {"units": [u.title for u in top],
            "answer": get_llm()(f"<context>{format_context(top)}</context>\n"
                                f"<question>{question}</question>", task="answer")}


def demo() -> None:
    show("Self-Route (answerable)", self_route("How much does the ST-2 weigh?"))
    show("Self-Route (falls back)", self_route("What is the dress code?"))
    show("LongRAG", longrag("Who are the people leading Orbital Systems?"))


if __name__ == "__main__":
    demo()
