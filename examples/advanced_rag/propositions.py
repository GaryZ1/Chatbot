"""Proposition indexing (Dense X Retrieval, Chen et al., 2023): an LLM rewrites each passage
into atomic, self-contained factoids (pronouns resolved). Index propositions, return them
(or their parent passage) at query time. Precise, at the cost of ingest LLM calls."""

from rag_kit import CORPUS, Chunk, Retriever, get_llm, parse_json, show

PROMPT = """Decompose the text into clear, simple propositions. Each proposition must be a
single fact that can be understood without the rest of the text: replace pronouns with the
full names they refer to, and keep any qualifying dates or numbers.
<title>{title}</title>
<text>{text}</text>
Return a JSON list of strings."""


def propositionize(docs=CORPUS) -> list[Chunk]:
    llm, out = get_llm(), []
    for d in docs:
        props = parse_json(llm(PROMPT.format(title=d.title, text=d.text), task="propositions"),
                           [d.text])
        out += [Chunk(f"{d.id}:p{i}", d.id, d.title, p, d.meta) for i, p in enumerate(props)]
    return out


def demo() -> None:
    props = propositionize(CORPUS[:1])
    show("Propositions for d01", [p.text for p in props])
    hits = Retriever(propositionize()).search("What is the main product of Orbital Systems?", 2)
    show("Proposition hits", [h.text for h in hits])


if __name__ == "__main__":
    demo()
