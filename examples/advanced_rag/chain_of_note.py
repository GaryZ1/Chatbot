"""Chain-of-Note (Yu et al., Tencent AI, 2023): write a short reading note per retrieved
document (relevant? what does it say?), then answer from the notes. Makes the model
explicitly discard noisy documents and say "unknown" when no note is useful."""

from rag_kit import Retriever, get_llm, show

NOTE = """<document>{doc}</document>
<question>{q}</question>
Write a one-sentence reading note: what does this document say that is relevant to the
question? If nothing, write exactly "This document is irrelevant to the question." """
ANSWER = """<notes>
{notes}
</notes>
<question>{q}</question>
Answer from the relevant notes only. If no note is relevant, say you don't know."""


def chain_of_note(question: str, k: int = 4) -> dict:
    llm = get_llm()
    hits = Retriever().search(question, k=k)
    notes = [(c.id, llm(NOTE.format(doc=c.text, q=question), task="notes").strip())
             for c in hits]
    useful = [(cid, n) for cid, n in notes if "irrelevant" not in n.lower()]
    note_xml = "\n".join(f'<source id="{cid}">{n}</source>' for cid, n in useful)
    answer = llm(ANSWER.format(notes=note_xml, q=question), task="answer") if useful else \
        "I don't know based on the retrieved documents."
    return {"notes": notes, "answer": answer}


def demo() -> None:
    show("Chain-of-Note", chain_of_note("How much did Orbital pay for Lumen Labs?"))


if __name__ == "__main__":
    demo()
