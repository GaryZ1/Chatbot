"""Small-to-big retrieval: index small units for precise matching, return a larger unit
for generation. Two variants: parent-document retrieval (child chunks -> parent chunk or
document) and sentence-window retrieval (sentence -> +-w neighbouring sentences)."""

from rag_kit import CORPUS, Chunk, DenseIndex, show, split_sentences


def build_children():
    children = []
    for d in CORPUS:
        sents = split_sentences(d.text)
        for i, s in enumerate(sents):
            children.append(Chunk(f"{d.id}:s{i}", d.id, d.title, s, {"pos": i, "sents": sents}))
    return children, DenseIndex([c.text for c in children])


CHILDREN, INDEX = build_children()


def parent_document_retrieval(query: str, k: int = 2) -> list[str]:
    docs = {d.id: d for d in CORPUS}
    parents: list[str] = []
    for i in INDEX.scores(query).argsort()[::-1]:
        pid = CHILDREN[i].doc_id
        if pid not in parents:            # dedupe: several children map to one parent
            parents.append(pid)
        if len(parents) == k:
            break
    return [docs[p].text for p in parents]


def sentence_window_retrieval(query: str, k: int = 2, window: int = 1) -> list[str]:
    out = []
    for i in INDEX.scores(query).argsort()[::-1][:k]:
        c = CHILDREN[i]
        lo, hi = max(0, c.meta["pos"] - window), c.meta["pos"] + window + 1
        out.append(" ".join(c.meta["sents"][lo:hi]))
    return out


def demo() -> None:
    q = "Who founded Lumen Labs?"
    show("Parent documents", parent_document_retrieval(q))
    show("Sentence windows", sentence_window_retrieval(q))


if __name__ == "__main__":
    demo()
