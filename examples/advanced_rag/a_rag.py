"""A-RAG, hierarchical retrieval interfaces for agentic RAG (arXiv 2602.03442, 2026):
instead of one fixed retrieval algorithm, give the agent tools at several granularities
(corpus -> document -> section -> chunk, plus keyword and semantic search) and let it
choose. It skims titles, opens what looks promising, and reads only the parts it needs,
like a person using a library."""

from agentic_react import agent_loop
from rag_kit import BM25, CORPUS, Retriever, show, split_sentences

DOCS = {d.id: d for d in CORPUS}


def hierarchical_tools():
    r, title_index = Retriever(), BM25([f"{d.title} {d.text}" for d in CORPUS])

    def list_documents(_: str) -> str:
        return "\n".join(f"{d.id}: {d.title}" for d in CORPUS)

    def find_documents(query: str) -> str:
        scores = title_index.scores(query)
        return "\n".join(f"{CORPUS[i].id}: {CORPUS[i].title}"
                         for i in scores.argsort()[::-1][:3] if scores[i] > 0)

    def outline(doc_id: str) -> str:
        d = DOCS.get(doc_id.strip())
        if not d:
            return "error: unknown doc id"
        return "\n".join(f"{doc_id}.{i}: {s[:60]}..." for i, s in
                         enumerate(split_sentences(d.text)))

    def read_section(ref: str) -> str:
        doc_id, _, i = ref.strip().partition(".")
        sents = split_sentences(DOCS[doc_id].text) if doc_id in DOCS else []
        return sents[int(i)] if i.isdigit() and int(i) < len(sents) else "error: bad ref"

    return {
        "list_documents": ("list_documents(''): every document id and title", list_documents),
        "find_documents": ("find_documents(query): keyword search over titles + text",
                           find_documents),
        "outline": ("outline(doc_id): numbered sections of a document", outline),
        "read_section": ("read_section('d03.1'): full text of one section", read_section),
        "search": ("search(query): semantic chunk search across the corpus",
                   lambda q: "\n".join(f"[{c.id}] {c.text}" for c in r.search(q, k=3))),
    }


def demo() -> None:
    result = agent_loop("What does the Lumen ST-2 weigh, and where is it built?",
                        tools=hierarchical_tools())
    show("A-RAG trace", [(s["action"], s["input"]) for s in result["trace"]])
    show("Answer", result["answer"])


if __name__ == "__main__":
    demo()
