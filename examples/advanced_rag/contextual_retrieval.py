"""Contextual Retrieval (Anthropic, 2024): before indexing, an LLM writes 1-2 sentences that
situate each chunk inside its whole document; prepend them to the chunk for BOTH the
embedding and the BM25 index. Put the whole document in a cached prefix so the per-chunk
calls are cheap. Reported -49% retrieval failures (-67% with reranking)."""

from rag_kit import CORPUS, Chunk, Retriever, chunk_docs, get_llm, show

CONTEXT_PROMPT = """<document>
{doc}
</document>
Here is a chunk from that document:
<chunk>
{chunk}
</chunk>
Write a short context (1-2 sentences) that situates this chunk within the overall document
to improve search retrieval of the chunk. Answer only with the context."""


def contextualize(chunks: list[Chunk]) -> list[Chunk]:
    llm, docs = get_llm(), {d.id: d for d in CORPUS}
    out = []
    for c in chunks:
        doc = docs[c.doc_id]
        # With ClaudeLLM, send the document as a cached system/prefix block and only the
        # chunk as the varying suffix, so each document is billed ~once.
        ctx = llm(CONTEXT_PROMPT.format(doc=f"{doc.title}\n{doc.text}", chunk=c.text),
                  task="context")
        out.append(Chunk(c.id, c.doc_id, c.title, f"{ctx}\n{c.text}", c.meta))
    return out


def demo() -> None:
    chunks = contextualize(chunk_docs())
    show("Contextualized chunk", chunks[3].text)
    hits = Retriever(chunks).search("Where is the ST-2 built?", k=2)
    show("Top hits", [h.id for h in hits])


if __name__ == "__main__":
    demo()
