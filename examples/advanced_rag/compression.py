"""Contextual compression: shrink retrieved text to what matters for THIS question before
generation. Extractive (sentence scoring, like RECOMP's extractive compressor, Xu et al.,
2023) and abstractive (an LLM writes a query-focused summary). LLMLingua / LongLLMLingua
(Jiang et al., 2023) prune at the token level with a small LM."""

from rag_kit import Chunk, HashingEmbedder, Retriever, cosine, get_llm, show, split_sentences


def extractive_compress(query: str, chunks: list[Chunk], keep: int = 3,
                        min_sim: float = 0.15) -> str:
    emb = HashingEmbedder()
    qv = emb.embed_one(query)
    sents = [(cosine(qv, emb.embed_one(s)), s, c.id) for c in chunks
             for s in split_sentences(c.text)]
    best = sorted((x for x in sents if x[0] >= min_sim), reverse=True)[:keep]
    return " ".join(f"{s} [{cid}]" for _, s, cid in best)


def abstractive_compress(query: str, chunks: list[Chunk]) -> str:
    docs = "\n".join(f"<document>{c.text}</document>" for c in chunks)
    prompt = (f"{docs}\n<question>{query}</question>\nSummarise only the information that "
              "helps answer the question. If nothing helps, return an empty string.")
    return get_llm()(prompt, task="summarize")


def demo() -> None:
    q = "How fast are refunds paid?"
    hits = Retriever().search(q, k=4)
    before = sum(len(c.text) for c in hits)
    after = extractive_compress(q, hits, keep=1)
    show("Extractive", {"chars_before": before, "chars_after": len(after), "text": after})
    show("Abstractive", abstractive_compress(q, hits))


if __name__ == "__main__":
    demo()
