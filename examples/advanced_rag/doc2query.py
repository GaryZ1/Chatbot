"""Doc2Query / hypothetical question embeddings (Nogueira et al., 2019; "HyPE" 2024-25):
at ingest, generate questions each chunk answers and index them, pointing back to the
chunk. Query-to-question matching is symmetric, which narrows the query/document gap
without any query-time LLM call."""

from rag_kit import Chunk, DenseIndex, chunk_docs, get_llm, parse_json, show

PROMPT = """<text>{text}</text>
Write 3 diverse questions that this text answers. Return a JSON list of strings."""


def build_question_index(chunks: list[Chunk]):
    llm, questions, owners = get_llm(), [], []
    for c in chunks:
        for q in parse_json(llm(PROMPT.format(text=c.text), task="questions"), []):
            questions.append(q)
            owners.append(c)
    return DenseIndex(questions), questions, owners


def search(query: str, k: int = 2):
    index, questions, owners = build_question_index(chunk_docs())
    seen, out = set(), []
    for i in index.scores(query).argsort()[::-1]:
        if owners[i].id not in seen:            # several questions -> one chunk: dedupe
            seen.add(owners[i].id)
            out.append((questions[i], owners[i].id))
        if len(out) == k:
            return out
    return out


def demo() -> None:
    show("Matched generated questions -> chunk", search("Tell me about Theo Brandt"))


if __name__ == "__main__":
    demo()
