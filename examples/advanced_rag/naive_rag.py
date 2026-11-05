"""Naive RAG (Lewis et al., 2020): embed -> top-k -> stuff into the prompt -> generate."""

from rag_kit import DenseIndex, chunk_docs, generate_answer, get_llm, show


def naive_rag(question: str, k: int = 3) -> str:
    chunks = chunk_docs()
    index = DenseIndex([c.text for c in chunks])          # 1. index (offline)
    scores = index.scores(question)                       # 2. embed the query, score all chunks
    top = [chunks[i] for i in scores.argsort()[::-1][:k]]  # 3. top-k by cosine similarity
    return generate_answer(get_llm(), question, top)      # 4. grounded generation


def demo() -> None:
    show("Naive RAG", naive_rag("How long do refunds take to be paid?"))


if __name__ == "__main__":
    demo()
