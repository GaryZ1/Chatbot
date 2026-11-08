"""Rewrite-Retrieve-Read (Ma et al., 2023) and conversational query condensing: turn the
user's message (chatty, elliptical, multi-turn) into a standalone search query first."""

from rag_kit import Retriever, generate_answer, get_llm, show

PROMPT = """<history>
{history}
</history>
<question>{question}</question>
Rewrite the question as a standalone, keyword-rich search query. Resolve pronouns using the
history. Answer only with the query."""


def rewrite_retrieve_read(question: str, history: list[str] | None = None) -> dict:
    llm = get_llm()
    query = llm(PROMPT.format(history="\n".join(history or []), question=question),
                task="rewrite").strip()
    hits = Retriever().search(query, k=3)
    return {"query": query, "answer": generate_answer(llm, question, hits)}


def demo() -> None:
    show("Rewrite-Retrieve-Read",
         rewrite_retrieve_read("ok so umm how much does the ST-2 star tracker weigh??"))


if __name__ == "__main__":
    demo()
