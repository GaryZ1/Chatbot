"""Step-back prompting (Zheng et al., Google DeepMind, 2023): also retrieve for a more
generic "step-back" question (principles, background), then answer from both contexts."""

from rag_kit import Retriever, generate_answer, get_llm, show

PROMPT = """<question>{q}</question>
Write a more generic step-back question about the background or principles needed to answer
it. Answer only with the step-back question."""


def step_back_rag(question: str) -> dict:
    llm, r = get_llm(), Retriever()
    abstract = llm(PROMPT.format(q=question), task="step_back").strip()
    specific, background = r.search(question, k=2), r.search(abstract, k=2)
    merged = list({c.id: c for c in specific + background}.values())
    return {"step_back": abstract, "answer": generate_answer(llm, question, merged)}


def demo() -> None:
    show("Step-back", step_back_rag("Did Theo Brandt work in the space industry in 2015?"))


if __name__ == "__main__":
    demo()
