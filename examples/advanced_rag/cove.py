"""Chain-of-Verification (CoVe, Dhuliawala et al., Meta, 2023) applied to RAG: draft an
answer, plan verification questions for each claim, answer them INDEPENDENTLY with fresh
retrieval (so the draft cannot bias them), then revise the draft."""

from rag_kit import Retriever, generate_answer, get_llm, parse_json, show

PLAN = """<draft>{draft}</draft>
List one verification question per factual claim in the draft. Return a JSON list."""
REVISE = """<question>{q}</question>
<draft>{draft}</draft>
<verifications>
{checks}
</verifications>
Rewrite the draft so that it keeps only claims confirmed by the verifications, with their
citations. If nothing is confirmed, say you don't know."""


def cove(question: str) -> dict:
    llm, r = get_llm(), Retriever()
    draft = generate_answer(llm, question, r.search(question, k=3))
    questions = parse_json(llm(PLAN.format(draft=draft), task="verify_questions"), [])
    checks = [(vq, generate_answer(llm, vq, r.search(vq, k=2))) for vq in questions]
    lines = "\n".join(f"<check q='{vq}'>{a}</check>" for vq, a in checks)
    final = llm(REVISE.format(q=question, draft=draft, checks=lines), task="revise")
    return {"draft": draft, "checks": checks, "final": final}


def demo() -> None:
    show("CoVe", cove("When was Skyline 3.0 released and what did it add?"))


if __name__ == "__main__":
    demo()
