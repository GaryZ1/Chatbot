"""Speculative RAG (Wang et al., Google, 2024): cluster retrieved docs into diverse subsets;
a small "drafter" model writes one draft + rationale per subset in parallel; a large
"verifier" model scores each draft and the best one is returned. Faster and often more
accurate than one big model reading everything."""

from concurrent.futures import ThreadPoolExecutor

from rag_kit import Chunk, HashingEmbedder, Retriever, format_context, get_llm, parse_json, show
from raptor import kmeans

DRAFT = """<context>{ctx}</context>
<question>{q}</question>
Answer the question from the context and explain why. Return JSON
{{"draft": "...", "rationale": "..."}}."""
VERIFY = """<question>{q}</question>
<answer>{draft}</answer>
<rationale>{why}</rationale>
Is the rationale sound and does it support the answer? Return JSON
{{"verdict": "yes"|"no", "score": 0-1}}."""


def subsets(docs: list[Chunk], m: int) -> list[list[Chunk]]:
    """Diverse subsets: one doc from each embedding cluster per subset."""
    labels = kmeans(HashingEmbedder().embed(d.text for d in docs), m)
    clusters = [[d for d, lab in zip(docs, labels, strict=True) if lab == j] for j in range(m)]
    clusters = [c for c in clusters if c]
    return [[c[i % len(c)] for c in clusters] for i in range(m)]


def speculative_rag(question: str, k: int = 6, m: int = 3) -> dict:
    llm = get_llm()          # with ClaudeLLM, route drafts to a small model, verify with Opus
    groups = subsets(Retriever().search(question, k=k), m)

    def draft(group):
        out = parse_json(llm(DRAFT.format(ctx=format_context(group), q=question),
                             task="draft"), {})
        return {"draft": out.get("draft", ""), "rationale": out.get("rationale", ""),
                "docs": [d.id for d in group]}

    with ThreadPoolExecutor() as pool:
        drafts = list(pool.map(draft, groups))
    for d in drafts:
        v = parse_json(llm(VERIFY.format(q=question, draft=d["draft"], why=d["rationale"]),
                           task="verify"), {})
        d["score"] = float(v.get("score", 0.0))
    return max(drafts, key=lambda d: d["score"]) | {"n_drafts": len(drafts)}


def demo() -> None:
    show("Speculative RAG", speculative_rag("What does Lumen Labs build?"))


if __name__ == "__main__":
    demo()
