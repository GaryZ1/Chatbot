"""Self-query retrieval (LangChain SelfQueryRetriever; "metadata extraction"): the LLM
turns the question into a semantic query plus structured filters, which are VALIDATED
against an allow-list before they touch the index (never trust model-written filters)."""

from rag_kit import Retriever, get_llm, parse_json, show

SCHEMA = {"year": int, "type": str}
ALLOWED_TYPES = {"company", "news", "people", "product", "policy", "security", "report",
                 "finance"}
PROMPT = """Metadata fields: year (int), type (one of {types}).
<question>{q}</question>
Return JSON with "query" (the semantic part of the question) and any metadata filters that
the question states explicitly."""


def self_query(question: str, k: int = 3) -> dict:
    raw = parse_json(get_llm()(PROMPT.format(types=sorted(ALLOWED_TYPES), q=question),
                               task="filters"), {})
    filters = {f: raw[f] for f, t in SCHEMA.items() if isinstance(raw.get(f), t)}
    if filters.get("type") not in ALLOWED_TYPES:
        filters.pop("type", None)
    query = raw.get("query") or question

    def where(c):
        return all(c.meta.get(f) == v for f, v in filters.items())

    hits = Retriever().search(query, k=k, where=where)
    return {"query": query, "filters": filters, "hits": [(c.id, c.meta) for c in hits]}


def demo() -> None:
    show("Self-query", self_query("Which policy changes happened in 2024?"))


if __name__ == "__main__":
    demo()
