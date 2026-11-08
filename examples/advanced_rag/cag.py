"""Cache-Augmented Generation (CAG, Chan et al., 2024, "Don't Do RAG"): when the knowledge
base is small and stable, skip retrieval entirely. Load ALL of it into the context once,
cache it (KV cache locally; prompt caching with an API), and answer each question against
the cached prefix. No retrieval errors, low latency; bounded by context size and cost."""

import os

from rag_kit import ANSWER_SYSTEM, CORPUS, get_llm, show

MAX_KB_TOKENS = 200_000           # keep well below the context window; quality degrades
                                  # with very long contexts ("context rot")


def knowledge_block() -> str:
    return "\n".join(f'<source id="{d.id}" title="{d.title}">{d.text}</source>' for d in CORPUS)


def cag_answer_claude(question: str) -> tuple[str, dict]:
    """The KB sits in the system prompt behind a cache breakpoint: the first call writes
    the cache, later calls read it at ~10% of the input price."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model="claude-opus-5-5", max_tokens=2048,
        system=[{"type": "text", "text": ANSWER_SYSTEM},
                {"type": "text", "text": f"<knowledge>\n{knowledge_block()}\n</knowledge>",
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": question}],
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    usage = {"cache_read": resp.usage.cache_read_input_tokens,
             "cache_write": resp.usage.cache_creation_input_tokens}
    return "".join(b.text for b in resp.content if b.type == "text"), usage


def cag_answer(question: str) -> str:
    kb = knowledge_block()
    if len(kb) / 4 > MAX_KB_TOKENS:                       # rough chars-per-token estimate
        raise ValueError("Knowledge base too large for CAG: use RAG instead.")
    if os.getenv("RAG_LLM") == "claude":
        return cag_answer_claude(question)[0]
    return get_llm()(f"<context>{kb}</context>\n<question>{question}</question>",
                     task="answer", system=ANSWER_SYSTEM)


def demo() -> None:
    for q in ["Which days do Lisbon teams meet in the office?", "Who is the CEO?"]:
        show(q, cag_answer(q))


if __name__ == "__main__":
    demo()
