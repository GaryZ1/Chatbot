"""Contextual retrieval enrichment with FakeLLM."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from chatbot.chunking import StructureChunker
from chatbot.core.types import ACL, Block, BlockType, Chunk, Document
from chatbot.enrich import CONTEXTUAL_USER, ContextualEnricher, chunk_breadcrumb
from chatbot.llm.base import LLM, LLMError, LLMRequest, LLMResponse, T, Tier
from chatbot.llm.fake import FakeLLM, last_user_text


def make_doc(n_sections: int = 3, para_repeat: int = 4) -> Document:
    blocks: list[Block] = [Block(type=BlockType.HEADING, level=1, text="ACME 10-Q Q2 2023")]
    for i in range(n_sections):
        blocks.append(Block(type=BlockType.HEADING, level=2, text=f"Section {i}"))
        blocks.append(
            Block(
                type=BlockType.PARAGRAPH,
                text=f"Section {i} marker. "
                + "Revenue grew three percent over the previous quarter. " * para_repeat,
            )
        )
    return Document(
        doc_id="acme-10q",
        title="ACME 10-Q Q2 2023",
        blocks=blocks,
        acl=ACL(allow_groups=["fin"]),
        tenant_id="t",
    )


def chunks_for(doc: Document, max_tokens: int = 120) -> list[Chunk]:
    return StructureChunker(max_tokens=max_tokens, min_tokens=0).chunk(doc)


async def test_contextual_prompt_layout_and_embed_text() -> None:
    doc = make_doc()
    chunks = chunks_for(doc)

    def handler(req: LLMRequest) -> str:
        user = last_user_text(req)
        marker = user.split("Section ")[1].split(" ")[0] if "Section " in user else "intro"
        return f"Context: This chunk is from ACME's Q2 2023 10-Q, section {marker}."

    llm = FakeLLM(handler=handler)
    out = await ContextualEnricher(llm).enrich(doc, chunks)
    assert len(llm.calls) == len(chunks)
    systems = {r.system for r in llm.calls}
    assert len(systems) == 1  # one cacheable prefix per document
    req = llm.calls[0]
    assert req.cache_system and req.tier == Tier.SMALL
    assert "<document>\n" + doc.text + "\n</document>" in req.system
    assert "untrusted data" in req.system
    assert doc.text not in last_user_text(req)  # the document is not repeated per chunk
    assert "<chunk>" in last_user_text(req)
    assert CONTEXTUAL_USER.split("{chunk}")[1].strip() in last_user_text(req)

    for before, after in zip(chunks, out, strict=True):
        assert after.text == before.text  # display text untouched
        ctx = after.metadata["context"]
        assert ctx.startswith("This chunk is from ACME")  # "Context:" prefix stripped
        crumb = chunk_breadcrumb(after)
        assert crumb.startswith("ACME 10-Q Q2 2023")
        assert after.embed_text == f"{crumb}\n\n{ctx}\n\n{after.text}"
        assert after.metadata["contextual_version"] == "ctx-v1"
        assert after.metadata["context_model_tier"] == "small"
        assert after.metadata["context_model"] == "fake"
        assert after.enrichment_version == "contextual@ctx-v1"
        assert after.acl == before.acl
    assert chunks[0].embed_text != out[0].embed_text  # inputs not mutated


async def test_context_inserted_after_breadcrumb_keeps_table_summary() -> None:
    doc = Document(
        doc_id="t",
        title="Rates",
        blocks=[
            Block(type=BlockType.HEADING, level=1, text="Fees"),
            Block(type=BlockType.TABLE, text="| Plan | Fee |\n| --- | --- |\n| Basic | 5 |"),
        ],
    )
    chunks = StructureChunker(max_tokens=120).chunk(doc)  # heading merged with the table
    out = await ContextualEnricher(FakeLLM(default="Fee schedule for plans.")).enrich(doc, chunks)
    et = out[0].embed_text or ""
    assert et.startswith("Rates > Fees\n\nFee schedule for plans.\n\nTable with columns Plan, Fee")
    assert et.endswith(out[0].text)


class ConcurrencyLLM(LLM):
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return LLMResponse(text="ctx", model="m")

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        yield "ctx"

    async def parse(self, request: LLMRequest, schema: type[T]) -> T:
        raise LLMError("unused")


async def test_bounded_concurrency() -> None:
    doc = make_doc(n_sections=12)
    llm = ConcurrencyLLM()
    await ContextualEnricher(llm, max_concurrency=3).enrich(doc, chunks_for(doc, 40))
    assert 1 < llm.peak <= 3


async def test_windowing_for_long_documents() -> None:
    doc = make_doc(n_sections=30, para_repeat=10)
    chunks = chunks_for(doc, 200)
    enricher = ContextualEnricher(FakeLLM(default="ctx"), max_doc_tokens=1000)
    texts, ranges = enricher.document_windows(doc)
    assert len(ranges) > 2 and len(texts) == len(ranges)
    # half-overlapping windows cover the whole document
    assert ranges[0][0] == 0 and ranges[-1][1] == len(doc.text.rstrip())
    assert all(ranges[i + 1][0] < ranges[i][1] for i in range(len(ranges) - 1))
    llm = FakeLLM(default="ctx")
    out = await ContextualEnricher(llm, max_doc_tokens=1000).enrich(doc, chunks)
    for req, c in zip(llm.calls, out, strict=True):
        assert c.text in req.system  # the chunk's own window is in the cached prefix
        assert "Excerpt" in req.system
    assert len({r.system for r in llm.calls}) < len(llm.calls)  # windows are shared → cache hits
    assert {c.metadata["context_window"] for c in out} >= {0, len(ranges) - 1}


async def test_llm_failure_degrades_to_breadcrumb() -> None:
    def boom(req: LLMRequest) -> str:
        raise LLMError("overloaded")

    doc = make_doc(1)
    out = await ContextualEnricher(FakeLLM(handler=boom)).enrich(doc, chunks_for(doc))
    assert out and all("context_error" in c.metadata for c in out)
    assert all(c.embed_text and c.embed_text.endswith(c.text) for c in out)
    assert all("context" not in c.metadata for c in out)


def test_validation() -> None:
    with pytest.raises(ValueError):
        ContextualEnricher(FakeLLM(), max_concurrency=0)
    with pytest.raises(ValueError):
        ContextualEnricher(FakeLLM(), max_doc_tokens=10)


async def test_context_truncated() -> None:
    doc = make_doc(1)
    long_ctx = "word " * 500
    out = await ContextualEnricher(FakeLLM(default=long_ctx), max_context_tokens=20).enrich(
        doc, chunks_for(doc)
    )
    assert len(out[0].metadata["context"].split()) <= 20
