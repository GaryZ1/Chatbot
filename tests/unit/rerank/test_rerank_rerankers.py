from __future__ import annotations

import json
import re
from collections.abc import Sequence

import numpy as np
import pytest

from chatbot.core.interfaces import Reranker
from chatbot.core.types import Chunk, ScoredChunk
from chatbot.llm import FakeLLM, LLMRequest
from chatbot.llm.fake import last_user_text
from chatbot.rerank import (
    BusinessPriorReranker,
    CascadeReranker,
    CrossEncoderReranker,
    LexicalOverlapReranker,
    LLMListwiseReranker,
    LLMPointwiseReranker,
    repair_permutation,
    smart_truncate,
)


def sc(cid: str, text: str, score: float = 0.0, **kw: object) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(chunk_id=cid, doc_id=cid, text=text, **kw), score=score)  # type: ignore[arg-type]


CANDS = [
    sc("a", "Office hours and holidays.", 0.9),
    sc("b", "Refunds for annual plans are prorated and sent within 5 days.", 0.5),
    sc("c", "Annual plans renew automatically.", 0.7),
]


async def test_lexical_reranker() -> None:
    r = LexicalOverlapReranker()
    assert isinstance(r, Reranker)
    out = await r.rerank("refund for annual plans", CANDS, 2)
    assert [h.chunk.chunk_id for h in out] == ["b", "c"]
    assert 0 <= out[1].score < out[0].score <= 1
    assert out[0].stage_scores["lexical"] == out[0].score
    assert await r.rerank("x", [], 3) == []


def test_smart_truncate_keeps_heading_and_best_window() -> None:
    body = (
        " ".join(["filler"] * 300) + " the refund window is 30 days " + " ".join(["filler"] * 300)
    )
    ch = Chunk(chunk_id="x", doc_id="d", text=body, heading_path=["Billing", "Refunds"])
    out = smart_truncate("refund window", ch, 40)
    assert out.startswith("Billing > Refunds\n")
    assert "refund window is 30 days" in out
    assert len(out.split()) < 60
    short = Chunk(chunk_id="y", doc_id="d", text="short", embed_text="Ctx\nshort")
    assert smart_truncate("q", short, 40) == "Ctx\nshort"


class FakeCE:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: Sequence[tuple[str, str]], **_: object) -> np.ndarray:
        self.pairs.extend(pairs)
        return np.array(
            [len(set(q.lower().split()) & set(d.lower().split())) for q, d in pairs],
            dtype=np.float32,
        )


async def test_cross_encoder_uses_embed_text_returns_original() -> None:
    model = FakeCE()
    r = CrossEncoderReranker(model=model)
    cands = [sc("p", "plain", embed_text="Doc: refunds\nplain"), sc("q", "other")]
    out = await r.rerank("refunds", cands, 2)
    assert out[0].chunk.chunk_id == "p" and out[0].chunk.text == "plain"
    assert model.pairs[0] == ("refunds", "Doc: refunds\nplain")


def _grade_handler(req: LLMRequest) -> str:
    text = last_user_text(req)
    docs = re.findall(r'<document id="(\d+)">\n(.*?)\n</document>', text, re.S)
    return json.dumps(
        {"grades": [{"id": int(i), "relevance": 3 if "Refund" in d else 0} for i, d in docs]}
    )


async def test_llm_pointwise_batches_and_degrades() -> None:
    llm = FakeLLM(handler=_grade_handler)
    r = LLMPointwiseReranker(llm, batch_size=2, concurrency=2)
    out = await r.rerank("refunds?", CANDS, 3)
    assert out[0].chunk.chunk_id == "b" and out[0].score == pytest.approx(1.0, abs=1e-2)
    assert len(llm.calls) == 2
    assert "untrusted data" in llm.calls[0].system and "<document" in last_user_text(llm.calls[0])
    broken = LLMPointwiseReranker(FakeLLM(default="not json"))
    out2 = await broken.rerank("refunds?", CANDS, 3)
    assert [h.chunk.chunk_id for h in out2] == ["a", "b", "c"]  # first-stage order kept


async def test_llm_listwise_sliding_window() -> None:
    def handler(req: LLMRequest) -> str:
        docs = re.findall(r'<document id="(\d+)">\n(.*?)\n</document>', last_user_text(req), re.S)
        # Rank by the integer embedded in the text, descending.
        ranked = sorted(docs, key=lambda d: -int(re.search(r"\d+", d[1]).group()))  # type: ignore[union-attr]
        return json.dumps({"ranking": [int(i) for i, _ in ranked]})

    cands = [sc(f"c{i}", f"value {i}") for i in range(7)]
    r = LLMListwiseReranker(
        FakeLLM(handler=handler), window=4, step=1
    )  # step <= window - top_k guarantees the top 3
    out = await r.rerank("biggest", cands, 3)
    assert [h.chunk.chunk_id for h in out] == ["c6", "c5", "c4"]
    assert repair_permutation([2, 2, 9, 0], 4) == [2, 0, 1, 3]


async def test_cascade_and_priors() -> None:
    class Reverse:
        name = "reverse"

        async def rerank(self, q: str, c: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
            return [h.with_score(float(i), self.name) for i, h in enumerate(c)][::-1][:top_k]

    cas = CascadeReranker(LexicalOverlapReranker(), Reverse(), expensive_top_m=2)
    out = await cas.rerank("refund annual plans", CANDS, 3)
    assert [h.chunk.chunk_id for h in out] == ["c", "b", "a"]
    gated = CascadeReranker(LexicalOverlapReranker(), Reverse(), use_expensive=lambda q, h: False)
    assert (await gated.rerank("refund annual plans", CANDS, 1))[0].chunk.chunk_id == "b"

    fresh = sc("f", "annual plans pricing", metadata={"authority": 1.0})
    stale = sc("s", "annual plans pricing", metadata={"authority": 0.0})
    pri = BusinessPriorReranker(LexicalOverlapReranker(), recency_weight=0.0, authority_weight=0.2)
    assert (await pri.rerank("annual plans", [stale, fresh], 1))[0].chunk.chunk_id == "f"
