from __future__ import annotations

import pytest

from chatbot.core.types import Chunk, ScoredChunk
from chatbot.rerank import (
    ThresholdGate,
    dedup_near_identical,
    fit_threshold,
    group_by_document,
    lost_in_the_middle_order,
    postprocess_context,
)


def sc(
    cid: str, score: float, text: str | None = None, doc: str | None = None, pos: int = 0
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=cid, doc_id=doc or cid, text=text or f"unique text {cid}", position=pos
        ),
        score=score,
    )


def test_fit_threshold_f1_and_precision() -> None:
    pairs = [
        (0.9, True),
        (0.8, True),
        (0.7, True),
        (0.6, False),
        (0.5, True),
        (0.3, False),
        (0.2, False),
    ]
    fit = fit_threshold(pairs)
    assert fit.tau == pytest.approx(0.5)  # answer >=0.5: P=4/5, R=1 → F1 0.889
    assert fit.f1 == pytest.approx(2 * 0.8 / 1.8)
    strict = fit_threshold(pairs, objective="precision", target_precision=1.0)
    assert (
        strict.tau == pytest.approx(0.7)
        and strict.precision == 1.0
        and strict.recall == pytest.approx(0.75)
    )
    never = fit_threshold([(0.4, False), (0.9, False)], objective="precision")
    assert never.answer_rate == 0.0
    with pytest.raises(ValueError):
        fit_threshold([])


def test_threshold_gate() -> None:
    gate = ThresholdGate.from_fit(fit_threshold([(0.9, True), (0.1, False)]))
    assert not gate.should_abstain([sc("a", 0.95)])
    assert gate.should_abstain([sc("a", 0.05)])
    d = gate.decide([])
    assert d.abstain and d.max_score is None
    staged = ThresholdGate(0.5, stage="ce")
    hit = sc("a", 9.0).with_score(0.2, "ce")
    assert staged.should_abstain([hit]) and staged.filter([hit]) == []


def test_dedup_and_ordering_and_grouping() -> None:
    base = "the refund policy allows returns within thirty days of purchase for all plans"
    hits = [
        sc("a", 0.9, base, doc="d1", pos=2),
        sc("b", 0.8, base + " today", doc="d2"),  # near-duplicate copy
        sc("c", 0.7, "completely different content about sso", doc="d1", pos=0),
        sc("d", 0.6),
        sc("e", 0.5),
    ]
    dd = dedup_near_identical(hits)
    assert [h.chunk.chunk_id for h in dd] == ["a", "c", "d", "e"]
    lim = lost_in_the_middle_order(dd)
    assert [h.chunk.chunk_id for h in lim] == ["a", "d", "e", "c"]  # best first, 2nd best last
    groups = group_by_document(dd)
    assert groups[0].doc_id == "d1" and [h.chunk.chunk_id for h in groups[0].chunks] == ["c", "a"]
    assert [h.chunk.chunk_id for h in postprocess_context(hits, order="document")][:2] == ["c", "a"]
    assert groups[0].header == "d1"
