from __future__ import annotations

import pytest

from chatbot.core.types import Chunk, ScoredChunk
from chatbot.retrieve import (
    FusionConfig,
    FusionMethod,
    convex_combination,
    dedup,
    fuse,
    fuse_many,
    normalize_scores,
    rrf,
    score_gap_cutoff,
    token_budget_cutoff,
)


def sc(cid: str, score: float, stage: str = "x", text: str | None = None) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=cid, doc_id=cid, text=text or f"text {cid}"),
        score=score,
        source=stage,
        stage_scores={stage: score},
    )


def test_rrf_math() -> None:
    a = [sc("x", 9, "dense"), sc("y", 8, "dense"), sc("z", 1, "dense")]
    b = [sc("y", 30, "bm25"), sc("w", 20, "bm25")]
    out = rrf([a, b], k=60)
    scores = {h.chunk.chunk_id: h.score for h in out}
    assert scores["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["w"] == pytest.approx(1 / 62)
    assert scores["z"] == pytest.approx(1 / 63)
    assert [h.chunk.chunk_id for h in out] == ["y", "x", "w", "z"]
    y = out[0]
    assert y.stage_scores["dense"] == 8 and y.stage_scores["bm25"] == 30 and "rrf" in y.stage_scores
    w = rrf([a, b], k=60, weights=[2.0, 1.0], top_k=1)
    assert len(w) == 1 and w[0].score == pytest.approx(2 / 62 + 1 / 61)
    with pytest.raises(ValueError):
        rrf([a, b], weights=[1.0])


def test_normalisation_and_convex() -> None:
    assert normalize_scores([1, 3, 5]) == [0.0, 0.5, 1.0]
    assert normalize_scores([2, 2]) == [1.0, 1.0]
    z = normalize_scores([1, 2, 3], "zscore")
    assert z[1] == pytest.approx(0.0) and z[0] == pytest.approx(-z[2])
    dense = [sc("a", 0.9), sc("b", 0.5), sc("c", 0.1)]
    bm25 = [sc("c", 12.0), sc("b", 6.0)]
    out = convex_combination([dense, bm25], weights=[0.7, 0.3])
    s = {h.chunk.chunk_id: h.score for h in out}
    assert s["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
    assert s["b"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.0)
    assert s["c"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    cfg = FusionConfig(method=FusionMethod.CONVEX, weights={"dense": 0.2, "bm25": 0.8})
    assert fuse([dense, bm25], cfg, names=["dense", "bm25"])[0].chunk.chunk_id == "c"


def test_fuse_many_and_dedup() -> None:
    q1 = [sc("a", 1), sc("dup1", 0.9, text="same text")]
    q2 = [sc("dup2", 1, text="same text"), sc("a", 0.5)]
    out = fuse_many([q1, q2])
    ids = [h.chunk.chunk_id for h in out]
    assert ids[0] == "a" and len(ids) == 2  # dup1/dup2 share content hash
    assert len(dedup(out + out)) == 2


def test_cutoffs() -> None:
    hits = [sc("a", 0.95), sc("b", 0.9), sc("c", 0.88), sc("d", 0.2), sc("e", 0.1)]
    assert [h.chunk.chunk_id for h in score_gap_cutoff(hits)] == ["a", "b", "c"]
    assert score_gap_cutoff(hits, threshold=0.99) == []
    assert len(score_gap_cutoff(hits, min_k=4)) == 5
    assert len(score_gap_cutoff(hits, max_k=2)) == 2
    long = [sc("x", 1, text="word " * 50), sc("y", 0.5, text="word " * 50)]
    assert len(token_budget_cutoff(long, 60)) == 1
    assert len(token_budget_cutoff(long, 10)) == 1  # min_k
