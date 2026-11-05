from datetime import UTC, datetime, timedelta

from chatbot.core.failures import FailureCode
from chatbot.core.types import Answer, Block, BlockType, Chunk, Document, ScoredChunk
from chatbot.obs.trace_store import InMemoryTraceStore
from chatbot.obs.tracing import Tracer
from chatbot.ops.feedback import (
    Feedback,
    JsonlFeedbackStore,
    Rating,
    ReasonCode,
    export_candidates,
    feedback_summary,
    to_golden_candidates,
)
from chatbot.ops.freshness import FreshnessTracker, stale_documents


async def test_feedback_to_golden_candidates(tmp_path):
    traces = InMemoryTraceStore()
    tracer = Tracer(traces, log_records=False)
    async with tracer.trace(query="What is the PTO policy?") as t:
        t.record_retrieval(
            "dense", [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="x"), score=0.8)]
        )
        t.record_context(["c1"])
        t.set_answer(Answer(text="20 days"))
    store = JsonlFeedbackStore(tmp_path / "fb.jsonl")
    await store.add(
        Feedback(
            trace_id=t.trace_id, rating=Rating.DOWN, reasons=[ReasonCode.OUTDATED], comment="now 25"
        )
    )
    await store.add(
        Feedback(trace_id=t.trace_id, rating=Rating.DOWN, reasons=[ReasonCode.UNAUTHORIZED])
    )
    await store.add(Feedback(trace_id=t.trace_id, rating=Rating.UP))
    await store.add(Feedback(trace_id="missing", rating=Rating.DOWN))
    items = await store.list()
    assert len(items) == 4
    cands = await to_golden_candidates(await store.list(negative_only=True), traces)
    assert len(cands) == 1
    c = cands[0]
    assert c.question == "What is the PTO policy?" and c.answer == "20 days"
    assert c.retrieved_ids == ["c1"] and c.context_ids == ["c1"]
    assert c.reasons == [ReasonCode.OUTDATED, ReasonCode.UNAUTHORIZED]
    assert c.suggested_failure is FailureCode.F12_STALE_CONFLICTING and c.urgent
    assert export_candidates(cands, tmp_path / "triage.jsonl") == 1
    summary = feedback_summary(items)
    assert summary["thumbs_up_rate"] == 0.25


def test_freshness_slo_report():
    now = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)
    tr = FreshnessTracker(slo_s=3600, stale_after_days=180, now=lambda: now)
    tr.record_indexed(
        "fast",
        source_updated_at=now - timedelta(minutes=30),
        indexed_at=now - timedelta(minutes=20),
    )
    tr.record_indexed(
        "slow", source_updated_at=now - timedelta(hours=5), indexed_at=now - timedelta(hours=1)
    )
    tr.record_source_update("pending", now - timedelta(hours=2))
    tr.record_indexed(
        "old",
        source_updated_at=now - timedelta(days=400),
        indexed_at=now - timedelta(days=400),
        content_hash="h",
    )
    tr.record_indexed(
        "old-copy",
        source_updated_at=now - timedelta(days=10),
        indexed_at=now - timedelta(days=10),
        content_hash="h",
    )
    r = tr.report()
    assert r.documents == 5
    assert r.violations == ["pending", "slow"]
    assert r.pending == ["pending"]
    assert r.stale == ["old"]
    assert r.duplicates == {"h": ["old", "old-copy"]}
    assert r.slo_attainment == 3 / 5 and not r.slo_met
    # re-indexing the pending doc clears it
    tr.record_indexed("pending", indexed_at=now)
    assert "pending" not in tr.report().pending


def test_stale_documents_helper():
    now = datetime(2026, 11, 5, tzinfo=UTC)
    docs = [
        Document(
            doc_id="a",
            updated_at=now - timedelta(days=365),
            blocks=[Block(type=BlockType.PARAGRAPH, text="x")],
        ),
        Document(doc_id="b", updated_at=now - timedelta(days=10)),
        Document(doc_id="c"),
    ]
    assert stale_documents(docs, months=6, now=now) == ["a"]
