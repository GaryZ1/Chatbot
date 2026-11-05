import json
import logging

import pytest

from chatbot.core.types import Answer, AnswerStatus, Chunk, Citation, Principal, ScoredChunk
from chatbot.guard.verdicts import GuardVerdict
from chatbot.llm.base import Usage
from chatbot.obs.metrics import PriceTable
from chatbot.obs.trace_store import InMemoryTraceStore, JsonlTraceStore, replay, replay_store
from chatbot.obs.tracing import (
    Tracer,
    TraceRecord,
    current_trace,
    genai_attributes,
    redact_record,
    traced,
)

ALICE = Principal(user_id="alice", groups=frozenset({"eng"}))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@traced("retrieve")
async def fake_retrieve(clock):
    clock.t += 0.05
    return [ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="x"), score=0.9)]


@traced()
def sync_step(clock):
    clock.t += 0.01
    return 1


async def test_trace_records_everything_and_redacts(caplog):
    clock = FakeClock()
    store = InMemoryTraceStore()
    tracer = Tracer(store, pipeline_version="v2", clock=clock, prices=PriceTable())
    caplog.set_level(logging.INFO, logger="chatbot.trace")
    async with tracer.trace(
        query="my email is jane@acme.com, why?", principal=ALICE, route="rag"
    ) as t:
        assert current_trace() is t
        t.set(
            condensed_query="why jane@acme.com", filters={"doc_type": "policy"}, prompt_version="p7"
        )
        with t.span("condense", model="small"):
            clock.t += 0.2
        hits = await fake_retrieve(clock)
        sync_step(clock)
        t.record_retrieval("dense", hits)
        t.record_context(hits)
        t.add_verdict(GuardVerdict(stage="input", flags=["pii_redacted"]))
        t.record_llm("claude-haiku-5-5", Usage(input_tokens=1_000_000, output_tokens=0))
        t.set_answer(
            Answer(
                text="Contact jane@acme.com [1]",
                status=AnswerStatus.ANSWERED,
                citations=[
                    Citation(index=1, chunk_id="c1", doc_id="d1", snippet="call 415-555-0132 now")
                ],
            )
        )
    assert current_trace() is None
    rec = await store.get(t.trace_id)
    assert rec is not None
    assert "jane@acme.com" not in rec.query and "[EMAIL_1]" in rec.query
    assert "jane@acme.com" not in rec.answer and "jane@acme.com" not in rec.condensed_query
    assert "415-555-0132" not in rec.citations[0]["snippet"]
    assert rec.latency_ms["condense"] == pytest.approx(200.0)
    assert rec.latency_ms["retrieve"] == pytest.approx(50.0)
    assert rec.latency_ms["sync_step"] == pytest.approx(10.0)
    assert rec.latency_ms["total"] == pytest.approx(260.0)
    assert rec.retrieval["dense"][0].score == 0.9 and rec.context_ids == ["c1"]
    assert rec.prompt_version == "p7" and rec.pipeline_version == "v2" and rec.route == "rag"
    assert rec.scope_key == ALICE.scope_key()
    assert rec.cost_usd == pytest.approx(0.10)
    assert rec.guardrails[0]["flags"] == ["pii_redacted"]
    # one JSON log line, redacted
    lines = [r.getMessage() for r in caplog.records if r.name == "chatbot.trace"]
    assert len(lines) == 1 and "jane@acme.com" not in lines[0]
    assert json.loads(lines[0])["trace_id"] == t.trace_id


async def test_span_error_recorded():
    tracer = Tracer(InMemoryTraceStore())
    with pytest.raises(RuntimeError):
        async with tracer.trace(query="q") as t:
            with t.span("generate"):
                raise RuntimeError("boom")
    rec = await tracer.store.get(t.trace_id)
    assert rec.error == "RuntimeError: boom"
    assert rec.spans[0].status == "error"


def test_redact_record_and_genai_attributes():
    rec = TraceRecord(
        trace_id="t", query="ssn 123-45-6789", model="claude-opus-5-5", usage={"input_tokens": 5}
    )
    red = redact_record(rec)
    assert "123-45-6789" not in red.query and "123-45-6789" in rec.query
    attrs = genai_attributes(red)
    assert attrs["gen_ai.request.model"] == "claude-opus-5-5"
    assert attrs["gen_ai.usage.input_tokens"] == 5
    assert attrs["gen_ai.operation.name"] == "chat"


async def test_jsonl_store_feedback_and_replay(tmp_path):
    store = JsonlTraceStore(tmp_path / "traces.jsonl")
    tracer = Tracer(store, log_records=False)
    for q, a in [("q1", "the vpn uses wireguard"), ("q2", "expenses due in 30 days")]:
        async with tracer.trace(query=q) as t:
            t.set_answer(Answer(text=a))
    recs = await store.list()
    assert [r.query for r in recs] == ["q1", "q2"]
    assert await store.attach_feedback(recs[0].trace_id, {"rating": "down"})
    assert (await store.get(recs[0].trace_id)).feedback == {"rating": "down"}

    async def new_pipeline(rec):
        if rec.query == "q1":
            return Answer(text="the vpn uses wireguard")
        return "I don't know"

    report = await replay_store(store, new_pipeline)
    assert report.change_rate == 0.5
    assert [r.trace_id for r in report.changed] == [recs[1].trace_id]

    async def broken(rec):
        raise ValueError("x")

    rep = await replay(recs, broken)
    assert len(rep.errors) == 2


async def test_sampling_off_skips_store():
    store = InMemoryTraceStore()
    async with Tracer(store, sample_rate=0.0).trace(query="q"):
        pass
    assert len(store) == 0
