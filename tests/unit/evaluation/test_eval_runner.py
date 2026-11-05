import json

import pytest

from chatbot.core.types import ACL, Answer, AnswerStatus, Chunk, Citation, ScoredChunk
from chatbot.evaluation import (
    EvalConfig,
    EvalRunner,
    EvalTrace,
    GateConfig,
    GoldenExample,
    GoldenSet,
    JudgeSuite,
    RegressionGate,
    compare,
    render_markdown,
    run_eval,
)
from chatbot.evaluation.judges import CorrectnessJudge, FaithfulnessJudge
from chatbot.llm.fake import FakeLLM


def _gs() -> GoldenSet:
    return GoldenSet(
        [
            GoldenExample(id="ok", question="refund window?", reference_answer="30 days", gold_doc_ids=["refund"]),
            GoldenExample(id="miss", question="pto days?", reference_answer="20 days", gold_doc_ids=["pto"]),
            GoldenExample(id="rerank", question="password length?", reference_answer="12 characters", gold_doc_ids=["sec"]),
            GoldenExample(id="ctx", question="meal allowance?", reference_answer="75 dollars", gold_doc_ids=["exp"]),
            GoldenExample(id="unans", question="pets?", should_abstain=True, must_cite=False),
            GoldenExample(id="halluc", question="bonus?", should_abstain=True, must_cite=False),
            GoldenExample(id="gone", question="parking?", reference_answer="lot B", gold_doc_ids=["parking"]),
        ],
        name="toy",
    )


async def toy_system(ex: GoldenExample) -> EvalTrace:
    lat = {"retrieve": 10.0, "generate": 90.0, "total": 100.0}
    if ex.id == "ok":
        return EvalTrace(
            answer="Refunds are possible within 30 days [1].",
            status=AnswerStatus.ANSWERED,
            citations=[Citation(index=1, chunk_id="c-refund", doc_id="refund", snippet="30 days")],
            candidate_ids=["c-refund", "c-x"],
            final_ids=["c-refund"],
            context_ids=["c-refund"],
            context_texts=["Refunds within 30 days."],
            doc_of={"c-refund": "refund", "c-x": "x"},
            latency_ms=lat,
            cost_usd=0.002,
        )
    if ex.id == "miss":
        return EvalTrace(answer="I don't know.", candidate_ids=["c-x"], doc_of={"c-x": "x"}, latency_ms=lat)
    if ex.id == "rerank":
        return EvalTrace(
            answer="Use a long passphrase.",
            candidate_ids=["c-x", "c-sec"],
            final_ids=["c-x"],
            context_ids=["c-x"],
            doc_of={"c-x": "x", "c-sec": "sec"},
            latency_ms=lat,
        )
    if ex.id == "ctx":
        return EvalTrace(
            answer="It is 50 euros.",
            candidate_ids=["c-exp"],
            final_ids=["c-exp"],
            context_ids=["c-y"],
            doc_of={"c-exp": "exp", "c-y": "y"},
            latency_ms=lat,
        )
    if ex.id == "unans":
        return EvalTrace(answer="Sorry.", status=AnswerStatus.ABSTAINED, latency_ms=lat)
    if ex.id == "halluc":
        return EvalTrace(answer="The bonus is 10%.", status=AnswerStatus.ANSWERED, latency_ms=lat)
    raise RuntimeError("boom")


async def test_runner_funnel_and_fcodes():
    res = await EvalRunner(toy_system, EvalConfig(ks=[1, 5]), name="toy").run(_gs())
    by = {e.id: e for e in res.examples}
    assert by["ok"].passed and by["ok"].metrics["recall@1"] == 1.0 and by["ok"].failure_code is None
    assert by["miss"].failure_code == "F5" and by["miss"].funnel.gold_in_candidates is False
    assert by["rerank"].failure_code == "F5" and by["rerank"].funnel.gold_in_candidates is True
    assert by["rerank"].funnel.gold_in_final is False
    assert by["ctx"].failure_code == "F7"
    assert by["unans"].passed and by["unans"].abstained
    assert by["halluc"].passed is False and by["halluc"].failure_stage == "abstention"
    assert by["gone"].error and "boom" in by["gone"].error
    assert res.failure_counts["F5"] == 2 and res.failure_counts["F7"] == 1 and res.failure_counts["error"] == 1
    assert res.funnel["n"] == 4 and res.funnel["gold_in_candidates"] == pytest.approx(0.75)
    assert res.funnel["gold_in_final"] == pytest.approx(0.5)
    allm = res.aggregate["all"]
    assert allm["error_rate"] == pytest.approx(1 / 7)
    assert allm["abstention_accuracy"] == pytest.approx(4 / 6)  # "miss" over-refused, "halluc" answered
    assert allm["latency_p95_ms"] == pytest.approx(100.0)
    assert res.latency["retrieve"]["p50"] == pytest.approx(10.0)
    assert res.slice_sizes["unanswerable"] == 2
    assert "recall@5" in res.aggregate["answerable"]
    md = render_markdown(res)
    assert "Retrieval funnel" in md and "F5" in md and "Decision:" in md


async def test_runner_with_fake_judges():
    def handler(req):
        if "atomic factual claims" in req.system:
            return json.dumps({"claims": ["Refunds are possible within 30 days."]})
        if "verify whether each claim" in req.system:
            return json.dumps({"verdicts": [{"claim": "x", "reasoning": "r", "verdict": "supported"}]})
        if "candidate answer against a reference" in req.system:
            return json.dumps({"reasoning": "ok", "correct": True, "score": 5})
        return "{}"

    llm = FakeLLM(handler=handler)
    judges = JudgeSuite(faithfulness=FaithfulnessJudge(llm), correctness=CorrectnessJudge(llm))
    gs = GoldenSet([e for e in _gs() if e.id == "ok"])
    res = await EvalRunner(toy_system, judges=judges).run(gs)
    m = res.examples[0].metrics
    assert m["faithfulness"] == 1.0 and m["correct"] == 1.0 and m["correctness_score"] == 5.0


async def test_judge_failure_is_skipped_not_zero():
    llm = FakeLLM(handler=lambda req: "not json")
    judges = JudgeSuite(correctness=CorrectnessJudge(llm))
    gs = GoldenSet([e for e in _gs() if e.id == "ok"])
    res = await EvalRunner(toy_system, judges=judges).run(gs)
    assert "correct" not in res.examples[0].metrics
    assert res.examples[0].passed  # falls back to lexical pass


def _variant(recall_ok: bool):
    async def system(ex: GoldenExample) -> EvalTrace:
        gold = ex.gold_doc_ids[0] if ex.gold_doc_ids else "none"
        hit = recall_ok or int(ex.id[1:]) % 2 == 0
        ids = [gold] if hit else ["other"]
        return EvalTrace(answer=ex.reference_answer, final_ids=ids, context_ids=ids, latency_ms={"total": 50.0})

    return system


def _numbered(n: int = 20) -> GoldenSet:
    return GoldenSet(
        GoldenExample(id=f"q{i}", question=f"q {i}?", reference_answer=f"answer {i}", gold_doc_ids=[f"d{i}"])
        for i in range(n)
    )


async def test_gate_and_compare():
    gs = _numbered()
    base = await EvalRunner(_variant(True), EvalConfig(ks=[10]), name="base").run(gs)
    worse = await EvalRunner(_variant(False), EvalConfig(ks=[10]), name="cand").run(gs)
    cmp = compare(base, worse, n_boot=2000)
    assert cmp.n_paired == 20
    rc = cmp.metrics["recall@10"]
    assert rc.diff.delta == pytest.approx(-0.5) and rc.significant and not rc.improved
    assert "recall@10" in cmp.regressed()
    gate = RegressionGate(GateConfig.from_dict({"recall@10": {"max_drop": 0.02}}))
    bad = gate.evaluate(worse, base)
    assert not bad.passed and bad.exit_code == 1 and "recall@10" in bad.reasons[0]
    good = gate.evaluate(base, base)
    assert good.passed and good.exit_code == 0
    absolute = RegressionGate(GateConfig(rules=[{"metric": "recall@10", "min_value": 0.9}]))
    assert not absolute.evaluate(worse).passed
    missing = RegressionGate(GateConfig(rules=[{"metric": "faithfulness", "max_drop": 0.02, "required": True}]))
    assert not missing.evaluate(base, base).passed
    md = render_markdown(worse, cmp, base)
    assert "Δ (95% CI)" in md and "recall@10" in md


def test_gate_from_yaml(tmp_path):
    p = tmp_path / "gate.yaml"
    p.write_text(
        "gate:\n  rules:\n    - {metric: 'recall@10', max_drop: 0.02}\n    - {metric: latency_p95_ms, max_relative_drop: 0.2}\n"
    )
    cfg = GateConfig.from_yaml(p)
    assert len(cfg.rules) == 2 and cfg.rules[1].lower_better
    with pytest.raises(ValueError):
        GateConfig.from_dict({"rules": [{"metric": "x"}]})


async def test_run_eval_end_to_end(tmp_path):
    gs = _numbered(30)
    base_out = await run_eval(_variant(True), gs, name="baseline", split="all", out_dir=tmp_path)
    assert base_out.report_path and base_out.report_path.exists()
    out = await run_eval(
        _variant(False),
        gs,
        name="candidate",
        split="all",
        baseline=base_out.json_path,
        gate={"recall@10": {"max_drop": 0.02}},
        out_dir=tmp_path,
    )
    assert out.exit_code == 1
    assert "gate FAILED" in out.report_path.read_text()
    with pytest.raises(ValueError):
        await run_eval(_variant(True), gs, split="test", out_dir=None)
    dev = await run_eval(_variant(True), gs, split="dev", sample=5, out_dir=None)
    assert dev.results.n == 5 and dev.exit_code == 0


def test_trace_from_answer():
    ch = Chunk(chunk_id="c1", doc_id="d1", text="t", acl=ACL(public=True))
    ans = Answer(text="x [1]", citations=[Citation(index=1, chunk_id="c1", doc_id="d1")], latency_ms={"total": 5})
    tr = EvalTrace.from_answer(ans, candidates=[ScoredChunk(chunk=ch, score=1.0)], final=[ScoredChunk(chunk=ch, score=1.0)], context=[ch])
    assert tr.to_docs(tr.final_ids) == ["d1"] and tr.context_texts == ["t"] and tr.total_latency_ms == 5
