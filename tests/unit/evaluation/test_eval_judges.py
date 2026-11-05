import json

from chatbot.core.types import Chunk
from chatbot.evaluation.judges import (
    CitationSupportJudge,
    ContextJudge,
    FaithfulnessJudge,
    PairwiseJudge,
    win_rate,
)
from chatbot.evaluation.synthetic import Evolution, SyntheticGenerator
from chatbot.llm.fake import FakeLLM, last_user_text


async def test_faithfulness_claim_level():
    script = [
        json.dumps({"claims": ["A is 30 days.", "B is 5 days."]}),
        json.dumps(
            {
                "verdicts": [
                    {"claim": "A is 30 days.", "reasoning": "stated", "verdict": "supported"},
                    {"claim": "B is 5 days.", "reasoning": "absent", "verdict": "unsupported"},
                ]
            }
        ),
    ]
    llm = FakeLLM(script=script)
    res = await FaithfulnessJudge(llm).judge("q", "A is 30 days. B is 5 days.", ["A is 30 days."])
    assert res is not None and res.score == 0.5 and res.unsupported == ["B is 5 days."]
    # Untrusted content is wrapped and the system prompt says it is data.
    assert "<passages>" in last_user_text(llm.calls[1]) and "DATA" in llm.calls[1].system


async def test_faithfulness_missing_verdicts_are_unsupported():
    llm = FakeLLM(script=[json.dumps({"claims": ["a", "b"]}), json.dumps({"verdicts": [{"claim": "a", "verdict": "supported"}]})])
    res = await FaithfulnessJudge(llm).judge("q", "a. b.", ["ctx"])
    assert res is not None and res.score == 0.5


async def test_pairwise_position_randomized_and_mapped():
    # The judge always prefers whatever is shown FIRST (pure position bias).
    llm = FakeLLM(handler=lambda req: json.dumps({"reasoning": "r", "winner": "first"}))
    judge = PairwiseJudge(llm, seed=1)
    results = [await judge.compare("q", "A", "B", item_id=f"i{i}") for i in range(40)]
    orders = {r.a_first for r in results}
    assert orders == {True, False}  # order is randomized
    for r in results:
        assert r.winner == ("a" if r.a_first else "b")
    # With both orders, a position-biased judge yields ties (inconsistent).
    both = PairwiseJudge(llm, seed=1, both_orders=True)
    r = await both.compare("q", "A", "B", item_id="x")
    assert r.winner == "tie" and not r.consistent
    wr = win_rate(results)
    assert abs(wr["a_win"] + wr["b_win"] - 1.0) < 1e-9


async def test_context_and_citation_judges():
    llm = FakeLLM(
        script=[
            json.dumps({"passages": [{"id": 1, "relevant": True}, {"id": 2, "relevant": False}]}),
            json.dumps({"statements": [{"index": 1, "supporting_ids": [1], "jointly_supported": True}]}),
        ]
    )
    assert await ContextJudge(llm).relevance("q", ["a", "b"]) == 0.5
    claims = await CitationSupportJudge(llm).judge("Refunds take 30 days [1][2].", {1: "30 days", 2: "other"})
    assert claims is not None and claims[0].supporting_ids == {"1"} and claims[0].cited_ids == ["1", "2"]


async def test_synthetic_generation_with_filter():
    chunks = [
        Chunk(chunk_id=f"c{i}", doc_id=f"file:d{i % 2}.md", text=("Policy text number %d. " % i) * 20, title=f"D{i}")
        for i in range(4)
    ]

    def handler(req):
        if "evaluation questions" in req.system:
            text = last_user_text(req)
            q = "How long is window %d?" % (hash(text) % 1000)
            return json.dumps({"question": q, "answer": "30 days", "intent": "lookup"})
        # Filter: reject half by realism based on the question text parity.
        user = last_user_text(req)
        realism = 4 if "window" in user else 1
        return json.dumps({"reasoning": "ok", "answerable": True, "standalone": True, "realism": realism})

    gen = SyntheticGenerator(FakeLLM(handler=handler))
    out = await gen.generate(chunks, 6, distribution={Evolution.SIMPLE: 0.5, Evolution.MULTI_CONTEXT: 0.5})
    assert out and all(e.source.value == "synthetic" for e in out)
    multi = [e for e in out if "multi_hop" in e.slices]
    assert all(len(e.gold_chunk_ids) == 2 for e in multi)

    reject = SyntheticGenerator(
        FakeLLM(
            handler=lambda req: json.dumps({"question": "Is this a question?", "answer": "x"})
            if "evaluation questions" in req.system
            else json.dumps({"answerable": False, "standalone": True, "realism": 5})
        )
    )
    assert await reject.generate(chunks, 3) == []
