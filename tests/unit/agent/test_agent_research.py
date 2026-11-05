import json

from agent_fakes import CORPUS, FakeRetriever

from chatbot.agent import DeepResearchConfig, DeepResearcher
from chatbot.core.types import AnswerStatus, Principal
from chatbot.llm.base import LLMError
from chatbot.llm.fake import FakeLLM

ALICE = Principal(user_id="alice", groups=frozenset({"staff"}))


async def test_deep_research_plan_research_synthesize():
    def handler(req):
        if "plan research" in req.system:
            return json.dumps(
                {
                    "reasoning": "r",
                    "sub_questions": ["refund days", "PTO accrue", "refund days", "x", "y", "z"],
                }
            )
        if "You are a researcher" in req.system:
            return json.dumps({"notes": "see [1]", "sufficient": True})
        return "Refunds take 30 days [1] and PTO accrues 20 days [2]. Fake [42]."

    retr = FakeRetriever(CORPUS)
    dr = DeepResearcher(FakeLLM(handler=handler), retr, DeepResearchConfig(max_subquestions=3))
    res = await dr.run("Compare refund window and PTO accrual", ALICE)
    assert res.plan == ["refund days", "PTO accrue", "x"]  # deduped and capped
    assert len(retr.calls) == 3 and all(p == ALICE for _, p in retr.calls)
    assert res.answer.status is AnswerStatus.ANSWERED and "[42]" not in res.answer.text
    assert {c.index for c in res.answer.citations} == {1, 2}
    assert all(c.chunk_id != "s1" for c in res.evidence)


async def test_deep_research_degrades_when_planner_fails():
    class Broken(FakeLLM):
        async def parse(self, request, schema):
            raise LLMError("down")

    dr = DeepResearcher(Broken(default="I don't know."), FakeRetriever(CORPUS))
    res = await dr.run("refund days", ALICE)
    assert res.plan == ["refund days"]
    assert res.answer.status is AnswerStatus.ABSTAINED
