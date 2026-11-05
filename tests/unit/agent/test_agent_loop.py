import json

from agent_fakes import CORPUS, FakeDocStore, FakeRetriever

from chatbot.agent import AgentBudget, AgenticRAG, StopReason
from chatbot.core.types import AnswerStatus, Principal
from chatbot.llm.base import LLMResponse, ToolCall
from chatbot.llm.fake import FakeLLM

ALICE = Principal(user_id="alice", groups=frozenset({"staff"}))


def tool_response(*calls: tuple[str, str, dict]) -> LLMResponse:
    tcs = [ToolCall(id=i, name=n, input=a) for i, n, a in calls]
    raw = [{"type": "text", "text": "Let me search."}] + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tcs
    ]
    return LLMResponse(
        text="Let me search.", tool_calls=tcs, stop_reason="tool_use", raw_content=raw
    )


def _agent(llm, **budget) -> AgenticRAG:
    return AgenticRAG(
        llm, FakeRetriever(CORPUS), FakeDocStore(CORPUS), budget=AgentBudget(**budget)
    )


async def test_loop_executes_tools_and_cites():
    llm = FakeLLM(
        script=[
            tool_response(
                ("t1", "search", {"query": "refund days"}),
                ("t2", "search", {"query": "PTO accrue"}),
            ),
            "Refunds within 30 days [1]; employees accrue 20 days of PTO [3]. Bogus [99].",
        ]
    )
    res = await _agent(llm).run("refund window and PTO?", ALICE)
    assert res.stop_reason is StopReason.ANSWERED
    assert res.tool_calls == 2
    # Second request: assistant raw content appended, then ONE user message with BOTH tool results.
    second = llm.calls[1].messages
    assert second[-2].role == "assistant" and second[-2].content[1]["type"] == "tool_use"
    results = second[-1].content
    assert second[-1].role == "user" and [b["type"] for b in results] == [
        "tool_result",
        "tool_result",
    ]
    assert {b["tool_use_id"] for b in results} == {"t1", "t2"}
    ans = res.answer
    assert ans.status is AnswerStatus.ANSWERED
    assert {c.index for c in ans.citations} <= {1, 2, 3}
    assert "[99]" not in ans.text
    assert all(c.doc_id.startswith("file:") for c in ans.citations)


async def test_step_cap_forces_final_answer():
    counter = {"n": 0}

    def handler(req):
        last = req.messages[-1].content
        if isinstance(last, list) and any(
            b.get("type") == "text" and "Budget reached" in b.get("text", "") for b in last
        ):
            return "Partial: refunds within 30 days [1]."
        counter["n"] += 1
        return tool_response((f"t{counter['n']}", "search", {"query": f"refund {counter['n']}"}))

    llm = FakeLLM(handler=handler)
    res = await _agent(llm, max_steps=3).run("refund?", ALICE)
    assert res.stop_reason is StopReason.MAX_STEPS
    assert counter["n"] == 3 and len(llm.calls) == 4  # 3 steps + 1 final
    assert res.answer.status is AnswerStatus.PARTIAL and res.answer.citations


async def test_loop_detection_refuses_identical_calls():
    llm = FakeLLM(handler=lambda req: tool_response(("x", "search", {"query": "refund"})))
    res = await _agent(llm, max_steps=10, max_loop_hits=2).run("refund?", ALICE)
    assert res.stop_reason is StopReason.LOOP_DETECTED
    assert res.loop_hits == 2 and res.tool_calls == 1
    repeated = [t for s in res.steps for t in s.tool_calls if t.repeated]
    assert len(repeated) == 2 and all(t.is_error for t in repeated)


async def test_refusal_and_abstention():
    llm = FakeLLM(script=[LLMResponse(text="", stop_reason="refusal")])
    res = await _agent(llm).run("q?", ALICE)
    assert res.stop_reason is StopReason.REFUSED and res.answer.status is AnswerStatus.BLOCKED
    llm = FakeLLM(script=["I don't know: nothing about pets."])
    res = await _agent(llm).run("pets?", ALICE)
    assert res.answer.status is AnswerStatus.ABSTAINED


async def test_token_budget():
    from chatbot.llm.base import Usage

    def handler(req):
        r = tool_response(
            (f"t{len(req.messages)}", "search", {"query": f"refund {len(req.messages)}"})
        )
        r.usage = Usage(input_tokens=600, output_tokens=0)
        return r

    res = await _agent(FakeLLM(handler=handler), max_tokens=1000).run("refund?", ALICE)
    assert res.stop_reason is StopReason.TOKEN_BUDGET


async def test_agent_cannot_see_restricted_evidence():
    llm = FakeLLM(
        script=[
            tool_response(("t1", "search", {"query": "confidential incident breach"})),
            "I don't know.",
        ]
    )
    retr = FakeRetriever(CORPUS)
    agent = AgenticRAG(llm, retr, FakeDocStore(CORPUS))
    res = await agent.run("incidents?", ALICE)
    assert all(c.chunk_id != "s1" for c in res.evidence)
    assert retr.calls and retr.calls[0][1] == ALICE
    assert json.dumps(llm.calls[1].messages[-1].content).find("March") == -1
