import json

import pytest

from chatbot.core.types import Principal, Role
from chatbot.llm import FakeLLM, LLMRequest
from chatbot.llm.fake import last_user_text
from chatbot.memory import ConversationMemory, InMemorySessionStore, Session
from chatbot.memory.session import SLOT_SYSTEM, SUMMARY_SYSTEM

ALICE = Principal(user_id="alice", tenant_id="t1")


def handler_factory(
    summary="SUMMARY: user asked about ACME order 42.", slots=None, fail_summary=False
):
    def handler(req: LLMRequest) -> str:
        if req.system.startswith(SUMMARY_SYSTEM[:40]):
            if fail_summary:
                return ""
            return summary
        if req.system.startswith(SLOT_SYSTEM[:40]):
            msg = last_user_text(req)
            if "order 42" in msg:
                return json.dumps({"set": {"order_id": "42", "password": "x"}, "clear": []})
            return json.dumps(slots or {"set": {}, "clear": []})
        raise AssertionError(f"unexpected prompt: {req.system[:60]}")

    return handler


async def test_summary_rollover():
    llm = FakeLLM(handler=handler_factory())
    mem = ConversationMemory(
        llm, window_turns=4, summarize_batch=2, condense_turns=2, max_raw_turns=10
    )
    s = await mem.load(None, ALICE)
    await mem.add_exchange(s, "Where is ACME order 42?", "It shipped.")
    await mem.add_exchange(s, "When will it arrive?", "Tuesday.")
    assert s.summary == "" and len(s.turns) == 4
    await mem.add_exchange(s, "Can I change the address?", "Yes, before dispatch.")
    assert s.summary.startswith("SUMMARY")
    assert len(s.turns) == 4 and s.summarized_turns == 2 and s.total_turns == 6
    summary_calls = [c for c in llm.calls if c.system.startswith(SUMMARY_SYSTEM[:40])]
    assert len(summary_calls) == 1
    assert "Where is ACME order 42?" in str(summary_calls[0].messages[0].content)

    ctx = mem.context_for_generation(s)
    assert ctx.summary == s.summary and len(ctx.recent_turns) == 4
    assert ctx.slots == {"order_id": "42"}  # "password" slot refused
    assert "order_id=42" in ctx.summary_block()
    hist = mem.history_for_condensing(s)
    assert hist[0].content.startswith("(Earlier conversation summary)")
    assert [t.content for t in hist[1:]] == ["Can I change the address?", "Yes, before dispatch."]

    reloaded = await mem.load(s.session_id, ALICE)
    assert reloaded.summary == s.summary and len(reloaded.turns) == 4


async def test_summary_failure_keeps_turns_until_hard_cap():
    llm = FakeLLM(handler=handler_factory(fail_summary=True))
    mem = ConversationMemory(
        llm, window_turns=2, summarize_batch=1, max_raw_turns=4, track_slots=False
    )
    s = await mem.load("s1", ALICE)
    for i in range(3):
        await mem.add_exchange(s, f"question {i}", f"answer {i}")
    assert len(s.turns) <= 4
    assert "User asked: question" in s.summary


async def test_slot_names_restrict_slots():
    llm = FakeLLM(handler=handler_factory())
    mem = ConversationMemory(llm, slot_names=["order_id"])
    s = await mem.load(None, ALICE)
    await mem.add_turn(s, Role.USER, "status of order 42")
    assert s.slots == {"order_id": "42"}


async def test_session_isolation_and_ttl():
    now = [0.0]
    store = InMemorySessionStore(ttl_s=10, clock=lambda: now[0])
    mem = ConversationMemory(FakeLLM(handler=handler_factory()), store, track_slots=False)
    s = await mem.load("abc", ALICE)
    with pytest.raises(PermissionError):
        await mem.load("abc", Principal(user_id="bob", tenant_id="t1"))
    with pytest.raises(PermissionError):
        await mem.load("abc", Principal(user_id="alice", tenant_id="t2"))
    now[0] = 11.0
    assert await store.get(s.session_id) is None


async def test_store_returns_copies():
    store = InMemorySessionStore()
    s = Session(session_id="x")
    await store.put(s)
    got = await store.get("x")
    assert got is not None
    got.summary = "changed"
    again = await store.get("x")
    assert again is not None and again.summary == ""
