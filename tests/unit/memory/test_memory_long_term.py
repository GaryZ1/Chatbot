import json

import pytest

from chatbot.core.types import Principal
from chatbot.llm import FakeLLM
from chatbot.memory import (
    JSONFileMemoryBackend,
    LongTermMemory,
    MemoryKind,
    SensitiveMemoryError,
    screen_sensitive,
)

ALICE = Principal(user_id="alice", tenant_id="t1")
ALICE_OTHER_TENANT = Principal(user_id="alice", tenant_id="t2")


def extraction(*mems):
    return json.dumps({"memories": list(mems)})


@pytest.mark.parametrize(
    "text,category",
    [
        ("My password is hunter2", "secret_assignment"),
        ("key sk-ant-abcdefghijklmnopqrstuv", "api_key"),
        ("AKIAABCDEFGHIJKLMNOP", "api_key"),
        ("card 4111 1111 1111 1111", "payment_card"),
        ("ssn 123-45-6789", "us_ssn"),
        ("mail me at alice@example.com", "email"),
        ("call +1 415 555 0100", "phone"),
    ],
)
def test_screen_sensitive(text, category):
    assert category in screen_sensitive(text)


def test_screen_allows_normal_facts():
    assert screen_sensitive("The user prefers answers in Python and works on the EMEA team.") == []
    assert screen_sensitive("Uses product SKU-1234 version 2.1") == []


async def test_refuses_secrets():
    llm = FakeLLM(
        handler=lambda r: extraction(
            {
                "kind": "fact",
                "key": "wifi",
                "content": "The user's password is hunter2",
                "confidence": 0.9,
            },
            {
                "kind": "preference",
                "key": "language",
                "content": "Prefers Python examples",
                "confidence": 0.9,
            },
        )
    )
    ltm = LongTermMemory(llm)
    written = await ltm.extract_and_store(ALICE, "my password is hunter2 and I like Python")
    assert [m.content for m in written] == ["Prefers Python examples"]
    with pytest.raises(SensitiveMemoryError):
        await ltm.remember(ALICE, "my api key is sk-abcdefghijklmnopqrstuvwxyz")
    items = await ltm.list_memories(ALICE)
    with pytest.raises(SensitiveMemoryError):
        await ltm.edit(ALICE, items[0].memory_id, "email alice@example.com")
    assert all("hunter2" not in m.content for m in await ltm.list_memories(ALICE))


async def test_dedupes_and_updates_instead_of_appending():
    script = [
        extraction(
            {"kind": "fact", "key": "team", "content": "Works on the EMEA team", "confidence": 0.9}
        ),
        extraction(
            {"kind": "fact", "key": "Team", "content": "Works on the APAC team", "confidence": 0.9}
        ),
        extraction(
            {
                "kind": "preference",
                "key": "",
                "content": "Prefers answers as bullet points",
                "confidence": 0.9,
            }
        ),
        extraction(
            {
                "kind": "preference",
                "key": "",
                "content": "prefers answers as bullet points!",
                "confidence": 0.9,
            }
        ),
        extraction({"kind": "fact", "key": "mood", "content": "Is tired today", "confidence": 0.3}),
    ]
    ltm = LongTermMemory(FakeLLM(script=script))
    for msg in ["I'm on EMEA", "I moved to APAC", "use bullets", "bullets please", "so tired"]:
        await ltm.extract_and_store(ALICE, msg)
    items = await ltm.list_memories(ALICE)
    assert len(items) == 2
    team = next(m for m in items if m.key == "team")
    assert team.content == "Works on the APAC team" and team.version == 2
    assert team.updated_at >= team.created_at


async def test_tenant_scoping_controls_and_retrieval(tmp_path):
    ltm = LongTermMemory(FakeLLM(), JSONFileMemoryBackend(tmp_path / "mem.json"))
    a = await ltm.remember(
        ALICE, "Prefers Python code examples", kind=MemoryKind.PREFERENCE, key="code_lang"
    )
    await ltm.remember(ALICE, "Works on the payments platform team", key="team")
    assert await ltm.list_memories(ALICE_OTHER_TENANT) == []
    hits = await ltm.retrieve(ALICE, "which team handles payments?", k=1)
    assert hits[0].key == "team"
    block = LongTermMemory.format_for_prompt(hits)
    assert "<user_memories" in block and "payments platform" in block

    edited = await ltm.edit(ALICE, a.memory_id, "Prefers TypeScript code examples")
    assert edited.version == 2
    export = await ltm.export(ALICE)
    assert len(export["memories"]) == 2 and json.dumps(export)

    # A fresh manager over the same file sees the same data.
    again = LongTermMemory(FakeLLM(), JSONFileMemoryBackend(tmp_path / "mem.json"))
    assert {m.content for m in await again.list_memories(ALICE)} == {
        "Prefers TypeScript code examples",
        "Works on the payments platform team",
    }
    assert await again.delete(ALICE, a.memory_id)
    assert not await again.delete(ALICE_OTHER_TENANT, a.memory_id)
    assert await again.delete_all(ALICE) == 1
    assert await again.list_memories(ALICE) == []


async def test_extraction_llm_failure_is_graceful():
    ltm = LongTermMemory(FakeLLM(handler=lambda r: "not json"))
    assert await ltm.extract_and_store(ALICE, "I like Go") == []
