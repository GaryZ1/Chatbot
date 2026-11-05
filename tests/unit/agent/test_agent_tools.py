from agent_fakes import CORPUS, FakeDocStore, FakeRetriever

from chatbot.agent import DocStoreCatalog, ToolContext, ToolPolicy, ToolRegistry
from chatbot.core.types import Principal
from chatbot.llm.base import ToolCall

ALICE = Principal(user_id="alice", groups=frozenset({"staff"}), roles=frozenset({"employee"}))
SEC = Principal(user_id="sam", groups=frozenset({"security-team"}))


def _ctx(p: Principal) -> ToolContext:
    ds = FakeDocStore(CORPUS)
    return ToolContext(
        principal=p,
        retriever=FakeRetriever(CORPUS),
        docstore=ds,
        catalog=DocStoreCatalog(ds, sorted({c.doc_id for c in CORPUS})),
    )


def test_specs_have_strict_schemas():
    specs = {s.name: s for s in ToolRegistry().specs_for(ALICE)}
    assert set(specs) == {"search", "read", "list_sources"}
    for s in specs.values():
        assert s.input_schema["additionalProperties"] is False
        assert s.input_schema["type"] == "object"
    assert specs["search"].input_schema["required"] == ["query"]


async def test_acl_respected_in_all_tools():
    reg = ToolRegistry()
    ctx = _ctx(ALICE)
    res = await reg.execute(
        ToolCall(id="1", name="search", input={"query": "confidential incident breach"}), ctx
    )
    assert not res.is_error and "March" not in res.content and "no results" in res.content
    res = await reg.execute(
        ToolCall(id="2", name="read", input={"doc_id": "file:incident-reports.md"}), ctx
    )
    assert res.is_error and "not found" in res.content and "March" not in res.content
    res = await reg.execute(ToolCall(id="3", name="list_sources", input={}), ctx)
    assert "incident" not in res.content and "file:refund-policy.md" in res.content
    # The security team can see it.
    sctx = _ctx(SEC)
    res = await reg.execute(
        ToolCall(id="4", name="search", input={"query": "confidential incident breach"}), sctx
    )
    assert "March" in res.content


async def test_read_section_and_range():
    reg = ToolRegistry()
    ctx = _ctx(ALICE)
    res = await reg.execute(
        ToolCall(
            id="1", name="read", input={"doc_id": "file:refund-policy.md", "section": "annual"}
        ),
        ctx,
    )
    assert "prorated" in res.content and "full refund" not in res.content
    res = await reg.execute(
        ToolCall(
            id="2", name="read", input={"doc_id": "file:refund-policy.md", "start": 0, "end": 0}
        ),
        ctx,
    )
    assert "full refund" in res.content and "prorated" not in res.content
    assert res.evidence_numbers == [ctx.evidence.number_of("r1")]


async def test_argument_validation_and_allow_list():
    reg = ToolRegistry(policy=ToolPolicy(by_role={"employee": {"search"}}))
    ctx = _ctx(ALICE)
    assert [s.name for s in reg.specs_for(ALICE)] == ["search"]
    res = await reg.execute(ToolCall(id="1", name="read", input={"doc_id": "x"}), ctx)
    assert res.is_error and "not available" in res.content
    res = await reg.execute(
        ToolCall(id="2", name="search", input={"query": "refund", "evil": 1}), ctx
    )
    assert res.is_error and "invalid arguments" in res.content
    res = await reg.execute(
        ToolCall(id="3", name="search", input={"query": "refund", "filters": {"acl": "public"}}),
        ctx,
    )
    assert res.is_error and "not allowed" in res.content
    block = res.to_block()
    assert (
        block["type"] == "tool_result" and block["tool_use_id"] == "3" and block["is_error"] is True
    )
