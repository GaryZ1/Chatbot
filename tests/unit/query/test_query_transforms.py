import json

from chatbot.core.types import Role, Turn
from chatbot.llm import FakeLLM
from chatbot.query import (
    FieldSpec,
    FieldType,
    FilterSchema,
    HyDEGenerator,
    MultiQueryGenerator,
    QueryCondenser,
    QueryDecomposer,
    QueryNormalizer,
    QueryRewriter,
    QueryRouter,
    SelfQueryExtractor,
    StepBackGenerator,
    find_identifiers,
    looks_like_identifier,
    routing_report,
)
from chatbot.query.plan import Intent, QueryPlan, Route


async def test_hyde_skipped_for_identifier_queries():
    llm = FakeLLM(handler=lambda r: "A hypothetical passage.")
    hyde = HyDEGenerator(llm)
    for q in [
        "SKU-48213 dimensions",
        "error E_CONN_RESET",
        "order #123456 status",
        "AB12 spec",
        "v2.3.1 changes",
    ]:
        res = await hyde.generate(q)
        assert not res.applied, q
        assert res.embed_text == q
    assert llm.calls == []
    res = await hyde.generate("why do refunds take long")
    assert res.applied and "hypothetical passage" in res.embed_text
    assert len(llm.calls) == 1


def test_identifier_detection():
    assert looks_like_identifier("SKU-1234")
    assert looks_like_identifier("ABC123")
    assert not looks_like_identifier("PTO")
    assert not looks_like_identifier("refund")
    assert find_identifiers("compare X100 and X200 plans") == ["X100", "X200"]


def test_normalizer_never_alters_codes():
    n = QueryNormalizer(
        {"PTO": "paid time off", "SKU-1234": "should be ignored", "sla": "service level agreement"}
    )
    out = n.normalize("  PTO   policy for SKU-1234 “now” and the SLA ")
    assert (
        out == 'PTO (paid time off) policy for SKU-1234 "now" and the SLA (service level agreement)'
    )
    # Case-sensitive acronym: lower-case "it" is not "IT".
    n2 = QueryNormalizer({"IT": "information technology"})
    assert n2.normalize("is it working") == "is it working"


async def test_condenser_restores_dropped_entities_and_handles_errors():
    llm = FakeLLM(handler=lambda r: "What was the revenue?")
    hist = [
        Turn(role=Role.USER, content="ACME revenue in Q2?"),
        Turn(role=Role.ASSISTANT, content="$5M"),
    ]
    out = await QueryCondenser(llm).condense("what about Q3 2025 for ACME-77?", hist)
    assert "Q3" in out and "2025" in out and "ACME-77" in out
    assert await QueryCondenser(FakeLLM()).condense("standalone", []) == "standalone"


async def test_rewriter_keeps_raw_when_identifier_lost():
    llm = FakeLLM(handler=lambda r: "product warranty duration")
    assert (
        await QueryRewriter(llm).rewrite("hey how long is warranty on SKU-9")
        == "hey how long is warranty on SKU-9"
    )
    llm2 = FakeLLM(handler=lambda r: "SKU-9 warranty duration")
    assert (
        await QueryRewriter(llm2).rewrite("hey how long is warranty on SKU-9")
        == "SKU-9 warranty duration"
    )


async def test_multi_query_dedupes():
    resp = json.dumps(
        {"queries": ["Refund policy?", "refund policy", "money back rules", "money back rules!"]}
    )
    out = await MultiQueryGenerator(FakeLLM(handler=lambda r: resp), n=3).generate("refund policy")
    assert out == ["refund policy", "money back rules"]
    assert await MultiQueryGenerator(FakeLLM(handler=lambda r: "bad")).generate("x y") == ["x y"]


async def test_decomposer_gate_and_parse():
    llm = FakeLLM(
        handler=lambda r: json.dumps(
            {"sub_queries": ["A price", "B price"], "strategy": "parallel"}
        )
    )
    d = QueryDecomposer(llm)
    assert (await d.decompose("refund window")).sub_queries == ["refund window"]
    assert llm.calls == []
    res = await d.decompose("compare plan A vs plan B price")
    assert res.sub_queries == ["A price", "B price"]


async def test_step_back():
    llm = FakeLLM(handler=lambda r: "What is the refund policy?")
    assert (
        await StepBackGenerator(llm).generate("why was my refund rejected")
        == "What is the refund policy?"
    )


async def test_self_query_extractor_validates():
    schema = FilterSchema.of(
        FieldSpec(name="year", type=FieldType.INT),
        FieldSpec(name="tags", type=FieldType.STRING_LIST),
    )
    resp = json.dumps(
        {
            "semantic_query": "security incidents",
            "filters": {"year": "2024", "tags": "vpn", "bogus": 1},
        }
    )
    res = await SelfQueryExtractor(FakeLLM(handler=lambda r: resp), schema).extract(
        "vpn security incidents 2024"
    )
    assert res.semantic_query == "security incidents"
    assert [d.field for d in res.dropped] == ["bogus"]
    assert res.filter is not None
    assert {(c.field, c.op.value, c.value) for c in res.filter.must} == {
        ("year", "eq", 2024),
        ("tags", "contains", "vpn"),
    }


def test_filter_schema_drops_bad_ops_and_types():
    schema = FilterSchema.of(
        FieldSpec(name="price", type=FieldType.FLOAT), FieldSpec(name="active", type=FieldType.BOOL)
    )
    v = schema.validate_raw({"price": {"contains": 3}, "active": "maybe"})
    assert v.filter is None
    assert {d.field for d in v.dropped} == {"price", "active"}


def test_router_logs_and_confusion_matrix():
    records = []
    router = QueryRouter(log_sink=records.append)
    plan = QueryPlan(
        intent=Intent.ANALYTICS_SQL,
        route=Route.RAG_SIMPLE,
        standalone_query="q",
        original_query="q",
    )
    routed = router.route(plan)
    assert routed.route is Route.SQL_TOOL
    assert records and records[0].route is Route.SQL_TOOL
    rep = routing_report(
        [("SQL_TOOL", "SQL_TOOL"), ("SQL_TOOL", "RAG_SIMPLE"), ("RAG_SIMPLE", "RAG_SIMPLE")]
    )
    assert rep.matrix["SQL_TOOL"]["RAG_SIMPLE"] == 1
    assert rep.recall["SQL_TOOL"] == 0.5
    assert abs(rep.accuracy - 2 / 3) < 1e-9
