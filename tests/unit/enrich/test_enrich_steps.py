"""Questions, metadata, summaries, glossary and the versioned Enricher pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chatbot.chunking import StructureChunker
from chatbot.core.types import ACL, Block, BlockType, Chunk, ChunkKind, Document
from chatbot.enrich import (
    BreadcrumbStep,
    ChunkSummarizer,
    ContextualEnricher,
    DocMetadata,
    DocumentSummarizer,
    Enricher,
    Glossary,
    GlossaryEnricher,
    MetadataEnricher,
    QuestionGenerator,
    RuleBasedExtractor,
    default_enricher,
    extract_dates,
    extract_identifiers,
    mine_acronyms,
)
from chatbot.llm.base import LLMRequest
from chatbot.llm.fake import FakeLLM, last_user_text

BODY = (
    "Employees accrue Paid Time Off (PTO) monthly. Effective date: 2025-01-15. "
    "Contact hr@acme.example for questions about SKU-1042 and part PN-4410-B. "
    "The policy expires on March 31, 2026. " * 3
)


def make_doc() -> Document:
    return Document(
        doc_id="hr:pto",
        version="3",
        title="PTO Policy",
        acl=ACL(allow_groups=["employees"]),
        tenant_id="acme",
        blocks=[
            Block(type=BlockType.HEADING, level=1, text="Leave"),
            Block(type=BlockType.PARAGRAPH, text=BODY),
            Block(type=BlockType.HEADING, level=1, text="Holidays"),
            Block(type=BlockType.PARAGRAPH, text="Public holidays follow the local calendar. " * 6),
        ],
    )


def make_chunks(doc: Document) -> list[Chunk]:
    return StructureChunker(max_tokens=200, min_tokens=0).chunk(doc)


async def test_questions_become_question_chunks() -> None:
    def handler(req: LLMRequest) -> str:
        assert "employee" in last_user_text(req)
        return json.dumps(
            {
                "questions": [
                    "How much PTO do I get?",
                    "how much pto do i get?",
                    "When does it expire?",
                ]
            }
        )

    doc = make_doc()
    chunks = make_chunks(doc)
    out = await QuestionGenerator(FakeLLM(handler=handler), persona="employee").apply(doc, chunks)
    qs = [c for c in out if c.kind == ChunkKind.QUESTION]
    parents = {c.chunk_id: c for c in out if c.kind == ChunkKind.CHUNK}
    assert len(qs) == 2 * len(parents)  # case-insensitive dedup
    for q in qs:
        p = parents[q.parent_id]  # type: ignore[index]
        assert q.embed_text == q.text
        assert q.acl == p.acl and q.tenant_id == "acme"
        assert (q.char_start, q.char_end) == (p.char_start, p.char_end)
    assert len({q.chunk_id for q in qs}) == len(qs)
    assert all(p.metadata["question_count"] == 2 for p in parents.values())


async def test_questions_invalid_json_degrades() -> None:
    doc = make_doc()
    out = await QuestionGenerator(FakeLLM(default="nope")).apply(doc, make_chunks(doc))
    assert all(c.kind == ChunkKind.CHUNK for c in out)


def test_rule_extractors() -> None:
    assert extract_dates("on 2025-01-15, March 31, 2026 and 5 May 2024; bad 2025-13-40") == [
        "2025-01-15",
        "2026-03-31",
        "2024-05-05",
    ]
    assert extract_identifiers("Order SKU-1042, part PN-4410-B, ticket JIRA-981, not Wi-Fi") == [
        "SKU-1042",
        "PN-4410-B",
        "JIRA-981",
    ]
    found = RuleBasedExtractor().extract(BODY)
    assert found["effective_date"] == "2025-01-15"
    assert found["expiry_date"] == "2026-03-31"
    assert found["emails"] == ["hr@acme.example"]


def test_doc_metadata_schema_validation() -> None:
    m = DocMetadata.model_validate(
        {
            "doc_type": "Policy",
            "products": ["PTO", "pto", " Leave "],
            "effective_date": "2025-01-15T00:00",
            "audience": "staff",
        }
    )
    assert m.doc_type == "policy" and m.products == ["PTO", "Leave"]
    assert m.effective_date == "2025-01-15" and m.audience is None
    assert (
        DocMetadata.model_validate({"doc_type": "memo", "effective_date": "soon"}).doc_type
        == "other"
    )


async def test_metadata_enricher_rules_and_llm() -> None:
    llm = FakeLLM(
        default=json.dumps(
            {
                "doc_type": "policy",
                "products": ["Workday"],
                "effective_date": "2025-02-01",
                "audience": "employee",
                "entities": [{"name": "ACME", "type": "organization"}],
            }
        )
    )
    doc = make_doc()
    out = await MetadataEnricher(llm).apply(doc, make_chunks(doc))
    first = out[0]
    assert first.metadata["doc_type"] == "policy"
    assert first.metadata["products"] == ["Workday"]
    assert first.metadata["effective_date"] == "2025-02-01"  # LLM overrides rule date
    assert first.metadata["audience"] == "employee"
    assert first.metadata["doc_entities"] == [{"name": "ACME", "type": "organization"}]
    leave = next(c for c in out if "SKU-1042" in c.text)
    assert {"name": "SKU-1042", "type": "identifier"} in leave.metadata["entities"]
    assert "PN-4410-B" in leave.metadata["keywords"]
    assert leave.metadata["emails"] == ["hr@acme.example"]
    assert all(c.metadata["metadata_version"] == "meta-v1+llm" for c in out)
    # rules-only + LLM failure both still work
    rules_only = await MetadataEnricher().apply(doc, make_chunks(doc))
    assert rules_only[0].metadata["effective_date"] == "2025-01-15"
    failed = await MetadataEnricher(FakeLLM(default="garbage")).apply(doc, make_chunks(doc))
    assert failed[0].metadata["effective_date"] == "2025-01-15"


async def test_document_summary_chunk() -> None:
    doc = make_doc()
    llm = FakeLLM(default="  The PTO policy explains accrual.  ")
    out = await DocumentSummarizer(llm).apply(doc, make_chunks(doc))
    summary = out[-1]
    assert summary.kind == ChunkKind.SUMMARY and summary.text == "The PTO policy explains accrual."
    assert summary.metadata["summary_of"] == "document"
    assert summary.embed_text == "PTO Policy\n\nThe PTO policy explains accrual."
    assert summary.acl == doc.acl and doc.metadata["summary"] == summary.text
    assert "<document" in last_user_text(llm.calls[0])


@pytest.mark.parametrize("mode", ["metadata", "alongside", "replace_embed"])
async def test_chunk_summary_modes(mode: str) -> None:
    doc = make_doc()
    chunks = make_chunks(doc)
    out = await ChunkSummarizer(FakeLLM(default="Short abstract."), mode=mode, min_tokens=50).apply(
        doc, chunks
    )  # type: ignore[arg-type]
    long_chunks = [c for c in out if c.kind == ChunkKind.CHUNK and c.token_count >= 50]
    assert long_chunks and all(c.metadata["summary"] == "Short abstract." for c in long_chunks)
    extra = [c for c in out if c.kind == ChunkKind.SUMMARY]
    if mode == "alongside":
        assert len(extra) == len(long_chunks)
        assert {e.parent_id for e in extra} == {c.chunk_id for c in long_chunks}
    else:
        assert extra == []
    if mode == "replace_embed":
        assert all(c.embed_text and c.embed_text.endswith("Short abstract.") for c in long_chunks)
        assert all(c.text != "Short abstract." for c in long_chunks)


def test_glossary_query_expansion() -> None:
    g = Glossary.from_mapping(
        {"PTO": ["paid time off", "vacation"], "IT": "information technology"}
    )
    assert (
        g.expand_query("How do I request PTO?") == "How do I request PTO? (paid time off; vacation)"
    )
    assert g.expand_query("how much paid time off") == "how much paid time off (PTO; vacation)"
    assert g.expand_query("is it raining") == "is it raining"  # acronym match is case-sensitive
    assert "information technology" in g.expand_query("Call IT support")
    g.add_group(["vacation", "holiday leave"])  # merges into the PTO group
    assert len(g) == 2 and "holiday leave" in g.groups[0]
    with pytest.raises(ValueError):
        g.add_group(["solo"])


def test_glossary_from_yaml_and_mining(tmp_path: Path) -> None:
    p = tmp_path / "g.yaml"
    p.write_text("SLA: [service level agreement]\n", encoding="utf-8")
    assert Glossary.from_yaml(p).expansions("our SLA") == ["service level agreement"]
    mined = mine_acronyms(
        "Employees accrue Paid Time Off (PTO). The Bank of America (BoA) and Service Level Agreement (SLA)."
    )
    assert mined["PTO"] == "Paid Time Off"
    assert mined["SLA"] == "Service Level Agreement"


async def test_glossary_enricher_and_pipeline() -> None:
    doc = make_doc()
    chunks = make_chunks(doc)
    glossary = Glossary()
    pipeline = Enricher(
        [
            MetadataEnricher(),
            BreadcrumbStep(),
            ContextualEnricher(FakeLLM(default="About PTO accrual.")),
            GlossaryEnricher(glossary, mine=True),
        ]
    )
    out = await pipeline.run(doc, chunks)
    leave = next(c for c in out if "Paid Time Off" in c.text)
    assert leave.embed_text is not None
    assert leave.embed_text.startswith("PTO Policy > Leave\n\nAbout PTO accrual.\n\n")
    assert leave.text == chunks[0].text
    assert all(c.enrichment_version == pipeline.version for c in out)
    assert out[0].metadata["enrichment_steps"][0] == "metadata@meta-v1"
    assert chunks[0].enrichment_version is None  # input untouched
    # mined PTO ↔ Paid Time Off; holidays chunk mentions neither
    assert len(glossary) == 1
    holidays = next(c for c in out if "Public holidays" in c.text)
    assert "glossary_terms" not in holidays.metadata


async def test_pipeline_versioning_and_default() -> None:
    a = Enricher([BreadcrumbStep(), ContextualEnricher(FakeLLM())])
    b = Enricher([BreadcrumbStep(), ContextualEnricher(FakeLLM(), version="ctx-v2")])
    c = Enricher([BreadcrumbStep(), ContextualEnricher(FakeLLM())])
    assert a.version != b.version and a.version == c.version
    with pytest.raises(ValueError):
        Enricher([BreadcrumbStep(), BreadcrumbStep()])
    doc = make_doc()
    offline = default_enricher(None, glossary=Glossary.from_mapping({"PTO": ["paid time off"]}))
    assert [s.name for s in offline.steps] == ["metadata", "breadcrumb", "glossary"]
    out = await offline.run(doc, make_chunks(doc))
    assert all(c.embed_text for c in out)
    full = default_enricher(
        FakeLLM(default=json.dumps({"questions": ["q?"]})), questions=True, doc_summary=True
    )
    assert [s.name for s in full.steps] == [
        "metadata",
        "breadcrumb",
        "contextual",
        "questions",
        "doc_summary",
    ]
