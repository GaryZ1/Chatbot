"""Deterministic chunkers: offsets, budgets, atomic tables/code, metadata, factory."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chatbot.chunking import (
    BaseChunker,
    DocIndex,
    FixedCharChunker,
    PageChunker,
    ParagraphChunker,
    ParentChildChunker,
    RecursiveChunker,
    RecursiveSplitter,
    SentenceChunker,
    SentenceWindowChunker,
    SlidingWindowChunker,
    StructureChunker,
    TableChunker,
    available_chunkers,
    block_spans,
    breadcrumb,
    build_chunker,
    default_chunker,
    register_chunker,
    split_sentences,
)
from chatbot.chunking.base import trim_span
from chatbot.core.tokens import default_tokenizer
from chatbot.core.types import ACL, Block, BlockType, Chunk, ChunkKind, Document

TOK = default_tokenizer()
PROSE = (
    "Payment retries follow an exponential schedule. The first retry happens after one hour. "
    "Customers are notified by email after the second failure. Dr. Smith approved the policy, "
    "e.g. for EU customers, on 3.5 days of notice. "
)
TABLE = "| Attempt | Delay | Channel |\n| --- | --- | --- |\n" + "\n".join(
    f"| {i} | {i * 2}h | email |" for i in range(1, 9)
)
BIG_TABLE = "| Attempt | Delay |\n| --- | --- |\n" + "\n".join(
    f"| {i} | {i * 3}h |" for i in range(1, 200)
)
CODE = "def retry(payment):\n\n    for attempt in range(3):\n\n        charge(payment)\n" * 5


def rich_doc() -> Document:
    blocks = [
        Block(type=BlockType.HEADING, level=1, text="Payment Retry Policy", page=1),
        Block(type=BlockType.PARAGRAPH, text="Short intro.", page=1),
        Block(type=BlockType.HEADING, level=2, text="Schedule", page=1),
        *[Block(type=BlockType.PARAGRAPH, text=PROSE * 3, page=1) for _ in range(3)],
        Block(type=BlockType.TABLE, text=TABLE, html="<table></table>", page=2),
        Block(type=BlockType.HEADING, level=2, text="Implementation", page=2),
        Block(type=BlockType.CODE, text=CODE, language="python", page=2),
        Block(type=BlockType.PARAGRAPH, text=PROSE * 12, page=3),
        Block(type=BlockType.HEADING, level=3, text="Edge cases", page=3),
        Block(
            type=BlockType.LIST, text="- Expired cards\n- Fraud holds\n- Partial captures", page=3
        ),
        Block(type=BlockType.FIGURE, text="", image_ref="s3://x.png", page=3),
        Block(type=BlockType.HEADING, level=1, text="Appendix", page=4),
        Block(type=BlockType.PARAGRAPH, text="See the runbook.", page=4),
    ]
    return Document(
        doc_id="confluence:ENG:1",
        version="v17",
        source_uri="https://wiki/x",
        title="Payment Retry Policy",
        breadcrumbs=["Engineering", "Payments"],
        language="en",
        updated_at=datetime(2025, 3, 1, tzinfo=UTC),
        tenant_id="acme",
        acl=ACL(allow_groups=["eng-payments"], deny_users=["mallory"]),
        doc_type="policy",
        blocks=blocks,
        metadata={"region": "EMEA"},
    )


SYNC_CHUNKERS: list[BaseChunker] = [
    FixedCharChunker(chunk_size=400, overlap=50),
    build_chunker("fixed_token", max_tokens=100, overlap_tokens=10),
    SentenceChunker(max_tokens=120),
    SentenceChunker(sentences_per_chunk=3, overlap_sentences=1),
    ParagraphChunker(max_tokens=200),
    SlidingWindowChunker(window_tokens=100, overlap=0.5),
    PageChunker(),
    PageChunker(max_tokens=150),
    RecursiveChunker(max_tokens=128),
    RecursiveChunker(max_tokens=128, overlap_tokens=20),
    StructureChunker(max_tokens=200),
    StructureChunker(max_tokens=200, table_strategy="row_groups"),
    ParentChildChunker(child_max_tokens=128, parent_max_tokens=600),
    SentenceWindowChunker(window=1),
    TableChunker(max_tokens=60),
    default_chunker(),
    build_chunker("code"),
]


def _ids(chunkers: list[BaseChunker]) -> list[str]:
    return [f"{c.name}-{i}" for i, c in enumerate(chunkers)]


@pytest.mark.parametrize("chunker", SYNC_CHUNKERS, ids=_ids(SYNC_CHUNKERS))
def test_offsets_map_back_exactly(chunker: BaseChunker) -> None:
    doc = rich_doc()
    text = doc.text
    chunks = chunker.chunk(doc)
    assert chunks
    for c in chunks:
        assert c.char_start is not None and c.char_end is not None
        assert 0 <= c.char_start < c.char_end <= len(text)
        if c.metadata.get("header_repeated"):
            # row group: header + exact rows span
            assert c.text.endswith(text[c.char_start : c.char_end])
        else:
            assert text[c.char_start : c.char_end] == c.text
        assert c.text == c.text.strip()
        assert c.token_count == TOK.count(c.text)


@pytest.mark.parametrize("chunker", SYNC_CHUNKERS, ids=_ids(SYNC_CHUNKERS))
def test_document_fields_copied(chunker: BaseChunker) -> None:
    doc = rich_doc()
    for c in chunker.chunk(doc):
        assert c.doc_id == doc.doc_id and c.doc_version == "v17"
        assert c.tenant_id == "acme"
        assert c.acl == doc.acl and c.acl is not doc.acl
        assert c.source_uri == "https://wiki/x"
        assert c.title == "Payment Retry Policy"
        m = c.metadata
        assert m["doc_type"] == "policy" and m["language"] == "en"
        assert m["updated_at"] == "2025-03-01T00:00:00+00:00"
        assert m["is_latest"] is True and m["region"] == "EMEA"
        assert m["breadcrumbs"] == ["Engineering", "Payments"]
        assert c.page is not None


@pytest.mark.parametrize("chunker", SYNC_CHUNKERS, ids=_ids(SYNC_CHUNKERS))
def test_ids_unique_and_deterministic(chunker: BaseChunker) -> None:
    a = chunker.chunk(rich_doc())
    b = chunker.chunk(rich_doc())
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


STRUCTURE_AWARE: list[BaseChunker] = [
    SentenceChunker(max_tokens=60),
    ParagraphChunker(max_tokens=60, min_tokens=10),
    StructureChunker(max_tokens=60, min_tokens=10),
    ParentChildChunker(child_max_tokens=60, parent_max_tokens=200),
    default_chunker(max_tokens=60),
]


@pytest.mark.parametrize("chunker", STRUCTURE_AWARE, ids=_ids(STRUCTURE_AWARE))
def test_tables_and_code_never_split(chunker: BaseChunker) -> None:
    doc = rich_doc()
    spans = [
        (s.block.type, *trim_span(doc.text, s.start, s.end))
        for s in block_spans(doc)
        if s.block.type in (BlockType.TABLE, BlockType.CODE)
    ]
    chunks = [c for c in chunker.chunk(doc) if c.kind == ChunkKind.CHUNK]
    for btype, start, end in spans:
        containing = [c for c in chunks if c.char_start <= start and c.char_end >= end]  # type: ignore[operator]
        assert containing, f"{btype} block was split"
        partial = [
            c
            for c in chunks
            if c.char_start < end and c.char_end > start and c not in containing  # type: ignore[operator]
        ]
        assert not partial


def test_structure_budget_breadcrumbs_and_tiny_merge() -> None:
    doc = rich_doc()
    chunks = StructureChunker(max_tokens=200, min_tokens=50).chunk(doc)
    for c in chunks:
        if not c.metadata.get("atomic"):
            assert c.token_count <= 200
        assert c.embed_text is not None
        crumb = breadcrumb(doc.title, c.heading_path)
        assert c.embed_text.startswith(crumb + "\n\n")
        assert c.embed_text.endswith(c.text)
    # "Short intro." (tiny) merged forward with the Schedule section
    first = chunks[0]
    assert first.text.startswith("Payment Retry Policy\n\nShort intro.\n\nSchedule")
    assert first.heading_path == ["Payment Retry Policy", "Schedule"]
    assert first.embed_text.startswith("Payment Retry Policy > Schedule\n\n")  # title not repeated
    # table chunk carries an NL summary in embed_text, not in text
    table = next(c for c in chunks if "| Attempt |" in c.text)
    assert "Table with columns Attempt, Delay, Channel" in (table.embed_text or "")
    assert "Table with columns" not in table.text
    edge = next(c for c in chunks if "Expired cards" in c.text)
    assert edge.heading_path[:2] == ["Payment Retry Policy", "Implementation"]
    appendix = next(c for c in chunks if "See the runbook." in c.text)
    assert appendix.heading_path == ["Appendix"]  # unrelated top-level sections never glued
    assert "Expired cards" not in appendix.text
    assert all(c.token_count >= 50 for c in chunks[:-1] if not c.metadata.get("atomic"))


def test_merge_heading_paths() -> None:
    from chatbot.chunking.base import merge_heading_paths

    assert merge_heading_paths(["A"], ["A", "B"]) == ("A", "B")
    assert merge_heading_paths(["A"], ["A", "B"], prefer_first=True) == ("A",)
    assert merge_heading_paths(["A", "B"], ["A", "C"]) == ("A",)
    assert merge_heading_paths(["A"], ["Z"]) == ()


def test_structure_big_table_atomic_vs_row_groups() -> None:
    doc = Document(
        doc_id="t",
        title="Rates",
        blocks=[
            Block(type=BlockType.HEADING, level=1, text="Rates"),
            Block(type=BlockType.TABLE, text=BIG_TABLE),
        ],
    )
    atomic = StructureChunker(max_tokens=100).chunk(doc)
    tables = [c for c in atomic if "| Attempt |" in c.text]
    assert len(tables) == 1 and tables[0].metadata["oversize"] is True
    groups = StructureChunker(max_tokens=100, table_strategy="row_groups").chunk(doc)
    rows = [c for c in groups if c.metadata.get("header_repeated")]
    assert len(rows) > 3
    for c in rows:
        assert c.text.startswith("| Attempt | Delay |\n| --- | --- |\n")
        assert c.token_count <= 100
    covered = sum(c.metadata["row_end"] - c.metadata["row_start"] for c in rows)
    assert covered == 199
    assert "Rows 1-" in (rows[0].embed_text or "")


def test_parent_child_nesting() -> None:
    doc = rich_doc()
    chunks = ParentChildChunker(child_max_tokens=100, parent_max_tokens=400).chunk(doc)
    parents = {c.chunk_id: c for c in chunks if c.kind == ChunkKind.PARENT}
    children = [c for c in chunks if c.kind == ChunkKind.CHUNK]
    assert parents and len(children) > len(parents)
    for p in parents.values():
        assert p.parent_id is None
        if not p.metadata.get("atomic"):
            assert p.token_count <= 400
    for ch in children:
        p = parents[ch.parent_id]  # type: ignore[index]
        assert p.char_start <= ch.char_start and ch.char_end <= p.char_end  # type: ignore[operator]
        assert ch.chunk_id in p.metadata["child_ids"]
        if not ch.metadata.get("atomic"):
            assert ch.token_count <= 100
    assert [c.position for c in children] == list(range(len(children)))


def test_default_recipe() -> None:
    chunker = default_chunker()
    assert isinstance(chunker, ParentChildChunker)
    assert chunker.child_max_tokens == 512 and chunker.parent_max_tokens == 1500
    chunks = chunker.chunk(rich_doc())
    kinds = {c.kind for c in chunks}
    assert kinds == {ChunkKind.PARENT, ChunkKind.CHUNK}
    assert all(c.embed_text and c.embed_text.startswith("Payment Retry Policy") for c in chunks)


def test_sentence_splitter_handles_abbreviations() -> None:
    text = "Dr. Smith approved it, e.g. for EU users. Rates rose 3.5 percent! Next one?\n- bullet"
    sents = [text[s:e] for s, e in split_sentences(text)]
    assert sents == [
        "Dr. Smith approved it, e.g. for EU users.",
        "Rates rose 3.5 percent!",
        "Next one?",
        "- bullet",
    ]


def test_sentence_window_metadata() -> None:
    doc = rich_doc()
    chunks = SentenceWindowChunker(window=1).chunk(doc)
    mid = chunks[5]
    assert mid.text in mid.metadata["window_text"]
    assert (
        doc.text[mid.metadata["window_start"] : mid.metadata["window_end"]]
        == mid.metadata["window_text"]
    )


def test_page_chunker_one_chunk_per_page() -> None:
    chunks = PageChunker().chunk(rich_doc())
    assert [c.page for c in chunks] == [1, 2, 3, 4]


def test_fixed_char_overlap_and_sliding_window() -> None:
    doc = rich_doc()
    chunks = FixedCharChunker(chunk_size=300, overlap=100).chunk(doc)
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[1].char_start < chunks[0].char_end  # type: ignore[operator]
    sw = SlidingWindowChunker(window_tokens=80, overlap=0.5).chunk(doc)
    assert all(c.token_count <= 80 for c in sw)
    assert sw[1].char_start < sw[0].char_end  # type: ignore[operator]
    assert sw[0].embed_text is not None and FixedCharChunker().chunk(doc)[0].embed_text is None


def test_recursive_splitter_respects_limit_and_overlap() -> None:
    text = (PROSE * 40).strip()
    sp = RecursiveSplitter(max_tokens=50, overlap_tokens=10)
    spans = sp.split(text)
    assert len(spans) > 5
    assert all(TOK.count(text[s:e]) <= 50 for s, e in spans)
    assert any(spans[i + 1][0] < spans[i][1] for i in range(len(spans) - 1))
    # hard split for a separator-free blob
    blob = "x" * 5000
    assert all(TOK.count(blob[s:e]) <= 50 for s, e in RecursiveSplitter(max_tokens=50).split(blob))


def test_doc_index_lookups() -> None:
    doc = rich_doc()
    idx = DocIndex(doc)
    assert idx.text == doc.text
    code_span = next(s for s in idx.spans if s.block.type == BlockType.CODE)
    assert idx.heading_path_at(code_span.start + 3) == ("Payment Retry Policy", "Implementation")
    assert idx.page_at(code_span.start) == 2
    assert idx.spans_in(code_span.start, code_span.end) == [code_span]


def test_empty_document_and_validation() -> None:
    empty = Document(doc_id="e")
    for name in ("structure", "recursive", "parent_child", "default", "sentence", "page"):
        assert build_chunker(name).chunk(empty) == []
    with pytest.raises(ValueError):
        StructureChunker(max_tokens=0)
    with pytest.raises(ValueError):
        RecursiveChunker(max_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        ParentChildChunker(child_max_tokens=500, parent_max_tokens=100)


def test_factory_registry() -> None:
    assert {"structure", "semantic", "late", "raptor", "default"} <= set(available_chunkers())
    assert isinstance(build_chunker("structure_aware", max_tokens=300), StructureChunker)
    with pytest.raises(ValueError, match="unknown chunker"):
        build_chunker("nope")
    with pytest.raises(ValueError, match="invalid parameters"):
        build_chunker("structure", bogus=1)

    class OneChunk(BaseChunker):
        name = "one"

        def chunk(self, doc: Document) -> list[Chunk]:
            idx = DocIndex(doc)
            return [self.make_chunk(idx, position=0, start=0, end=len(idx.text))]

    register_chunker("one_chunk", OneChunk, replace=True)
    assert len(build_chunker("one_chunk").chunk(rich_doc())) == 1
    with pytest.raises(ValueError):
        register_chunker("one_chunk", OneChunk)


async def test_achunk_adapter() -> None:
    chunks = await StructureChunker().achunk(rich_doc())
    assert chunks == StructureChunker().chunk(rich_doc())
