"""Chunkers that need models (semantic, proposition, late, RAPTOR) and code-aware chunking."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pytest

from chatbot.chunking import (
    CodeChunker,
    LateChunker,
    PropositionChunker,
    RaptorChunker,
    SemanticChunker,
    StructureChunker,
    build_chunker,
    kmeans,
    pool_spans,
    python_definitions,
    regex_definitions,
)
from chatbot.core.types import ACL, Block, BlockType, ChunkKind, Document
from chatbot.llm.base import LLMRequest
from chatbot.llm.fake import FakeLLM, last_user_text

TOPICS = {"payment": 0, "invoice": 0, "refund": 0, "garden": 1, "tomato": 1, "soil": 1}


class KeywordEmbedder:
    """2-topic embedder: counts topic keywords (deterministic)."""

    name = "kw"
    dim = 3

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().replace(".", " ").split():
            if w in TOPICS:
                v[TOPICS[w]] += 1
        v[2] = 0.01
        return v

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([self._vec(t) for t in texts])

    async def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def _doc(*paras: str, blocks: list[Block] | None = None, **kw: object) -> Document:
    return Document(
        doc_id="d",
        title="Doc",
        acl=ACL(allow_groups=["g"]),
        tenant_id="t",
        blocks=blocks or [Block(type=BlockType.PARAGRAPH, text=p) for p in paras],
        **kw,  # type: ignore[arg-type]
    )


PAY = " ".join(f"The payment invoice refund rule number {i} applies." for i in range(8))
GARDEN = " ".join(f"The garden tomato soil tip number {i} helps." for i in range(8))


async def test_semantic_chunker_splits_at_topic_change() -> None:
    doc = _doc(PAY + " " + GARDEN)
    emb = KeywordEmbedder()
    chunks = await SemanticChunker(
        emb, breakpoint_percentile=90, min_tokens=5, max_tokens=500
    ).achunk(doc)
    assert emb.calls == 1  # one batched call
    assert len(chunks) == 2
    assert "garden" not in chunks[0].text and "payment" not in chunks[1].text
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
        assert c.acl.allow_groups == ["g"]


async def test_semantic_max_guard_and_atomic_table() -> None:
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    doc = _doc(
        blocks=[Block(type=BlockType.PARAGRAPH, text=PAY), Block(type=BlockType.TABLE, text=table)]
    )
    chunks = await SemanticChunker(KeywordEmbedder(), min_tokens=1, max_tokens=40).achunk(doc)
    assert all(c.token_count <= 40 or c.text == table for c in chunks)
    assert any(c.text == table for c in chunks)


def test_semantic_sync_wrapper_outside_loop() -> None:
    chunks = SemanticChunker(KeywordEmbedder(), min_tokens=5).chunk(_doc(PAY))
    assert chunks


async def test_sync_wrapper_refuses_inside_loop() -> None:
    with pytest.raises(RuntimeError, match="achunk"):
        SemanticChunker(KeywordEmbedder()).chunk(_doc(PAY))


async def test_proposition_chunker_points_to_parent() -> None:
    def handler(req: LLMRequest) -> str:
        assert "<passage>" in last_user_text(req)
        assert "untrusted" in req.system
        return json.dumps(
            {
                "propositions": [
                    "Payments are retried 3 times.",
                    "Payments are retried 3 times.",
                    "Refunds take 5 days.",
                ]
            }
        )

    llm = FakeLLM(handler=handler)
    doc = _doc(PAY, GARDEN)
    chunks = await PropositionChunker(llm, StructureChunker(max_tokens=60, min_tokens=0)).achunk(
        doc
    )
    base = {c.chunk_id: c for c in chunks if c.kind == ChunkKind.CHUNK}
    props = [c for c in chunks if c.kind == ChunkKind.PROPOSITION]
    assert len(llm.calls) == len(base)
    assert len(props) == 2 * len(base)  # duplicates removed
    for p in props:
        parent = base[p.parent_id]  # type: ignore[index]
        assert (p.char_start, p.char_end) == (parent.char_start, parent.char_end)
        assert p.acl == parent.acl and p.metadata["offsets_from"] == "parent"
    assert len({p.chunk_id for p in props}) == len(props)


async def test_proposition_chunker_degrades_on_bad_json() -> None:
    chunks = await PropositionChunker(FakeLLM(default="not json")).achunk(_doc(PAY))
    assert chunks and all(c.kind == ChunkKind.CHUNK for c in chunks)


class FakeTokenEmbedder:
    """Token vectors = one-hot of a hash of the token, offsets from whitespace tokens."""

    name = "fake-late"
    dim = 8
    max_tokens = 4096

    def __init__(self) -> None:
        self.windows: list[str] = []

    async def embed_tokens(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        import re

        self.windows.append(text)
        offsets = [m.span() for m in re.finditer(r"\S+", text)]
        vecs = np.zeros((len(offsets), self.dim), dtype=np.float32)
        for i, (s, e) in enumerate(offsets):
            vecs[i, len(text[s:e]) % self.dim] = 1.0
            vecs[i, (i // 10) % self.dim] += 0.5  # positional context
        return vecs, offsets


def test_pool_spans_mean_pools_overlapping_tokens() -> None:
    tv = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    out = pool_spans(tv, [(0, 3), (4, 7), (8, 11)], [(0, 7), (8, 11), (20, 30)])
    np.testing.assert_allclose(out[0], np.array([0.5, 0.5]) / np.linalg.norm([0.5, 0.5]), rtol=1e-6)
    np.testing.assert_allclose(out[1], np.array([1, 1]) / np.sqrt(2), rtol=1e-6)
    assert not out[2].any()


async def test_late_chunker_vectors_aligned_and_windowed() -> None:
    te = FakeTokenEmbedder()
    doc = _doc(*(PAY for _ in range(6)))
    chunker = LateChunker(te, StructureChunker(max_tokens=80, min_tokens=0), window_tokens=200)
    res = await chunker.aembed(doc)
    assert len(res.chunks) == res.vectors.shape[0] > 1
    assert res.vectors.shape[1] == te.dim
    np.testing.assert_allclose(np.linalg.norm(res.vectors, axis=1), 1.0, rtol=1e-5)
    assert len(te.windows) > 1  # long document processed in several windows
    assert all(c.metadata["late_chunking"] is True for c in res.chunks)
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in res.chunks)


class FailingTokenEmbedder(FakeTokenEmbedder):
    async def embed_tokens(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        raise NotImplementedError("provider does not expose token outputs")


async def test_late_chunker_fallback() -> None:
    res = await LateChunker(FailingTokenEmbedder(), fallback_embedder=KeywordEmbedder()).aembed(
        _doc(PAY)
    )
    assert res.vectors.shape == (len(res.chunks), 3)
    assert all(c.metadata["late_chunking"] is False for c in res.chunks)
    only_fallback = await LateChunker(None, fallback_embedder=KeywordEmbedder()).achunk(_doc(PAY))
    assert only_fallback
    with pytest.raises(ValueError):
        LateChunker(None)


async def test_late_chunker_keeps_parents_unembedded() -> None:
    res = await LateChunker(FakeTokenEmbedder(), build_chunker("default")).aembed(_doc(PAY))
    assert res.unembedded and all(c.kind == ChunkKind.PARENT for c in res.unembedded)
    assert all(c.kind == ChunkKind.CHUNK for c in res.chunks)


PY_SRC = '''"""Module doc."""
import os
from typing import Any

CONSTANT = 3


# helper comment
@decorator
def top(a: int) -> int:
    """Top level."""
    return a + 1


class Service:
    """A service."""

    attr = 1

    def start(self) -> None:
        print("start")
        print("more")

    async def stop(self) -> None:
        print("stop")
'''


def _code_doc(src: str, lang: str | None, path: str) -> Document:
    return Document(
        doc_id="repo:" + path,
        title=path,
        doc_type="code",
        metadata={"path": path},
        blocks=[Block(type=BlockType.CODE, text=src.strip("\n"), language=lang)],
    )


def test_python_ast_chunking() -> None:
    doc = _code_doc(PY_SRC, "python", "svc/service.py")
    chunks = CodeChunker(max_tokens=512).chunk(doc)
    syms = [(c.metadata["symbol"], c.metadata["symbol_type"]) for c in chunks]
    assert syms == [("<module>", "module"), ("top", "function"), ("Service", "class")]
    top = chunks[1]
    assert top.text.startswith(
        "# helper comment\n@decorator\ndef top"
    )  # comments + decorators kept
    assert top.embed_text is not None and top.embed_text.startswith(
        "File: svc/service.py\nLanguage: python"
    )
    assert "import os" in top.embed_text
    for c in chunks:
        assert doc.text[c.char_start : c.char_end] == c.text
        assert c.metadata["code_parser"] == "ast"
    # small budget → class split by method with class header in embed_text
    small = CodeChunker(max_tokens=30).chunk(doc)
    methods = [c for c in small if c.metadata["symbol_type"] == "method"]
    assert [m.metadata["symbol"] for m in methods] == ["Service.start", "Service.stop"]
    assert all("class Service:" in (m.embed_text or "") for m in methods)
    assert methods[0].heading_path[-2:] == ["Service", "start"]
    assert any(c.metadata["symbol_type"] == "class_header" for c in small)


def test_python_definitions_and_syntax_error_fallback() -> None:
    units, imports = python_definitions(PY_SRC)
    assert imports == ["import os", "from typing import Any"]
    assert [u.symbol for u in units[-1].children] == ["Service.start", "Service.stop"]
    broken = "def ok():\n    return 1\n\ndef broken(:\n    pass\n"
    chunks = CodeChunker().chunk(_code_doc(broken, "python", "b.py"))
    assert chunks and all(c.metadata["code_parser"] == "regex" for c in chunks)


JS_SRC = """import { x } from './x';

// Adds numbers
export function add(a, b) {
  return a + b;
}

export const mul = (a, b) => a * b;

class Calc {
  run() { return 1; }
}
"""


def test_regex_fallback_for_javascript() -> None:
    units, imports = regex_definitions(JS_SRC)
    assert [u.symbol for u in units] == ["<module>", "add", "mul", "Calc"]
    assert imports == ["import { x } from './x';"]
    doc = _code_doc(JS_SRC, None, "src/calc.js")
    chunks = CodeChunker().chunk(doc)
    add = next(c for c in chunks if c.metadata["symbol"] == "add")
    assert add.text.startswith("// Adds numbers\nexport function add")
    assert add.metadata["code_language"] == "javascript"
    assert all(doc.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_code_chunker_delegates_prose() -> None:
    chunks = CodeChunker().chunk(_doc(PAY))
    assert chunks and chunks[0].metadata["chunker"] == "structure"


def test_kmeans_separates_clusters() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal([5, 0], 0.1, size=(10, 2))
    b = rng.normal([0, 5], 0.1, size=(10, 2))
    labels = kmeans(np.vstack([a, b]), 2)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1 and labels[0] != labels[10]
    assert list(kmeans(np.eye(3), 5)) == [0, 1, 2]


async def test_raptor_builds_summary_levels() -> None:
    llm = FakeLLM(handler=lambda req: "Summary of: " + str(last_user_text(req).count("<passage>")))
    paras = [PAY if i % 2 == 0 else GARDEN for i in range(12)]
    doc = _doc(*paras)
    chunker = RaptorChunker(
        KeywordEmbedder(), llm, StructureChunker(max_tokens=120, min_tokens=0), cluster_size=3
    )
    chunks = await chunker.achunk(doc)
    leaves = [c for c in chunks if c.kind == ChunkKind.CHUNK]
    summaries = [c for c in chunks if c.kind == ChunkKind.SUMMARY]
    assert len(leaves) == 12 and summaries
    levels = {s.metadata["raptor_level"] for s in summaries}
    assert 1 in levels and max(levels) >= 2
    leaf_ids = {c.chunk_id for c in leaves}
    level1 = [s for s in summaries if s.metadata["raptor_level"] == 1]
    assert {cid for s in level1 for cid in s.metadata["children_ids"]} == leaf_ids
    assert all(s.acl == doc.acl and s.tenant_id == "t" for s in summaries)
    assert all("RAPTOR" not in s.text for s in summaries)
    assert "untrusted" in llm.calls[0].system
