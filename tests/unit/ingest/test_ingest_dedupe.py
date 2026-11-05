"""Exact and near-duplicate detection."""

from __future__ import annotations

import random

import pytest

from chatbot.core.types import Block, BlockType, Document
from chatbot.ingest import Deduplicator, DedupStatus, MinHasher, NearDuplicateIndex, jaccard, shingles
from chatbot.ingest.dedupe import optimal_bands

random.seed(7)
VOCAB = [f"w{i}" for i in range(500)]
BASE = " ".join(random.choice(VOCAB) for _ in range(800))


def _doc(doc_id: str, text: str) -> Document:
    return Document(doc_id=doc_id, blocks=[Block(type=BlockType.PARAGRAPH, text=text)])


def test_exact_duplicate_ignores_whitespace_and_case_of_unicode_forms() -> None:
    d = Deduplicator()
    assert d.check(_doc("a", "Hello   world\n\nagain")).status == DedupStatus.UNIQUE
    res = d.check(_doc("b", "Hello world again"))
    assert res.status == DedupStatus.EXACT and res.duplicate_of == "a"


def test_near_duplicate_detected_and_distinct_text_unique() -> None:
    d = Deduplicator()
    assert d.check_text("orig", BASE).status == DedupStatus.UNIQUE
    words = BASE.split()
    words[400] = "CHANGED"  # one-word edit → Jaccard ≈ 0.99
    near = d.check_text("copy", " ".join(words))
    assert near.status == DedupStatus.NEAR and near.duplicate_of == "orig"
    assert near.similarity >= 0.9
    other = " ".join(random.choice(VOCAB) for _ in range(800))
    assert d.check_text("other", other).status == DedupStatus.UNIQUE


def test_moderate_edit_below_threshold_is_unique() -> None:
    d = Deduplicator(near_threshold=0.9)
    d.check_text("orig", BASE)
    words = BASE.split()
    for i in range(0, len(words), 8):  # ~12% of words changed → shingle Jaccard well below 0.9
        words[i] = f"x{i}"
    assert d.check_text("edit", " ".join(words)).status == DedupStatus.UNIQUE


def test_same_key_new_version_is_not_a_duplicate_and_remove() -> None:
    d = Deduplicator()
    d.check_text("a", BASE)
    assert d.check_text("a", BASE).status == DedupStatus.UNIQUE
    d.remove("a")
    assert d.check_text("b", BASE).status == DedupStatus.UNIQUE


def test_minhash_estimates_jaccard() -> None:
    h = MinHasher(num_perm=256)
    words = BASE.split()
    a = " ".join(words[:600])
    b = " ".join(words[200:800])
    true = jaccard(shingles(a), shingles(b))
    est = h.similarity(h.signature(a), h.signature(b))
    assert abs(true - est) < 0.1


def test_lsh_index_query_and_bands() -> None:
    bands, rows = optimal_bands(128, 0.9)
    assert bands * rows == 128 and (1 / bands) ** (1 / rows) <= 0.9
    idx = NearDuplicateIndex(threshold=0.8)
    idx.add("x", BASE)
    assert "x" in idx and len(idx) == 1
    assert idx.query(BASE)[0][0] == "x"
    idx.remove("x")
    assert idx.query(BASE) == []
    with pytest.raises(ValueError):
        NearDuplicateIndex(threshold=0)
