"""Exact and near-duplicate detection (docs/01-ingestion-parsing.md §1, "Dedup").

* **Exact:** SHA-256 of NFKC-normalised, whitespace-collapsed text
  (:attr:`Document.content_hash`).
* **Near-duplicate:** MinHash signatures over word shingles + LSH banding, with a
  Jaccard threshold of 0.9 by default. Pure Python + NumPy.

Hash arithmetic is done modulo the Mersenne prime 2**31 - 1 so that
``a * x + b`` always fits in ``uint64`` without overflow.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from collections import defaultdict
from enum import StrEnum

import numpy as np
from pydantic import BaseModel

from chatbot.core.types import Document, sha256_text

logger = logging.getLogger(__name__)

_PRIME = np.uint64((1 << 31) - 1)
_MAX_HASH = np.uint64((1 << 31) - 1)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def shingles(text: str, k: int = 5) -> set[str]:
    """Word k-shingles over normalised, lower-cased text (short texts → one shingle)."""
    words = _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())
    if not words:
        return set()
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash32(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little") % (
        (1 << 31) - 1
    )


class MinHasher:
    """MinHash signatures with ``num_perm`` universal hash permutations."""

    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 1) -> None:
        if num_perm < 8:
            raise ValueError("num_perm must be >= 8")
        if shingle_size < 1:
            raise ValueError("shingle_size must be >= 1")
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.default_rng(seed)
        self._a = rng.integers(1, int(_PRIME), size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, int(_PRIME), size=num_perm, dtype=np.uint64)

    def signature(self, text: str) -> np.ndarray:
        """``uint64`` array of shape ``(num_perm,)``. Empty text → all-max signature."""
        sh = shingles(text, self.shingle_size)
        sig = np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        if not sh:
            return sig
        hashes = np.fromiter((_hash32(s) for s in sh), dtype=np.uint64, count=len(sh))
        for i in range(0, len(hashes), 8192):  # bound memory: num_perm x batch
            batch = hashes[i : i + 8192]
            perm = (np.outer(self._a, batch) + self._b[:, None]) % _PRIME
            sig = np.minimum(sig, perm.min(axis=1))
        return sig

    @staticmethod
    def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimated Jaccard similarity: share of equal signature slots."""
        return float(np.mean(sig_a == sig_b))


def optimal_bands(num_perm: int, threshold: float) -> tuple[int, int]:
    """Choose ``(bands, rows)`` with ``bands * rows == num_perm`` whose S-curve threshold
    ``(1/b)^(1/r)`` sits just below ``threshold`` (favouring recall; candidates are verified)."""
    best: tuple[int, int] | None = None
    best_gap = float("inf")
    for rows in range(1, num_perm + 1):
        if num_perm % rows:
            continue
        bands = num_perm // rows
        t = (1 / bands) ** (1 / rows)
        gap = threshold - t
        if 0 <= gap < best_gap:
            best, best_gap = (bands, rows), gap
    return best or (num_perm, 1)


class NearDuplicateIndex:
    """MinHash LSH index. ``query`` returns stored keys with estimated Jaccard ≥ threshold."""

    def __init__(
        self, threshold: float = 0.9, num_perm: int = 128, shingle_size: int = 5, seed: int = 1
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.threshold = threshold
        self.hasher = MinHasher(num_perm=num_perm, shingle_size=shingle_size, seed=seed)
        self.bands, self.rows = optimal_bands(num_perm, threshold)
        self._buckets: list[dict[bytes, set[str]]] = [defaultdict(set) for _ in range(self.bands)]
        self._sigs: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._sigs)

    def __contains__(self, key: object) -> bool:
        return key in self._sigs

    def _band_keys(self, sig: np.ndarray) -> list[bytes]:
        return [sig[i * self.rows : (i + 1) * self.rows].tobytes() for i in range(self.bands)]

    def add(self, key: str, text: str, signature: np.ndarray | None = None) -> None:
        if key in self._sigs:
            self.remove(key)
        sig = signature if signature is not None else self.hasher.signature(text)
        self._sigs[key] = sig
        for band, bk in zip(self._buckets, self._band_keys(sig), strict=True):
            band[bk].add(key)

    def remove(self, key: str) -> None:
        sig = self._sigs.pop(key, None)
        if sig is None:
            return
        for band, bk in zip(self._buckets, self._band_keys(sig), strict=True):
            band[bk].discard(key)
            if not band[bk]:
                del band[bk]

    def query(self, text: str, signature: np.ndarray | None = None) -> list[tuple[str, float]]:
        sig = signature if signature is not None else self.hasher.signature(text)
        candidates: set[str] = set()
        for band, bk in zip(self._buckets, self._band_keys(sig), strict=True):
            candidates |= band.get(bk, set())
        scored = [(k, MinHasher.similarity(sig, self._sigs[k])) for k in candidates]
        return sorted(
            ((k, s) for k, s in scored if s >= self.threshold), key=lambda kv: (-kv[1], kv[0])
        )


class DedupStatus(StrEnum):
    UNIQUE = "unique"
    EXACT = "exact"
    NEAR = "near"


class DedupResult(BaseModel):
    status: DedupStatus
    duplicate_of: str | None = None
    similarity: float = 0.0


class Deduplicator:
    """Exact + near-duplicate check for documents (or any keyed text).

    Duplicates are reported, not silently dropped: the pipeline decides whether to
    skip, link (``metadata['duplicate_of']``) or down-rank them. Re-checking the
    same key (a new version of the same document) is never reported as a duplicate
    of itself.
    """

    def __init__(self, near_threshold: float = 0.9, num_perm: int = 128, shingle_size: int = 5) -> None:
        self._exact: dict[str, str] = {}
        self._key_hash: dict[str, str] = {}
        self.near = NearDuplicateIndex(
            threshold=near_threshold, num_perm=num_perm, shingle_size=shingle_size
        )

    def check_text(self, key: str, text: str, *, add: bool = True) -> DedupResult:
        digest = sha256_text(text)
        owner = self._exact.get(digest)
        if owner is not None and owner != key:
            return DedupResult(status=DedupStatus.EXACT, duplicate_of=owner, similarity=1.0)
        sig = self.near.hasher.signature(text)
        for other, sim in self.near.query(text, signature=sig):
            if other != key:
                return DedupResult(status=DedupStatus.NEAR, duplicate_of=other, similarity=sim)
        if add:
            self._add(key, text, digest, sig)
        return DedupResult(status=DedupStatus.UNIQUE)

    def check(self, doc: Document, *, add: bool = True) -> DedupResult:
        return self.check_text(doc.doc_id, doc.text, add=add)

    def _add(self, key: str, text: str, digest: str, sig: np.ndarray) -> None:
        old = self._key_hash.get(key)
        if old is not None and self._exact.get(old) == key:
            del self._exact[old]
        self._exact[digest] = key
        self._key_hash[key] = digest
        self.near.add(key, text, signature=sig)

    def remove(self, key: str) -> None:
        """Forget a key (e.g. on a tombstone) so a re-created document is not a 'duplicate'."""
        digest = self._key_hash.pop(key, None)
        if digest is not None and self._exact.get(digest) == key:
            del self._exact[digest]
        self.near.remove(key)
