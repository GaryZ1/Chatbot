"""Dependency-free deterministic embedder (test / offline default).

Implements the *feature hashing* trick over a TF-IDF-like bag of features:
word unigrams, light word stems, word bigrams and character n-grams of each word (with boundary
markers so prefixes/suffixes are distinguishable). Each feature is hashed with
BLAKE2b into one of ``dim`` buckets with a ±1 sign (signed hashing keeps collisions
unbiased), weighted by sublinear term frequency ``1 + log(tf)`` and a static
"IDF-ish" prior that down-weights stopwords and very short tokens. The result is
L2-normalised by :class:`~chatbot.embed.base.BaseEmbedder`.

It is not semantic, but texts with overlapping words or morphology (``"refund"``
vs ``"refunds"``) get high cosine similarity, which is what tests and local
development need. It is symmetric, so no query/document prefixes are required.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import numpy as np

from chatbot.embed.base import BaseEmbedder

_WORD_RE = re.compile(r"\w+", re.UNICODE)

#: Compact English stopword list; such tokens get a low weight rather than zero.
STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


_STEM_SUFFIXES = ("ing", "ies", "ed", "es", "s")


def _stem(word: str) -> str:
    """Tiny suffix stripper so inflections share a feature ("refunds" ~ "refund")."""
    if len(word) <= 4 or any(ch.isdigit() for ch in word):
        return word
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)] + ("y" if suf == "ies" else "")
    return word


@lru_cache(maxsize=200_000)
def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


class HashingEmbedder(BaseEmbedder):
    """Feature-hashed word + char-n-gram vectors. Deterministic across processes."""

    def __init__(
        self,
        dim: int = 384,
        *,
        char_ngrams: tuple[int, int] = (3, 5),
        word_weight: float = 1.0,
        bigram_weight: float = 0.5,
        char_weight: float = 0.35,
        stopword_weight: float = 0.15,
        batch_size: int = 256,
        version: str = "1",
    ) -> None:
        lo, hi = char_ngrams
        if lo < 1 or hi < lo:
            raise ValueError("char_ngrams must satisfy 1 <= lo <= hi")
        super().__init__(
            name="hashing",
            dim=dim,
            version=version,
            query_prefix="",
            document_prefix="",
            batch_size=batch_size,
        )
        self.char_ngrams = char_ngrams
        self.word_weight = word_weight
        self.bigram_weight = bigram_weight
        self.char_weight = char_weight
        self.stopword_weight = stopword_weight

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _WORD_RE.findall(unicodedata.normalize("NFKC", text).casefold())

    def _features(self, text: str) -> Counter[str]:
        feats: Counter[str] = Counter()
        words = self.tokenize(text)
        lo, hi = self.char_ngrams
        for w in words:
            feats["w:" + w] += 1
            if w not in STOPWORDS:
                feats["s:" + _stem(w)] += 1
            if w in STOPWORDS:
                continue
            padded = f"<{w}>"
            for n in range(lo, hi + 1):
                for i in range(0, max(len(padded) - n + 1, 0)):
                    feats["c:" + padded[i : i + n]] += 1
        for a, b in itertools.pairwise(words):
            if a in STOPWORDS and b in STOPWORDS:
                continue
            feats[f"b:{a} {b}"] += 1
        return feats

    def _weight(self, feature: str) -> float:
        kind, _, body = feature.partition(":")
        if kind == "w":
            if body in STOPWORDS:
                return self.stopword_weight
            # IDF-ish prior: very short tokens are less informative, long ones (IDs) more.
            return self.word_weight * min(1.5, 0.5 + len(body) / 6.0)
        if kind == "s":
            return self.word_weight * min(1.5, 0.5 + len(body) / 6.0)
        if kind == "b":
            return self.bigram_weight
        return self.char_weight

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feature, tf in self._features(text).items():
            idx, sign = _bucket(feature, self.dim)
            vec[idx] += sign * self._weight(feature) * (1.0 + math.log(tf))
        return vec

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        if len(texts) > 64:
            return await asyncio.to_thread(lambda: np.vstack([self.embed_one(t) for t in texts]))
        return np.vstack([self.embed_one(t) for t in texts])
