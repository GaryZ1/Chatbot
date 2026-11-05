"""Dependency-free lexical utilities shared by compression, citation checks and memory.

These implement the "lexical overlap" defaults referenced in
docs/10-generation-context.md §3 (cheap citation-support check) and the
query-relevance scoring used by extractive compression. They are deliberately
simple, deterministic and fast; swap in an NLI model or an LLM judge where
precision matters more than cost.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WORD_RE = re.compile(r"[\w][\w.\-/]*[\w]|[\w]", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}|\n(?=\s*[-*•]\s)")

#: Small English stopword list; enough to stop function words dominating overlap.
_STOPWORDS_TEXT = """
    a an and are as at be been being but by can could did do does doing for from had has have
    having he her here hers him his how i if in into is it its itself just me more most my no nor
    not of off on once only or other our ours out over own same she should so some such than that
    the their theirs them then there these they this those through to too under until up very was
    we were what when where which while who whom why will with would you your yours yourself also
    about above after again against all am any because before below between both down during each
    few further let lets may might must shall us via per within without
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())


def normalize_text(text: str) -> str:
    """NFKC-normalise and lowercase (casefold) text for matching."""
    return unicodedata.normalize("NFKC", text).casefold()


def terms(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Content terms in order: casefolded word tokens, stopwords removed.

    Tokens may contain inner ``.-/`` so identifiers such as ``SKU-1234`` or
    ``v2.1`` stay intact.
    """
    out = [m.group(0) for m in _WORD_RE.finditer(normalize_text(text))]
    if keep_stopwords:
        return out
    return [t for t in out if t not in STOPWORDS]


def term_set(text: str) -> set[str]:
    return set(terms(text))


def overlap_ratio(needle: Iterable[str], haystack: Iterable[str]) -> float:
    """Fraction of ``needle`` terms that occur in ``haystack`` (0 when needle is empty)."""
    n = set(needle)
    if not n:
        return 0.0
    return len(n & set(haystack)) / len(n)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter (sentence punctuation, blank lines, list bullets)."""
    parts = [p.strip() for p in _SENT_RE.split(text)]
    return [p for p in parts if p]
