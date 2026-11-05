"""Spelling-safe query normalisation (docs/06-query-understanding.md §1, "Spelling / normalization").

The doc's warning for this technique is that it "can 'fix' valid product codes".
Everything here is therefore conservative:

* Unicode NFKC, smart quotes/dashes → ASCII, control characters removed, whitespace collapsed.
* Glossary acronym expansion *appends* the expansion (``PTO`` → ``PTO (paid time off)``) so
  the original token stays available for lexical (BM25) matching.
* Tokens that look like identifiers (SKUs, error codes, versions, ticket numbers, hashes)
  are never expanded, re-cased or otherwise altered.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
        "​": "",
        "‌": "",
        "‍": "",
        "﻿": "",
    }
)

#: Patterns for identifier-like tokens (product codes, ticket ids, error codes, versions...).
_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^[A-Za-z]+[-_./:#]?\d[\w\-./:#]*$"),  # SKU-1234, AB12, X100, ERR_42, JIRA-77
    re.compile(r"^\d+[-_./:]?[A-Za-z][\w\-./]*$"),  # 4K-200, 12b, 3M-tape
    re.compile(r"^[vV]?\d+(?:\.\d+){1,3}[\w\-.]*$"),  # 1.2.3, v2.10.0-rc1
    re.compile(r"^#?\d{4,}$"),  # order numbers / ticket numbers (#12345, 1234567)
    re.compile(r"^(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}$"),  # git / content hashes
    re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$"),  # E_CONN_RESET, MAX_RETRY_COUNT
    re.compile(r"^[\w.\-]+(?:/[\w.\-]+)+$"),  # paths / api routes (orders/v2/refund)
)

_TOKEN_RE = re.compile(r"[\w#][\w\-./:#]*[\w]|[\w#]", re.UNICODE)


def looks_like_identifier(token: str) -> bool:
    """True for tokens that must be matched exactly (codes, ids, versions, hashes).

    Plain acronyms (``PTO``, ``SLA``) and plain numbers shorter than four digits are
    *not* identifiers; years such as ``2025`` are treated as identifiers (exact match).
    """
    tok = token.strip(".,;:!?()[]{}\"'")
    if len(tok) < 2:
        return False
    return any(p.match(tok) for p in _IDENTIFIER_PATTERNS)


def find_identifiers(text: str) -> list[str]:
    """Identifier-like tokens in order of appearance (deduplicated)."""
    seen: dict[str, None] = {}
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0).strip(".,;:")
        if looks_like_identifier(tok):
            seen.setdefault(tok, None)
    return list(seen)


def contains_identifier(text: str) -> bool:
    return bool(find_identifiers(text))


def clean_text(text: str) -> str:
    """Unicode / whitespace clean-up only; never changes word content."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return " ".join(text.split())


class QueryNormalizer:
    """Deterministic, spelling-safe normaliser with glossary acronym expansion.

    Args:
        glossary: acronym/term → expansion. Keys are matched as whole tokens,
            case-sensitively for all-caps keys (``IT`` must not match "it") and
            case-insensitively otherwise. Keys that themselves look like
            identifiers are ignored with a warning-free skip (they are codes, not acronyms).
        expand: when False only clean-up is performed.
    """

    def __init__(self, glossary: Mapping[str, str] | None = None, *, expand: bool = True) -> None:
        self.expand = expand
        self._exact: dict[str, str] = {}
        self._folded: dict[str, str] = {}
        for key, value in (glossary or {}).items():
            k = key.strip()
            if not k or not value.strip() or looks_like_identifier(k):
                continue
            if k.isupper():
                self._exact[k] = value.strip()
            else:
                self._folded[k.casefold()] = value.strip()

    def _expansion(self, token: str) -> str | None:
        if looks_like_identifier(token):
            return None
        if token in self._exact:
            return self._exact[token]
        return self._folded.get(token.casefold())

    def normalize(self, text: str) -> str:
        """Clean the text and append glossary expansions after matching acronyms."""
        cleaned = clean_text(text)
        if not self.expand or not (self._exact or self._folded):
            return cleaned
        out: list[str] = []
        last = 0
        expanded: set[str] = set()
        for m in re.finditer(r"[\w][\w\-./]*", cleaned):
            tok = m.group(0).rstrip(".")
            exp = self._expansion(tok)
            if exp is None or tok in expanded or exp.casefold() in cleaned.casefold():
                continue
            end = m.start() + len(tok)
            out.append(cleaned[last:end])
            out.append(f" ({exp})")
            last = end
            expanded.add(tok)
        out.append(cleaned[last:])
        return "".join(out)

    @staticmethod
    def cache_key(text: str) -> str:
        """Key used for plan caching: cleaned + casefolded, trailing punctuation removed."""
        return clean_text(text).casefold().rstrip(" ?!.")
