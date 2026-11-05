"""Text analyzer shared by BM25 indexing *and* querying (docs/07-retrieval.md §6:
"different tokenizers or analyzers for indexing vs. querying" is a common bug).

Pipeline: NFKC → casefold → Unicode word tokens → stopword removal → light
suffix stemming. Identifier-like tokens (``ERR-4012``, ``v2.3.1``, ``SKU_991``)
are emitted both as the compound and as their parts, and tokens containing digits
are never stemmed, so exact-ID queries keep matching (docs/07 §1).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

#: Words joined by ``-``, ``_``, ``.``, ``/`` are kept as a compound token too.
_COMPOUND_RE = re.compile(r"\w+(?:[-_./]\w+)*", re.UNICODE)
_PART_RE = re.compile(r"[^\W_]+", re.UNICODE)

ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours yourself yourselves""".split()
)

_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("ization", "ize"),
    ("fulness", "ful"),
    ("iveness", "ive"),
    ("ements", "e"),
    ("ement", "e"),
    ("ments", ""),
    ("ment", ""),
    ("ingly", ""),
    ("edly", ""),
    ("ies", "y"),
    ("ied", "y"),
    ("sses", "ss"),
    ("ing", ""),
    ("ed", ""),
    ("ly", ""),
    ("es", ""),
    ("s", ""),
)


@lru_cache(maxsize=100_000)
def light_stem(token: str) -> str:
    """Conservative suffix stripping (S-stemmer + common English suffixes).

    Tokens with digits, short tokens and ``-ss`` endings are left alone; the stem
    must keep at least three characters.
    """
    if len(token) <= 3 or any(ch.isdigit() for ch in token) or token.endswith("ss"):
        return token
    for suffix, repl in _SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)] + repl
            if len(stem) < 3:
                continue
            if suffix == "es" and not stem.endswith(("s", "x", "z", "ch", "sh")):
                # "files" → "file", not "fil": drop only the "s".
                stem = token[:-1]
            if suffix in {"ing", "ed"} and len(stem) >= 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]  # "running" → "run"
            return stem
    return token


class Analyzer:
    """Deterministic analyzer; the *same instance config* must index and query."""

    def __init__(
        self,
        *,
        stopwords: frozenset[str] | None = ENGLISH_STOPWORDS,
        stem: bool = True,
        keep_compounds: bool = True,
        min_token_len: int = 1,
    ) -> None:
        self.stopwords = stopwords or frozenset()
        self.stem = stem
        self.keep_compounds = keep_compounds
        self.min_token_len = min_token_len

    @property
    def signature(self) -> str:
        """Identifies the analyzer config; persisted with indexes to detect mismatches."""
        return f"sw={len(self.stopwords)},stem={self.stem},comp={self.keep_compounds},min={self.min_token_len}"

    def _norm_token(self, tok: str) -> str | None:
        if len(tok) < self.min_token_len or tok in self.stopwords:
            return None
        return light_stem(tok) if self.stem else tok

    def analyze(self, text: str) -> list[str]:
        """Return index terms in order (compounds precede their parts)."""
        norm = unicodedata.normalize("NFKC", text).casefold()
        out: list[str] = []
        for m in _COMPOUND_RE.finditer(norm):
            compound = m.group(0)
            parts = _PART_RE.findall(compound)
            if self.keep_compounds and len(parts) > 1:
                out.append(compound)
            for p in parts:
                t = self._norm_token(p)
                if t:
                    out.append(t)
        return out

    __call__ = analyze


DEFAULT_ANALYZER = Analyzer()
