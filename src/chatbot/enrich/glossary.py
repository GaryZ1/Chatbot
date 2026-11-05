"""Synonym / acronym expansion (docs/03-enrichment.md §1; fixes vocabulary mismatch, failure F6).

One :class:`Glossary` is used at **index time** (append related terms to
``embed_text`` so BM25 and dense both see them) and at **query time**
(:meth:`Glossary.expand_query`). Groups are bidirectional: ``PTO`` ↔ ``paid time off``.

Acronyms (short all-caps tokens) are matched case-sensitively so ``IT`` the
department does not match ``it`` the pronoun; phrases are matched
case-insensitively on word boundaries. :func:`mine_acronyms` bootstraps a
glossary from definitions in the corpus ("Paid Time Off (PTO)").
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from chatbot.core.types import Chunk, ChunkKind, Document

logger = logging.getLogger(__name__)

_ACRONYM = re.compile(r"^[A-Z][A-Z0-9&]{1,7}s?$")
_MINOR_WORDS = frozenset({"of", "and", "the", "for", "to", "in", "on", "a", "an", "&"})


def _is_acronym(term: str) -> bool:
    return bool(_ACRONYM.match(term)) and sum(ch.isupper() for ch in term) >= 2


def _pattern(term: str) -> re.Pattern[str]:
    escaped = r"\s+".join(re.escape(p) for p in term.split())
    flags = 0 if _is_acronym(term) else re.IGNORECASE
    return re.compile(rf"(?<![\w-]){escaped}(?![\w-])", flags)


class Glossary:
    """Bidirectional synonym groups."""

    def __init__(self, groups: Iterable[Iterable[str]] = ()) -> None:
        self._groups: list[list[str]] = []
        self._patterns: list[tuple[re.Pattern[str], int, str]] = []
        for g in groups:
            self.add_group(g)

    # ------------------------------------------------------------------ construction

    def add_group(self, terms: Iterable[str]) -> None:
        clean = list(dict.fromkeys(" ".join(t.split()) for t in terms if t and t.strip()))
        if len(clean) < 2:
            raise ValueError(f"a synonym group needs at least two terms: {clean}")
        # merge with an existing group sharing a term
        for gi, existing in enumerate(self._groups):
            if {t.casefold() for t in existing} & {t.casefold() for t in clean}:
                merged = list(dict.fromkeys([*existing, *clean]))
                self._groups[gi] = merged
                self._rebuild()
                return
        self._groups.append(clean)
        gi = len(self._groups) - 1
        for t in clean:
            self._patterns.append((_pattern(t), gi, t))
        self._sort()

    def _rebuild(self) -> None:
        self._patterns = [(_pattern(t), gi, t) for gi, g in enumerate(self._groups) for t in g]
        self._sort()

    def _sort(self) -> None:
        self._patterns.sort(key=lambda p: -len(p[2]))  # longest terms first

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Sequence[str] | str]) -> Glossary:
        """``{"PTO": ["paid time off", "vacation"]}`` → glossary."""
        return cls([[k, *([v] if isinstance(v, str) else v)] for k, v in mapping.items()])

    @classmethod
    def from_yaml(cls, path: str | Path) -> Glossary:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if isinstance(data, list):
            return cls(data)
        if isinstance(data, dict):
            return cls.from_mapping(data)
        raise ValueError(f"glossary file {path} must contain a mapping or a list of groups")

    @property
    def groups(self) -> list[list[str]]:
        return [list(g) for g in self._groups]

    def __len__(self) -> int:
        return len(self._groups)

    # ------------------------------------------------------------------ matching

    def matches(self, text: str) -> list[tuple[str, list[str]]]:
        """``[(term found, [synonyms not present in text])]`` per matched group."""
        found: dict[int, str] = {}
        taken: list[tuple[int, int]] = []
        for pat, gi, term in self._patterns:
            if gi in found:
                continue
            for m in pat.finditer(text):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                found[gi] = term
                taken.append(m.span())
                break
        out: list[tuple[str, list[str]]] = []
        for gi, term in found.items():
            missing = [t for t in self._groups[gi] if t != term and not _pattern(t).search(text)]
            if missing:
                out.append((term, missing))
        return out

    def expansions(self, text: str) -> list[str]:
        return [syn for _, syns in self.matches(text) for syn in syns]

    def expand_query(self, query: str, *, max_terms: int = 6) -> str:
        """Append missing synonyms to a query: ``"PTO policy"`` → ``"PTO policy (paid time off)"``."""
        extra = self.expansions(query)[:max_terms]
        return f"{query} ({'; '.join(extra)})" if extra else query

    def annotate(self, text: str) -> str | None:
        """``"Related terms: PTO = paid time off; …"`` for index-time injection, or ``None``."""
        found = self.matches(text)
        if not found:
            return None
        return "Related terms: " + "; ".join(f"{t} = {', '.join(s)}" for t, s in found)


def mine_acronyms(text: str, *, max_words: int = 8) -> dict[str, str]:
    """Find ``"Long Form (LF)"`` definitions whose initials spell the acronym."""
    out: dict[str, str] = {}
    for m in re.finditer(r"\(([A-Z][A-Za-z0-9&]{1,7})\)", text):
        acr = m.group(1)
        letters = [c.lower() for c in acr if c.isalpha() and c.isupper()]
        if len(letters) < 2 or not _is_acronym(acr.rstrip("s")):
            continue
        words = re.findall(r"[\w&'-]+", text[max(0, m.start() - 120) : m.start()])[-max_words:]
        for k in range(len(words) - 1, -1, -1):
            cand = words[k:]
            initials = [w[0].lower() for w in cand if w.lower() not in _MINOR_WORDS]
            if initials == letters:
                out.setdefault(acr, " ".join(cand))
                break
            if len(initials) > len(letters):
                break
    return out


class GlossaryEnricher:
    """Enrichment step: append related terms to ``embed_text`` and ``metadata['glossary_terms']``.

    With ``mine=True`` acronym definitions found in the document are added to the
    glossary first (so later documents and queries benefit too).
    """

    name = "glossary"

    def __init__(
        self,
        glossary: Glossary,
        *,
        mine: bool = False,
        kinds: Sequence[ChunkKind] = (ChunkKind.CHUNK, ChunkKind.QUESTION),
        version: str = "gloss-v1",
    ) -> None:
        self.glossary = glossary
        self.mine = mine
        self.kinds = frozenset(kinds)
        digest = hashlib.sha256(repr(glossary.groups).encode()).hexdigest()[:8]
        self.version = f"{version}-{digest}{'-mined' if mine else ''}"

    async def apply(self, doc: Document, chunks: list[Chunk]) -> list[Chunk]:
        if self.mine:
            for acr, long_form in mine_acronyms(doc.text).items():
                try:
                    self.glossary.add_group([acr, long_form])
                except ValueError:
                    logger.debug("skipping invalid mined acronym %s", acr)
        for c in chunks:
            if c.kind not in self.kinds:
                continue
            found = self.glossary.matches(c.text)
            if not found:
                continue
            note = "Related terms: " + "; ".join(f"{t} = {', '.join(s)}" for t, s in found)
            c.embed_text = f"{c.embed_text or c.text}\n\n{note}"
            c.metadata["glossary_terms"] = sorted({s for _, syns in found for s in syns})
        return chunks
