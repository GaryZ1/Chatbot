"""Evidence bookkeeping shared by the agentic loops (docs/09-advanced-rag.md §4).

Every chunk an agent sees gets a stable ``[n]`` number for the lifetime of one
run, so the final answer can cite gathered evidence and citations can be mapped
back to chunk ids, documents and pages (docs/10-generation-context.md citations).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from html import escape

from chatbot.core.types import Chunk, Citation

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_ABSTAIN_RE = re.compile(
    r"^\s*(i don't know|i do not know|i couldn't find|i could not find|the (?:available )?(?:documents|sources) do not)",
    re.IGNORECASE,
)

#: Default per-chunk character cap when rendering evidence into a prompt.
DEFAULT_MAX_CHARS = 2000


class EvidenceStore:
    """Assigns 1-based evidence numbers to chunks (dedup by chunk id)."""

    def __init__(self) -> None:
        self._index: dict[str, int] = {}
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunk: Chunk) -> int:
        n = self._index.get(chunk.chunk_id)
        if n is None:
            self._chunks.append(chunk)
            n = len(self._chunks)
            self._index[chunk.chunk_id] = n
        return n

    def add_all(self, chunks: Iterable[Chunk]) -> list[int]:
        return [self.add(c) for c in chunks]

    def get(self, n: int) -> Chunk | None:
        return self._chunks[n - 1] if 1 <= n <= len(self._chunks) else None

    def number_of(self, chunk_id: str) -> int | None:
        return self._index.get(chunk_id)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def render(self, numbers: Iterable[int], max_chars: int = DEFAULT_MAX_CHARS) -> str:
        return "\n".join(
            format_evidence(n, c, max_chars) for n in numbers if (c := self.get(n)) is not None
        )


def format_evidence(n: int, chunk: Chunk, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Render one chunk as an ``<evidence>`` XML element (attribute values escaped)."""
    text = chunk.text if len(chunk.text) <= max_chars else chunk.text[:max_chars] + " …"
    attrs = {
        "n": str(n),
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "section": " > ".join(chunk.heading_path),
        "position": str(chunk.position),
    }
    if chunk.page is not None:
        attrs["page"] = str(chunk.page)
    attr = " ".join(f'{k}="{escape(v, quote=True)}"' for k, v in attrs.items() if v)
    return f"<evidence {attr}>\n{text}\n</evidence>"


def extract_markers(text: str) -> list[int]:
    return list(
        dict.fromkeys(int(n) for grp in _MARKER_RE.findall(text) for n in re.split(r"\s*,\s*", grp))
    )


def build_citations(
    text: str, store: EvidenceStore, snippet_chars: int = 300
) -> tuple[str, list[Citation]]:
    """Map ``[n]`` markers to :class:`Citation` objects; markers that do not refer to
    gathered evidence are removed from the text (never cite what was not seen)."""
    valid: dict[int, Citation] = {}
    invalid: set[int] = set()
    for n in extract_markers(text):
        chunk = store.get(n)
        if chunk is None:
            invalid.add(n)
            continue
        valid[n] = Citation(
            index=n,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            source_uri=chunk.source_uri,
            page=chunk.page,
            heading_path=list(chunk.heading_path),
            snippet=chunk.text[:snippet_chars],
        )
    if invalid:
        logger.warning("dropping citation markers without evidence: %s", sorted(invalid))

        def _fix(m: re.Match[str]) -> str:
            keep = [n for n in re.split(r"\s*,\s*", m.group(1)) if int(n) not in invalid]
            return f"[{', '.join(keep)}]" if keep else ""

        text = re.sub(
            r"\s?" + _MARKER_RE.pattern, lambda m: (" " + _fix(m)) if _fix(m) else "", text
        )
    return text, [valid[n] for n in sorted(valid)]


def looks_like_abstention(text: str) -> bool:
    return not text.strip() or bool(_ABSTAIN_RE.search(text))
