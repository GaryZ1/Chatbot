"""Tokenizers used for chunk sizing and context budgets.

Chunk sizes must be measured with the *embedding model's* tokenizer
(docs/02-chunking.md §2.2). :class:`RegexTokenizer` is a dependency-free default
that over-approximates sub-word tokenizers slightly (it is conservative, so chunks
never exceed the real limit). :class:`HFTokenizer` wraps a Hugging Face tokenizer
when exact counts are required.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Character spans (start, end) of each token, in order."""
        ...


class RegexTokenizer:
    """Word-piece approximation: words longer than ``piece_len`` count as several tokens."""

    name = "regex-approx"

    def __init__(self, piece_len: int = 6) -> None:
        self.piece_len = piece_len

    def spans(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for m in _TOKEN_RE.finditer(text):
            s, e = m.span()
            if e - s <= self.piece_len:
                out.append((s, e))
                continue
            for i in range(s, e, self.piece_len):
                out.append((i, min(i + self.piece_len, e)))
        return out

    def count(self, text: str) -> int:
        return len(self.spans(text))


class HFTokenizer:
    """Exact token counts with a Hugging Face tokenizer (``pip install transformers``)."""

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # optional dependency

        self.name = model_name
        self._tok = AutoTokenizer.from_pretrained(model_name)

    def spans(self, text: str) -> list[tuple[int, int]]:
        enc = self._tok(text, return_offsets_mapping=True, add_special_tokens=False)
        return [(int(s), int(e)) for s, e in enc["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def default_tokenizer() -> Tokenizer:
    return RegexTokenizer()


def truncate_to_tokens(text: str, max_tokens: int, tokenizer: Tokenizer | None = None) -> str:
    tok = tokenizer or default_tokenizer()
    spans = tok.spans(text)
    if len(spans) <= max_tokens:
        return text
    return text[: spans[max_tokens - 1][1]]
