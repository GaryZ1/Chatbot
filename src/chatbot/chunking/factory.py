"""Chunker registry, factory and the recommended default recipe (docs/02-chunking.md §4–§5).

``build_chunker(name, **params)`` makes every method swappable from config so the
evaluation harness can run the chunking grid (size × overlap × method × ±context).
Chunkers that need models take them as params (``embedder=``, ``llm=``,
``token_embedder=``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chatbot.chunking.base import BaseChunker
from chatbot.chunking.code_ast import CodeChunker
from chatbot.chunking.fixed import FixedCharChunker, FixedTokenChunker
from chatbot.chunking.late import LateChunker
from chatbot.chunking.parent_child import ParentChildChunker, SentenceWindowChunker
from chatbot.chunking.proposition import PropositionChunker
from chatbot.chunking.raptor import RaptorChunker
from chatbot.chunking.recursive import RecursiveChunker
from chatbot.chunking.semantic import SemanticChunker
from chatbot.chunking.sentence import (
    PageChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowChunker,
)
from chatbot.chunking.structure import StructureChunker
from chatbot.chunking.table import TableChunker
from chatbot.core.tokens import Tokenizer

ChunkerFactory = Callable[..., BaseChunker]

CHUNKER_REGISTRY: dict[str, ChunkerFactory] = {
    "fixed_char": FixedCharChunker,
    "fixed_token": FixedTokenChunker,
    "sentence": SentenceChunker,
    "paragraph": ParagraphChunker,
    "sliding_window": SlidingWindowChunker,
    "page": PageChunker,
    "recursive": RecursiveChunker,
    "structure": StructureChunker,
    "semantic": SemanticChunker,
    "parent_child": ParentChildChunker,
    "sentence_window": SentenceWindowChunker,
    "proposition": PropositionChunker,
    "late": LateChunker,
    "code": CodeChunker,
    "table": TableChunker,
    "raptor": RaptorChunker,
}

_ALIASES = {
    "fixed": "fixed_token",
    "token": "fixed_token",
    "char": "fixed_char",
    "structure_aware": "structure",
    "markdown": "structure",
    "small_to_big": "parent_child",
    "code_ast": "code",
    "ast": "code",
    "default": "default",
}


def register_chunker(name: str, factory: ChunkerFactory, *, replace: bool = False) -> None:
    """Register a custom chunker factory under ``name``."""
    key = name.strip().lower()
    if not key:
        raise ValueError("chunker name must be non-empty")
    if key in CHUNKER_REGISTRY and not replace:
        raise ValueError(f"chunker {key!r} already registered (pass replace=True)")
    CHUNKER_REGISTRY[key] = factory


def available_chunkers() -> list[str]:
    return sorted([*CHUNKER_REGISTRY, "default"])


def build_chunker(name: str, **params: Any) -> BaseChunker:
    """Instantiate a chunker by name, e.g. ``build_chunker("recursive", max_tokens=256)``."""
    key = _ALIASES.get(name.strip().lower(), name.strip().lower())
    if key == "default":
        return default_chunker(**params)
    try:
        factory = CHUNKER_REGISTRY[key]
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; available: {available_chunkers()}") from None
    try:
        return factory(**params)
    except TypeError as exc:
        raise ValueError(f"invalid parameters for chunker {key!r}: {exc}") from exc


def default_chunker(
    max_tokens: int = 512,
    parent_max_tokens: int = 1500,
    min_tokens: int = 50,
    tokenizer: Tokenizer | None = None,
) -> ParentChildChunker:
    """The 2026 "boringly excellent" default (docs/02 §4).

    Structure-aware children (≤ 512 tokens; tables/code never split; pieces < 50
    tokens merged), ``"Title > H1 > H2"`` breadcrumbs in ``embed_text``, and parent
    sections of up to ~1,500 tokens returned for generation. Contextual retrieval
    (``chatbot.enrich.contextual``) is applied afterwards as enrichment.
    """
    return ParentChildChunker(
        child_max_tokens=max_tokens,
        parent_max_tokens=parent_max_tokens,
        min_tokens=min_tokens,
        child_min_tokens=min_tokens,
        tokenizer=tokenizer,
        breadcrumbs=True,
    )
