"""Config-driven embedder construction (docs/04-embeddings.md §6 bake-offs swap embedders)."""

from __future__ import annotations

from typing import Any

from chatbot.embed.base import BaseEmbedder
from chatbot.embed.providers.hashing import HashingEmbedder
from chatbot.embed.providers.sentence_transformers import SentenceTransformerEmbedder
from chatbot.embed.providers.voyage import VoyageEmbedder

_ALIASES: dict[str, type[BaseEmbedder]] = {
    "hashing": HashingEmbedder,
    "hash": HashingEmbedder,
    "sentence-transformers": SentenceTransformerEmbedder,
    "sentence_transformers": SentenceTransformerEmbedder,
    "st": SentenceTransformerEmbedder,
    "voyage": VoyageEmbedder,
}


def build_embedder(name: str = "hashing", **params: Any) -> BaseEmbedder:
    """Build an embedder by provider name.

    ``name`` is a provider (``hashing``, ``sentence-transformers``/``st``, ``voyage``)
    or ``provider:model`` shorthand, e.g. ``"st:intfloat/e5-large-v2"`` or
    ``"voyage:voyage-3-large"``. Remaining ``params`` go to the constructor; wrap the
    result in :class:`chatbot.embed.cache.CachingEmbedder` for query caching.
    """
    provider, _, model = name.partition(":")
    cls = _ALIASES.get(provider.strip().lower())
    if cls is None:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(_ALIASES)}")
    if model:
        if cls is SentenceTransformerEmbedder:
            params.setdefault("model_name", model)
        elif cls is VoyageEmbedder:
            params.setdefault("model", model)
        else:
            raise ValueError(f"provider {provider!r} does not take a model name")
    return cls(**params)
