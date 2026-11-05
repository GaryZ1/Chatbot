"""Embedding models (docs/04-embeddings.md)."""

from chatbot.embed.base import (
    KNOWN_PREFIXES,
    BaseEmbedder,
    EmbeddingInfo,
    default_prefixes,
    l2_normalize,
)
from chatbot.embed.cache import CacheStats, CachingEmbedder, QueryEmbeddingCache, normalize_query
from chatbot.embed.factory import build_embedder
from chatbot.embed.providers import HashingEmbedder, SentenceTransformerEmbedder, VoyageEmbedder
from chatbot.embed.quantize import (
    BinaryRescoreIndex,
    Int8Quantizer,
    Int8RescoreIndex,
    binary_quantize,
    hamming_distances,
    hamming_search,
    matryoshka_truncate,
    recall_at_k,
    rescore,
    storage_bytes,
    top_k_exact,
)

__all__ = [
    "KNOWN_PREFIXES",
    "BaseEmbedder",
    "BinaryRescoreIndex",
    "CacheStats",
    "CachingEmbedder",
    "EmbeddingInfo",
    "HashingEmbedder",
    "Int8Quantizer",
    "Int8RescoreIndex",
    "QueryEmbeddingCache",
    "SentenceTransformerEmbedder",
    "VoyageEmbedder",
    "binary_quantize",
    "build_embedder",
    "default_prefixes",
    "hamming_distances",
    "hamming_search",
    "l2_normalize",
    "matryoshka_truncate",
    "normalize_query",
    "recall_at_k",
    "rescore",
    "storage_bytes",
    "top_k_exact",
]
