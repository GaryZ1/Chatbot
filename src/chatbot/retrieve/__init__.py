"""Retrieval (docs/07-retrieval.md)."""

from chatbot.retrieve.cutoff import score_gap_cutoff, token_budget_cutoff
from chatbot.retrieve.dense import DenseRetriever
from chatbot.retrieve.expand import (
    apply_boosts,
    authority_score,
    expand_neighbours,
    freshness_score,
    resolve_parents,
)
from chatbot.retrieve.fusion import (
    FusionConfig,
    FusionMethod,
    Normalization,
    convex_combination,
    dedup,
    fuse,
    fuse_many,
    normalize_scores,
    rrf,
)
from chatbot.retrieve.hybrid import HybridRetriever, ParentMappingRetriever
from chatbot.retrieve.mmr import MMRSelector, mmr_indices
from chatbot.retrieve.sparse import SparseRetriever

__all__ = [
    "DenseRetriever",
    "FusionConfig",
    "FusionMethod",
    "HybridRetriever",
    "MMRSelector",
    "Normalization",
    "ParentMappingRetriever",
    "SparseRetriever",
    "apply_boosts",
    "authority_score",
    "convex_combination",
    "dedup",
    "expand_neighbours",
    "freshness_score",
    "fuse",
    "fuse_many",
    "mmr_indices",
    "normalize_scores",
    "resolve_parents",
    "rrf",
    "score_gap_cutoff",
    "token_budget_cutoff",
]
