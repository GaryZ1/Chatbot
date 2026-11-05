"""Pipeline configuration (docs/16-build-plan.md: config-driven pipelines, every
component behind an interface).

Configuration is loaded from YAML and can be overridden with environment variables
of the form ``CHATBOT__<SECTION>__<KEY>=value`` (values are parsed as YAML, so
``CHATBOT__RETRIEVAL__FINAL_K=8`` yields an int).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Section):
    provider: Literal["anthropic", "fake"] = "anthropic"
    models: dict[str, str] = Field(
        default_factory=lambda: {
            "small": "claude-haiku-5-5",
            "mid": "claude-opus-5-5",
            "frontier": "claude-opus-5-5",
        }
    )
    effort: dict[str, str] = Field(
        default_factory=lambda: {"small": "low", "mid": "medium", "frontier": "high"}
    )
    timeout_s: float = 120.0
    max_retries: int = 3
    enable_fallbacks: bool = True


class IngestConfig(_Section):
    include: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.txt", "**/*.html", "**/*.pdf"]
    )
    exclude: list[str] = Field(default_factory=list)
    default_acl_groups: list[str] = Field(default_factory=list)
    default_public: bool = False
    tenant_id: str = "default"
    use_docling: bool = False
    near_dup_threshold: float = 0.9
    scan_poisoning: bool = True
    concurrency: int = 4


class ChunkingConfig(_Section):
    method: str = "default"
    params: dict[str, Any] = Field(default_factory=dict)


class EnrichmentConfig(_Section):
    contextual: bool = False  # needs an LLM call per chunk at ingest
    questions: bool = False
    doc_summary: bool = False
    llm_metadata: bool = False
    glossary: dict[str, list[str] | str] = Field(default_factory=dict)


class EmbeddingConfig(_Section):
    name: str = "hashing"
    params: dict[str, Any] = Field(default_factory=dict)
    cache_size: int = 10_000


class IndexConfig(_Section):
    backend: Literal["memory", "qdrant"] = "memory"
    path: str = "data/index"
    qdrant_url: str | None = None
    collection: str = "chunks"
    version: int = 1


class RetrievalConfig(_Section):
    leg_k: int = 100
    fusion: Literal["rrf", "weighted_rrf", "convex"] = "rrf"
    rrf_k: int = 60
    weights: dict[str, float] = Field(default_factory=dict)
    question_leg: bool = True
    candidates: int = 50  # passed to the reranker
    final_k: int = 8
    multi_query: int = 0  # extra paraphrases; 0 disables
    hyde: bool = False
    neighbour_window: int = 0
    parent_expansion: bool = True
    mmr_lambda: float | None = None
    freshness_weight: float = 0.0
    only_latest: bool = True


class RerankConfig(_Section):
    method: Literal["none", "lexical", "cross_encoder", "llm", "cascade"] = "lexical"
    model: str = "BAAI/bge-reranker-v2-m3"
    abstain_threshold: float | None = 0.05
    cascade_top_m: int = 20


class GenerationConfig(_Section):
    max_context_tokens: int = 8000
    max_sources: int = 12
    order: Literal["relevance", "doc_grouped", "edges"] = "relevance"
    assistant_name: str = "Assistant"
    organization: str = ""
    audience: str = "employees"
    verify_citations: bool = True


class MemoryConfig(_Section):
    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    window_turns: int = 8
    session_ttl_s: int = 86_400
    long_term: bool = False
    long_term_path: str | None = None


class GuardConfig(_Section):
    enabled: bool = True
    allowed_domains: list[str] = Field(default_factory=list)
    llm_judge: bool = False
    scan_retrieved_chunks: bool = True
    policies: dict[str, Any] = Field(default_factory=dict)  # raw GuardPolicies overrides


class OpsConfig(_Section):
    semantic_cache: bool = True
    cache_threshold: float = 0.95
    cache_ttl_s: int = 3600
    rate_per_s: float = 2.0
    burst: int = 10
    daily_tokens: int | None = 2_000_000
    daily_usd: float | None = 20.0
    request_timeout_s: float = 60.0


class ObsConfig(_Section):
    trace_path: str | None = "data/traces.jsonl"
    feedback_path: str | None = "data/feedback.jsonl"
    redact: bool = True
    sample_rate: float = 1.0


class AgentConfig(_Section):
    enabled: bool = False
    max_steps: int = 8


class Settings(_Section):
    pipeline_version: str = "v1"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    guard: GuardConfig = Field(default_factory=GuardConfig)
    ops: OpsConfig = Field(default_factory=OpsConfig)
    obs: ObsConfig = Field(default_factory=ObsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, *, env: dict[str, str] | None = None) -> Settings:
        data: dict[str, Any] = {}
        path = path or os.environ.get("CHATBOT_CONFIG")
        if path:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        _apply_env(data, os.environ if env is None else env)
        return cls.model_validate(data)


def _apply_env(data: dict[str, Any], env: Any) -> None:
    for key, raw in env.items():
        if not key.startswith("CHATBOT__"):
            continue
        parts = [p.lower() for p in key[len("CHATBOT__") :].split("__") if p]
        if not parts:
            continue
        node = data
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot override non-section {key}")
        node[parts[-1]] = yaml.safe_load(raw)
