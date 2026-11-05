"""Assemble every pipeline component from :class:`~chatbot.config.Settings`.

This is the only place that knows which concrete class implements each stage, so
swapping a component (embedder, store, reranker, LLM) is a config change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chatbot.chunking import build_chunker
from chatbot.config import Settings
from chatbot.core.interfaces import Chunker, DocStore, Embedder, Reranker, Retriever, VectorStore
from chatbot.core.types import ChunkKind
from chatbot.embed import CachingEmbedder, QueryEmbeddingCache, build_embedder
from chatbot.enrich import Enricher, Glossary, default_enricher
from chatbot.generate import AnswerGenerator, ContextBuilder, OrderStrategy, PromptConfig
from chatbot.generate.citations import CitationVerifier
from chatbot.guard.injection import InjectionDetector
from chatbot.guard.input import InputGuard
from chatbot.guard.output import OutputGuard
from chatbot.guard.poisoning import PoisoningScanner
from chatbot.guard.policies import GuardPolicies
from chatbot.index import BM25Index, IndexWriter, InMemoryVectorStore
from chatbot.index.docstore import CompositeDocStore, InMemoryDocStore
from chatbot.index.schema import IndexSchema
from chatbot.ingest import Deduplicator, default_registry
from chatbot.ingest.parse import ParserRegistry
from chatbot.llm import LLM, FakeLLM
from chatbot.llm.fake import extractive_handler
from chatbot.memory import ConversationMemory, InMemorySessionStore, LongTermMemory
from chatbot.obs.metrics import MetricsRegistry, PriceTable
from chatbot.obs.trace_store import InMemoryTraceStore, JsonlTraceStore, TraceStore
from chatbot.obs.tracing import Tracer
from chatbot.ops.feedback import InMemoryFeedbackStore, JsonlFeedbackStore
from chatbot.ops.ratelimit import DailyBudget, RateLimiter, UserLimits
from chatbot.ops.semantic_cache import SemanticCache
from chatbot.query import QueryUnderstanding, RouterConfig
from chatbot.rerank import (
    CascadeReranker,
    CrossEncoderReranker,
    LexicalOverlapReranker,
    LLMPointwiseReranker,
    ThresholdGate,
)
from chatbot.retrieve import (
    DenseRetriever,
    FusionConfig,
    HybridRetriever,
    ParentMappingRetriever,
    SparseRetriever,
)

log = logging.getLogger(__name__)


def build_llm(settings: Settings) -> LLM:
    cfg = settings.llm
    if cfg.provider == "fake":
        return FakeLLM(handler=extractive_handler)
    from chatbot.llm.anthropic_llm import AnthropicLLM
    from chatbot.llm.base import Tier

    return AnthropicLLM(
        models={Tier(k): v for k, v in cfg.models.items()},
        effort={Tier(k): v for k, v in cfg.effort.items()},
        timeout_s=cfg.timeout_s,
        max_retries=cfg.max_retries,
        enable_fallbacks=cfg.enable_fallbacks,
    )


@dataclass
class Components:
    settings: Settings
    llm: LLM
    embedder: Embedder
    vector_store: VectorStore
    bm25: BM25Index
    parents: InMemoryDocStore
    docstore: DocStore
    retriever: Retriever
    reranker: Reranker | None
    gate: ThresholdGate | None
    writer: IndexWriter
    parser: ParserRegistry
    chunker: Chunker
    enricher: Enricher
    dedup: Deduplicator
    poisoning: PoisoningScanner
    understanding: QueryUnderstanding
    generator: AnswerGenerator
    memory: ConversationMemory
    long_term: LongTermMemory | None
    input_guard: InputGuard | None
    output_guard: OutputGuard | None
    chunk_scanner: InjectionDetector | None
    cache: SemanticCache | None
    limits: UserLimits
    tracer: Tracer
    trace_store: TraceStore
    feedback: InMemoryFeedbackStore | JsonlFeedbackStore
    metrics: MetricsRegistry
    prices: PriceTable
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def index_version(self) -> str:
        return f"{self.settings.index.collection}_v{self.settings.index.version}"

    # ------------------------------------------------------------------ persistence

    def save(self) -> None:
        """Persist local indexes (in-memory vector store, BM25, parents)."""
        root = Path(self.settings.index.path)
        root.mkdir(parents=True, exist_ok=True)
        if isinstance(self.vector_store, InMemoryVectorStore):
            self.vector_store.save(root / "vectors")
        self.bm25.save(root / "bm25")
        self.parents.save(root / "parents.jsonl")


def _load_or_new_vector_store(settings: Settings, dim: int) -> VectorStore:
    idx = settings.index
    if idx.backend == "qdrant":
        from chatbot.index.qdrant_store import QdrantVectorStore

        schema = IndexSchema(base_name=idx.collection, version=idx.version)
        store = QdrantVectorStore(schema, dim=dim, url=idx.qdrant_url)
        return store
    path = Path(idx.path) / "vectors"
    if path.exists():
        loaded = InMemoryVectorStore.load(path)
        if loaded.dim == dim:
            return loaded
        log.warning(
            "index at %s has dim %d != embedder dim %d; starting empty", path, loaded.dim, dim
        )
    return InMemoryVectorStore(dim)


def _reranker(settings: Settings, llm: LLM) -> Reranker | None:
    cfg = settings.rerank
    match cfg.method:
        case "none":
            return None
        case "lexical":
            return LexicalOverlapReranker()
        case "cross_encoder":
            return CrossEncoderReranker(cfg.model)
        case "llm":
            return LLMPointwiseReranker(llm)
        case "cascade":
            return CascadeReranker(
                LexicalOverlapReranker(),
                LLMPointwiseReranker(llm),
                expensive_top_m=cfg.cascade_top_m,
            )
    raise ValueError(f"unknown reranker {cfg.method}")


def build_components(settings: Settings, *, llm: LLM | None = None) -> Components:
    llm = llm or build_llm(settings)

    # ---------------------------------------------------------------- indexing plane
    base_embedder = build_embedder(settings.embedding.name, **settings.embedding.params)
    embedder: Embedder = CachingEmbedder(
        base_embedder, QueryEmbeddingCache(maxsize=settings.embedding.cache_size)
    )
    vector_store = _load_or_new_vector_store(settings, embedder.dim)
    root = Path(settings.index.path)
    bm25 = BM25Index.load(root / "bm25") if (root / "bm25").exists() else BM25Index()
    parents = InMemoryDocStore.load(root / "parents.jsonl")
    docstore = CompositeDocStore([parents, vector_store])
    writer = IndexWriter(embedder, vector_store, bm25)

    glossary = (
        Glossary.from_mapping(settings.enrichment.glossary)
        if settings.enrichment.glossary
        else None
    )
    enr = settings.enrichment
    needs_llm = enr.contextual or enr.questions or enr.doc_summary or enr.llm_metadata
    enricher = default_enricher(
        llm if needs_llm else None,
        glossary=glossary,
        contextual=enr.contextual,
        questions=enr.questions,
        doc_summary=enr.doc_summary,
        llm_metadata=enr.llm_metadata,
    )
    chunker = build_chunker(settings.chunking.method, **settings.chunking.params)
    policies = GuardPolicies.from_dict(settings.guard.policies)
    policies.output.allowed_domains = list(settings.guard.allowed_domains)

    # ---------------------------------------------------------------- serving plane
    r = settings.retrieval
    legs: list[Retriever] = [DenseRetriever(embedder, vector_store), SparseRetriever(bm25)]
    retriever: Retriever = HybridRetriever(
        legs,
        fusion=FusionConfig(method=r.fusion, rrf_k=r.rrf_k, weights=r.weights),
        leg_k=r.leg_k,
    )
    if r.question_leg:
        # Hypothetical-question hits resolve to the chunk they were generated from.
        retriever = ParentMappingRetriever(
            retriever, vector_store, kinds=frozenset({ChunkKind.QUESTION})
        )
    reranker = _reranker(settings, llm)
    gate = (
        # Gate on the reranker's own score (stage name == reranker name), not fused RRF.
        ThresholdGate(settings.rerank.abstain_threshold, stage=reranker.name)
        if settings.rerank.abstain_threshold is not None and reranker is not None
        else None
    )

    g = settings.generation
    verifier = CitationVerifier() if g.verify_citations else None
    generator = AnswerGenerator(
        llm,
        context_builder=ContextBuilder(
            max_tokens=g.max_context_tokens, max_sources=g.max_sources, order=OrderStrategy(g.order)
        ),
        prompt_config=PromptConfig(
            assistant_name=g.assistant_name, organization=g.organization, audience=g.audience
        ),
        verifier=verifier,
    )
    understanding = QueryUnderstanding(
        llm,
        normalizer=None,
        router_config=RouterConfig(agent_enabled=settings.agent.enabled, sql_enabled=False),
    )

    m = settings.memory
    if m.backend == "redis":
        from chatbot.memory import RedisSessionStore

        session_store: Any = RedisSessionStore(m.redis_url, ttl_s=m.session_ttl_s)
    else:
        session_store = InMemorySessionStore(ttl_s=m.session_ttl_s)
    memory = ConversationMemory(llm, session_store, window_turns=m.window_turns)
    long_term = None
    if m.long_term:
        from chatbot.memory import InMemoryMemoryBackend, JSONFileMemoryBackend

        backend: Any = (
            JSONFileMemoryBackend(m.long_term_path) if m.long_term_path else InMemoryMemoryBackend()
        )
        long_term = LongTermMemory(llm, backend)

    o = settings.ops
    limits = UserLimits(
        RateLimiter(rate=o.rate_per_s, burst=o.burst),
        DailyBudget(max_tokens_per_day=o.daily_tokens, max_usd_per_day=o.daily_usd),
    )
    guard_llm = llm if settings.guard.llm_judge else None
    input_guard = (
        InputGuard(policies.input, rate_limit=limits, llm=guard_llm)
        if settings.guard.enabled
        else None
    )
    output_guard = OutputGuard(policies.output, llm=guard_llm) if settings.guard.enabled else None
    chunk_scanner = (
        InjectionDetector(policies.input.injection)
        if settings.guard.enabled and settings.guard.scan_retrieved_chunks
        else None
    )
    cache = (
        SemanticCache(embedder, threshold=o.cache_threshold, ttl_s=o.cache_ttl_s)
        if o.semantic_cache
        else None
    )

    prices = PriceTable()
    ob = settings.obs
    trace_store: TraceStore = (
        JsonlTraceStore(ob.trace_path) if ob.trace_path else InMemoryTraceStore()
    )
    tracer = Tracer(
        trace_store,
        redact=ob.redact,
        pipeline_version=settings.pipeline_version,
        sample_rate=ob.sample_rate,
        prices=prices,
    )
    feedback: InMemoryFeedbackStore | JsonlFeedbackStore = (
        JsonlFeedbackStore(ob.feedback_path) if ob.feedback_path else InMemoryFeedbackStore()
    )

    return Components(
        settings=settings,
        llm=llm,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        parents=parents,
        docstore=docstore,
        retriever=retriever,
        reranker=reranker,
        gate=gate,
        writer=writer,
        parser=default_registry(use_docling=settings.ingest.use_docling),
        chunker=chunker,
        enricher=enricher,
        dedup=Deduplicator(near_threshold=settings.ingest.near_dup_threshold),
        poisoning=PoisoningScanner(policies.poisoning),
        understanding=understanding,
        generator=generator,
        memory=memory,
        long_term=long_term,
        input_guard=input_guard,
        output_guard=output_guard,
        chunk_scanner=chunk_scanner,
        cache=cache,
        limits=limits,
        tracer=tracer,
        trace_store=trace_store,
        feedback=feedback,
        metrics=MetricsRegistry(),
        prices=prices,
    )
