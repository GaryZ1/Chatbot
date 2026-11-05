"""Enrichment: breadcrumbs, contextual retrieval, questions, metadata, summaries and
glossary expansion (docs/03-enrichment.md), composed by a versioned :class:`Enricher`."""

from __future__ import annotations

from chatbot.enrich.base import (
    BreadcrumbStep,
    Enricher,
    EnrichmentStep,
    chunk_breadcrumb,
    compose_embed_text,
)
from chatbot.enrich.contextual import CONTEXTUAL_SYSTEM, CONTEXTUAL_USER, ContextualEnricher
from chatbot.enrich.glossary import Glossary, GlossaryEnricher, mine_acronyms
from chatbot.enrich.metadata import (
    DocMetadata,
    Entity,
    LLMMetadataExtractor,
    MetadataEnricher,
    RuleBasedExtractor,
    extract_dates,
    extract_emails,
    extract_identifiers,
)
from chatbot.enrich.questions import QuestionGenerator, QuestionsOut
from chatbot.enrich.summaries import ChunkSummarizer, DocumentSummarizer
from chatbot.llm.base import LLM


def default_enricher(
    llm: LLM | None,
    *,
    glossary: Glossary | None = None,
    contextual: bool = True,
    questions: bool = False,
    doc_summary: bool = False,
    llm_metadata: bool = False,
) -> Enricher:
    """Recommended order (docs/03 §3): metadata → breadcrumbs → contextual → glossary →
    questions → document summary. Without an LLM only the free steps run."""
    steps: list[EnrichmentStep] = [
        MetadataEnricher(llm if llm_metadata else None),
        BreadcrumbStep(),
    ]
    if llm is not None and contextual:
        steps.append(ContextualEnricher(llm))
    if glossary is not None:
        steps.append(GlossaryEnricher(glossary))
    if llm is not None and questions:
        steps.append(QuestionGenerator(llm))
    if llm is not None and doc_summary:
        steps.append(DocumentSummarizer(llm))
    return Enricher(steps)


__all__ = [
    "CONTEXTUAL_SYSTEM",
    "CONTEXTUAL_USER",
    "BreadcrumbStep",
    "ChunkSummarizer",
    "ContextualEnricher",
    "DocMetadata",
    "DocumentSummarizer",
    "Enricher",
    "EnrichmentStep",
    "Entity",
    "Glossary",
    "GlossaryEnricher",
    "LLMMetadataExtractor",
    "MetadataEnricher",
    "QuestionGenerator",
    "QuestionsOut",
    "RuleBasedExtractor",
    "chunk_breadcrumb",
    "compose_embed_text",
    "default_enricher",
    "extract_dates",
    "extract_emails",
    "extract_identifiers",
    "mine_acronyms",
]
