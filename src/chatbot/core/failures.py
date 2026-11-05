"""Failure taxonomy from docs/00-architecture-overview.md §3.

Every bad answer gets an F-code. :func:`diagnose` assigns the retrieval-funnel
codes automatically from a trace (docs/12-evaluation.md §7); generation-side
codes (F8-F11) need a judge or a human.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum


class FailureCode(StrEnum):
    F1_MISSING_CONTENT = "F1"
    F2_PARSE_LOSS = "F2"
    F3_BAD_CHUNK_BOUNDARIES = "F3"
    F4_CONTEXTLESS_CHUNKS = "F4"
    F5_MISSED_TOP_K = "F5"
    F6_VOCABULARY_MISMATCH = "F6"
    F7_NOT_IN_CONTEXT = "F7"
    F8_NOT_EXTRACTED = "F8"
    F9_WRONG_FORMAT = "F9"
    F10_INCORRECT_SPECIFICITY = "F10"
    F11_INCOMPLETE = "F11"
    F12_STALE_CONFLICTING = "F12"
    F13_LEAKAGE = "F13"
    F14_INJECTION = "F14"
    F15_CONVERSATION_DRIFT = "F15"


FAILURE_STAGE: dict[FailureCode, str] = {
    FailureCode.F1_MISSING_CONTENT: "ingestion",
    FailureCode.F2_PARSE_LOSS: "parsing",
    FailureCode.F3_BAD_CHUNK_BOUNDARIES: "chunking",
    FailureCode.F4_CONTEXTLESS_CHUNKS: "enrichment",
    FailureCode.F5_MISSED_TOP_K: "retrieval",
    FailureCode.F6_VOCABULARY_MISMATCH: "retrieval",
    FailureCode.F7_NOT_IN_CONTEXT: "assembly",
    FailureCode.F8_NOT_EXTRACTED: "generation",
    FailureCode.F9_WRONG_FORMAT: "generation",
    FailureCode.F10_INCORRECT_SPECIFICITY: "generation",
    FailureCode.F11_INCOMPLETE: "generation",
    FailureCode.F12_STALE_CONFLICTING: "lifecycle",
    FailureCode.F13_LEAKAGE: "security",
    FailureCode.F14_INJECTION: "security",
    FailureCode.F15_CONVERSATION_DRIFT: "multi-turn",
}


def diagnose(
    gold_doc_ids: Sequence[str],
    candidate_doc_ids: Sequence[str],
    final_k_doc_ids: Sequence[str],
    context_doc_ids: Sequence[str],
    corpus_doc_ids: set[str] | None = None,
) -> FailureCode | None:
    """Locate where gold evidence was lost in the retrieval funnel.

    Returns ``None`` when gold evidence reached the context window (the failure, if
    any, is generation-side and needs a judge).
    """
    gold = set(gold_doc_ids)
    if not gold:
        return None
    if corpus_doc_ids is not None and not gold & corpus_doc_ids:
        return FailureCode.F1_MISSING_CONTENT
    if not gold & set(candidate_doc_ids):
        return FailureCode.F5_MISSED_TOP_K  # parse/chunk/embed/coverage problem upstream
    if not gold & set(final_k_doc_ids):
        return FailureCode.F5_MISSED_TOP_K  # rerank / fusion problem
    if not gold & set(context_doc_ids):
        return FailureCode.F7_NOT_IN_CONTEXT
    return None
