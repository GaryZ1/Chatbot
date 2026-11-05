"""Reranking (docs/08-reranking.md)."""

from chatbot.rerank.base import BaseReranker, heading_line, rerank_text, smart_truncate
from chatbot.rerank.boosts import BusinessPriorReranker
from chatbot.rerank.calibration import GateDecision, ThresholdFit, ThresholdGate, fit_threshold
from chatbot.rerank.cascade import CascadeReranker
from chatbot.rerank.cross_encoder import CrossEncoderReranker
from chatbot.rerank.lexical import LexicalOverlapReranker
from chatbot.rerank.llm_rerank import LLMListwiseReranker, LLMPointwiseReranker, repair_permutation
from chatbot.rerank.postprocess import (
    ContextOrder,
    DocumentGroup,
    dedup_near_identical,
    document_order,
    group_by_document,
    jaccard,
    lost_in_the_middle_order,
    postprocess_context,
)

__all__ = [
    "BaseReranker",
    "BusinessPriorReranker",
    "CascadeReranker",
    "ContextOrder",
    "CrossEncoderReranker",
    "DocumentGroup",
    "GateDecision",
    "LLMListwiseReranker",
    "LLMPointwiseReranker",
    "LexicalOverlapReranker",
    "ThresholdFit",
    "ThresholdGate",
    "dedup_near_identical",
    "document_order",
    "fit_threshold",
    "group_by_document",
    "heading_line",
    "jaccard",
    "lost_in_the_middle_order",
    "postprocess_context",
    "repair_permutation",
    "rerank_text",
    "smart_truncate",
]
