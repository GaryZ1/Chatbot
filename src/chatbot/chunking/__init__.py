"""Chunking: every method from docs/02-chunking.md behind one interface.

Start with :func:`default_chunker` (structure-aware + breadcrumbs + parent-child)
and compare alternatives with :func:`build_chunker` in the evaluation harness.
"""

from chatbot.chunking.base import (
    ATOMIC_TYPES,
    AsyncChunker,
    BaseChunker,
    BlockSpan,
    DocIndex,
    Piece,
    achunk_any,
    block_spans,
    breadcrumb,
    doc_metadata,
)
from chatbot.chunking.code_ast import CodeChunker, python_definitions, regex_definitions
from chatbot.chunking.factory import (
    CHUNKER_REGISTRY,
    available_chunkers,
    build_chunker,
    default_chunker,
    register_chunker,
)
from chatbot.chunking.fixed import FixedCharChunker, FixedTokenChunker
from chatbot.chunking.late import LateChunker, LateChunkingResult, TokenEmbedder, pool_spans
from chatbot.chunking.parent_child import ParentChildChunker, SentenceWindowChunker
from chatbot.chunking.proposition import PropositionChunker, Propositions
from chatbot.chunking.raptor import RaptorBuilder, RaptorChunker, kmeans
from chatbot.chunking.recursive import DEFAULT_SEPARATORS, RecursiveChunker, RecursiveSplitter
from chatbot.chunking.semantic import SemanticChunker, cosine_distances
from chatbot.chunking.sentence import (
    PageChunker,
    ParagraphChunker,
    SentenceChunker,
    SlidingWindowChunker,
    split_sentences,
)
from chatbot.chunking.structure import StructureChunker
from chatbot.chunking.table import TableChunker, describe_table, split_table_lines

__all__ = [
    "ATOMIC_TYPES",
    "CHUNKER_REGISTRY",
    "DEFAULT_SEPARATORS",
    "AsyncChunker",
    "BaseChunker",
    "BlockSpan",
    "CodeChunker",
    "DocIndex",
    "FixedCharChunker",
    "FixedTokenChunker",
    "LateChunker",
    "LateChunkingResult",
    "PageChunker",
    "ParagraphChunker",
    "ParentChildChunker",
    "Piece",
    "PropositionChunker",
    "Propositions",
    "RaptorBuilder",
    "RaptorChunker",
    "RecursiveChunker",
    "RecursiveSplitter",
    "SemanticChunker",
    "SentenceChunker",
    "SentenceWindowChunker",
    "SlidingWindowChunker",
    "StructureChunker",
    "TableChunker",
    "TokenEmbedder",
    "achunk_any",
    "available_chunkers",
    "block_spans",
    "breadcrumb",
    "build_chunker",
    "cosine_distances",
    "default_chunker",
    "describe_table",
    "doc_metadata",
    "kmeans",
    "pool_spans",
    "python_definitions",
    "regex_definitions",
    "register_chunker",
    "split_sentences",
    "split_table_lines",
]
