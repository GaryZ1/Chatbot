"""Advanced / agentic RAG (docs/09-advanced-rag.md): hierarchical tools, a manual
tool-use loop, CRAG corrective loop, HippoRAG-style graph retrieval, deep research."""

from chatbot.agent.crag import CorrectiveRAG, CRAGConfig, CRAGResult, GenerateFn
from chatbot.agent.deep_research import DeepResearchConfig, DeepResearcher, ResearchResult
from chatbot.agent.evidence import EvidenceStore, build_citations, format_evidence
from chatbot.agent.graph import (
    EntityExtractor,
    GraphRetriever,
    HeuristicEntityExtractor,
    LLMEntityExtractor,
    personalized_pagerank,
)
from chatbot.agent.loop import AgentBudget, AgenticRAG, AgentResult, AgentStep, StopReason
from chatbot.agent.tools import (
    AgentTool,
    DocStoreCatalog,
    ListSourcesTool,
    ReadTool,
    SearchTool,
    SourceCatalog,
    SourceInfo,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    default_tools,
)

__all__ = [
    "AgentBudget",
    "AgentResult",
    "AgentStep",
    "AgentTool",
    "AgenticRAG",
    "CRAGConfig",
    "CRAGResult",
    "CorrectiveRAG",
    "DeepResearchConfig",
    "DeepResearcher",
    "DocStoreCatalog",
    "EntityExtractor",
    "EvidenceStore",
    "GenerateFn",
    "GraphRetriever",
    "HeuristicEntityExtractor",
    "LLMEntityExtractor",
    "ListSourcesTool",
    "ReadTool",
    "ResearchResult",
    "SearchTool",
    "SourceCatalog",
    "SourceInfo",
    "StopReason",
    "ToolContext",
    "ToolError",
    "ToolPolicy",
    "ToolRegistry",
    "ToolResult",
    "build_citations",
    "default_tools",
    "format_evidence",
    "personalized_pagerank",
]
