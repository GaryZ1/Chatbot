"""Run every method's demo (offline by default; RAG_LLM=claude for real Claude calls).

    python examples/advanced_rag/run_all.py            # all methods
    python examples/advanced_rag/run_all.py hyde crag  # a subset
"""

import importlib
import sys
import time

METHODS = [
    "naive_rag", "hybrid_rrf", "rerank", "colbert_maxsim",
    "contextual_retrieval", "parent_child", "late_chunking", "propositions", "raptor",
    "doc2query",
    "query_rewrite", "hyde", "rag_fusion", "step_back", "decomposition", "self_query",
    "adaptive_rag",
    "mmr_reorder", "compression", "chain_of_note",
    "self_ask", "ircot", "iter_retgen", "flare",
    "self_rag", "crag", "cove", "sufficient_context", "robust_rag",
    "replug", "speculative_rag", "raft",
    "graphrag", "lightrag", "hipporag", "lazy_graphrag", "think_on_graph",
    "agentic_react", "a_rag", "deep_research", "search_r1",
    "cag", "self_route", "memorag",
    "colpali", "text_to_sql",
]


def main(names: list[str]) -> int:
    failed = []
    for name in names or METHODS:
        print(f"\n################ {name}")
        t0 = time.perf_counter()
        try:
            importlib.import_module(name).demo()
        except Exception as exc:  # keep going; report at the end
            failed.append(name)
            print(f"FAILED: {exc!r}")
        print(f"({time.perf_counter() - t0:.2f}s)")
    print(f"\n{len(names or METHODS) - len(failed)} ok, {len(failed)} failed {failed or ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
