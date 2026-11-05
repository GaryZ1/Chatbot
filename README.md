# Chatbot: from zero to the top 0.1% of RAG chatbot engineers

This repo is a **curriculum and build plan** for production-grade, retrieval-augmented (RAG) and agentic chatbots. It covers each stage of the pipeline in depth: what each technique is, why it exists, how it works, its pros and cons, when to use it, how to tune it, and how it fails.

> Research snapshot: **November 2026**. The field moves fast. Every claim cites a source, and benchmark numbers from vendors or blogs are marked *(vendor / unverified)*. Before you pick a model or method, rerun the evaluation in [`docs/12-evaluation.md`](docs/12-evaluation.md) **on your own data**.

---

## What "top 0.1%" means here

Most engineers can wire LangChain to a vector database. The top 0.1% can do the following:

1. **Diagnose failures by stage.** They can tell whether a bad answer came from parsing, chunking, embedding, retrieval, ranking, context assembly, or generation, and they can prove it with metrics.
2. **Reason about trade-offs.** They can explain why recursive 512-token chunks beat semantic chunking on one corpus and lose on another, and what that costs.
3. **Treat evaluation as the product.** They build golden sets, calibrated LLM judges, and regression gates before they tune anything.
4. **Know the frontier and its limits.** They know contextual retrieval, late chunking, ColBERT/ColPali, GraphRAG, agentic and RL-trained search, and context engineering. They also know when *not* to use each one.
5. **Ship safely.** They handle ACL-aware retrieval, prompt-injection defense, PII, and cost and latency budgets, and they monitor drift in production.

---

## Curriculum (read in order)

| # | Module | You will be able to... |
|---|--------|------------------------|
| 00 | [Architecture overview](docs/00-architecture-overview.md) | Draw the whole system and name every failure point |
| 01 | [Ingestion & document parsing](docs/01-ingestion-parsing.md) | Turn PDFs, HTML, slides, tables, and scans into clean, structured text |
| 02 | [Chunking (every method)](docs/02-chunking.md) | Pick, tune, and evaluate any of 15+ chunking strategies |
| 03 | [Enrichment & metadata](docs/03-enrichment.md) | Add contextual headers, summaries, questions, entities, and ACLs |
| 04 | [Embedding models](docs/04-embeddings.md) | Choose, benchmark, fine-tune, quantize, and migrate embeddings |
| 05 | [Indexing & vector stores](docs/05-indexing-vector-stores.md) | Choose HNSW, IVF-PQ, or DiskANN; filtered search; picking a database |
| 06 | [Query understanding & transformation](docs/06-query-understanding.md) | Rewriting, HyDE, multi-query, decomposition, routing |
| 07 | [Retrieval](docs/07-retrieval.md) | Sparse, dense, hybrid, learned-sparse, late-interaction, and fusion |
| 08 | [Reranking](docs/08-reranking.md) | Cross-encoders, LLM rerankers, ColBERT, and their latency budgets |
| 09 | [Advanced RAG: Graph, Agentic, Multimodal](docs/09-advanced-rag.md) | GraphRAG, HippoRAG, Self-RAG/CRAG, agentic and RL search, ColPali |
| 10 | [Generation & context engineering](docs/10-generation-context.md) | Prompt layout, citations, long context vs. RAG, context rot |
| 11 | [Conversation & memory](docs/11-conversation-memory.md) | Multi-turn condensing, memory tiers, personalization |
| 12 | [Evaluation](docs/12-evaluation.md) | Golden sets, retrieval/generation metrics, LLM judges, online A/B tests |
| 13 | [Guardrails & security](docs/13-guardrails-security.md) | Prompt injection, poisoning, ACLs, PII, hallucination guards |
| 14 | [Production & LLMOps](docs/14-production-llmops.md) | Latency, cost, caching, observability, index lifecycle |
| 15 | [Enterprise case studies](docs/15-enterprise-case-studies.md) | What worked at LinkedIn, DoorDash, Merck, and others, and why |
| 16 | [Build plan & roadmap](docs/16-build-plan.md) | A phased, milestone-based plan to build this chatbot in this repo |
| 17 | [Reading list & repos](docs/17-reading-list.md) | The papers and repos worth reading, ranked |

### Suggested pace

- **Weeks 1–2:** Modules 00–05. Build a naive RAG baseline and its evaluation harness.
- **Weeks 3–4:** Modules 06–08. Add hybrid search and reranking, and measure the gain.
- **Weeks 5–6:** Modules 09–11. Add agentic and graph retrieval and memory, and justify each with the evaluation.
- **Weeks 7–8:** Modules 12–15. Add production hardening and do a red-team pass.
- **Ongoing:** Module 16 milestones and Module 17 papers.

Each module ends with **"Expert checks"**: questions you should be able to answer cold. If you can't, re-read the module.

---

## The one-page mental model

```
           OFFLINE (indexing)                                  ONLINE (serving)
┌──────────────────────────────────────────┐   ┌───────────────────────────────────────────────────┐
│ Sources → Parse → Clean → Chunk → Enrich │   │ User msg → Condense/rewrite → Route → Retrieve    │
│   → Embed (dense/sparse/multi-vector)    │   │  (hybrid, filtered by ACL) → Fuse → Rerank        │
│   → Index (+ metadata, ACLs, versions)   │   │  → Context assembly → LLM → Guardrail → Cite      │
└──────────────────────────────────────────┘   │  → Log/trace → Feedback → Eval → Improve          │
                                               └───────────────────────────────────────────────────┘
        Evaluation wraps BOTH halves: component metrics + end-to-end metrics + online signals.
```

**Iron law:** if the right evidence never reaches the context window, no LLM can save you. If it does reach the window, most modern LLMs will use it, *provided the context is short, well ordered, and clean*.
