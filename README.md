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
| 18 | [Mastery quiz: questions](docs/18-quiz-questions.md) | About 130 questions (fundamentals → expert, with calculations and design scenarios) to test yourself |
| 19 | [Mastery quiz: answers](docs/19-quiz-answers.md) | A worked answer key for every question |

### Suggested pace

- **Weeks 1–2:** Modules 00–05. Build a naive RAG baseline and its evaluation harness.
- **Weeks 3–4:** Modules 06–08. Add hybrid search and reranking, and measure the gain.
- **Weeks 5–6:** Modules 09–11. Add agentic and graph retrieval and memory, and justify each with the evaluation.
- **Weeks 7–8:** Modules 12–15. Add production hardening and do a red-team pass.
- **Ongoing:** Module 16 milestones and Module 17 papers.

Each module ends with **"Expert checks"** and a **"Deep dive"** section: internals, formulas, model reference tables, and worked numeric examples (BM25, RRF, score normalization, cross-encoder scoring, InfoNCE, nDCG, Cohen's κ, cost math). When you finish, take the [mastery quiz](docs/18-quiz-questions.md).

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

---

## The code: every module implemented in `src/chatbot/`

| Doc | Package | What's implemented |
|---|---|---|
| 00 | `core/`, `pipeline.py`, `ingestion.py` | Canonical types, ACL/filter DSL, F1–F15 failure taxonomy + funnel diagnosis; both planes wired end to end |
| 01 | `ingest/` | Filesystem connector (incremental sync, tombstones, ACL sidecars), Markdown/HTML/text/email/code/PDF/Docling parsers, cleaning, parse-quality scoring, exact + MinHash near-dup |
| 02 | `chunking/` | Fixed, sentence, paragraph, sliding, page, recursive, structure-aware, semantic, parent-child, sentence-window, proposition, late, code-AST, table, RAPTOR; `default_chunker()` recipe |
| 03 | `enrich/` | Contextual retrieval (cached whole-doc prefix), hypothetical questions, rule + LLM metadata, summaries, glossary; versioned `Enricher` |
| 04 | `embed/` | Embedder base (query prefixes, normalisation), hashing / sentence-transformers / Voyage, query cache, int8 / binary + rescoring, Matryoshka |
| 05 | `index/` | In-memory vector store, BM25F, Qdrant store (ACL inside the query), schema, blue/green aliases, idempotent `IndexWriter`, parent doc store |
| 06 | `query/` | One structured query-understanding call, router, condensing, rewrite, HyDE, multi-query, decomposition, step-back, validated self-query filters |
| 07 | `retrieve/` | Dense, sparse, parallel hybrid, RRF / weighted / convex fusion, MMR, neighbour + parent expansion, boosts, dynamic K |
| 08 | `rerank/` | Lexical, cross-encoder, LLM pointwise / listwise, cascade, business priors, abstention-threshold calibration, context post-processing |
| 09 | `agent/` | Tool-using agentic RAG (budgets, loop detection), CRAG, HippoRAG-style PPR graph retrieval, deep research |
| 10 | `generate/` | Cache-friendly prompts, budgeted XML context builder, streaming answer generator, citation parsing + verification, compression |
| 11 | `memory/` | Sessions (window + running summary + slots; in-memory / Redis), long-term memory with user controls and a PII screen |
| 12 | `evaluation/` | Golden sets with frozen splits, retrieval + generation metrics, calibrated LLM judges, bootstrap CIs, runner + funnel, reports, CI gate, synthetic data |
| 13 | `guard/` | PII / secret redaction, prompt-injection detection (direct + retrieved chunks), input / output guards, groundedness, exfiltration-safe markdown, ACL matrix, poisoning scan |
| 14 | `obs/`, `ops/` | Redacted tracing + replay, metrics + cost, permission-scoped semantic cache, rate limits / budgets, feedback → golden candidates, freshness SLO |
| 16 | `config.py`, `build.py`, `api/`, `cli.py`, `configs/`, CI | Config-driven assembly, FastAPI service, CLI, Docker, GitHub Actions with the eval gate |

### Quick start

```bash
pip install -e ".[dev,anthropic,api]"
```

```bash
export ANTHROPIC_API_KEY=...   # or `ant auth login`
```

```bash
chatbot --config configs/pipeline.yaml ingest data/sample_corpus
```

```bash
chatbot --config configs/pipeline.yaml ask -q "How long do refunds take by credit card?"
```

```bash
chatbot --config configs/pipeline.yaml serve --port 8000
```

```bash
curl -s localhost:8000/v1/chat -H 'X-User-Id: alice' -H 'X-User-Groups: employees' \
  -H 'content-type: application/json' -d '{"message": "What is the PTO carry-over limit?"}'
```

Offline (no API key; deterministic fake LLM) evaluation and gate, as run in CI:

```bash
chatbot --config configs/ci.yaml eval --dataset eval/datasets/golden_dev.jsonl eval/datasets/unanswerable.jsonl eval/datasets/adversarial.jsonl --gate configs/eval_gate.yaml
```

Tests, lint and types:

```bash
make lint type test
```

**Models:** generation and agentic routes use Claude Opus 5.5 (`claude-opus-5-5`); routing, condensing,
grading and enrichment use Claude Haiku 5.5 (`claude-haiku-5-5`). Both are configurable in `configs/pipeline.yaml`.

**Auth:** the API trusts identity headers from an authenticating gateway (`CHATBOT_AUTH_MODE=gateway`,
protect with `CHATBOT_GATEWAY_SECRET`), or uses static API keys (`CHATBOT_AUTH_MODE=api_key`,
`CHATBOT_API_KEYS`). Group memberships drive the ACL pre-filter on every retrieval leg.
