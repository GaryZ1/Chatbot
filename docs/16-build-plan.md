# 16 — Build Plan & Roadmap

A phased plan to build the chatbot **in this repo**. Each phase has deliverables, an evaluation gate, and the modules it applies. **No phase starts until the previous gate passes.**

## Target stack (swappable; every component behind an interface)

| Layer | Default choice | Alternatives |
|---|---|---|
| Language / API | Python 3.12, FastAPI, Pydantic | — |
| Orchestration | Plain Python pipelines + LangGraph for agentic flows | LlamaIndex, Haystack, DSPy |
| Parsing | Docling (+ VLM fallback for low-quality pages) | MinerU, Unstructured, LlamaParse |
| Chunking | Structure-aware + recursive (own implementation, Chonkie for experiments) | — |
| Enrichment | Contextual retrieval with a small LLM + prompt caching | Late chunking |
| Embeddings | Benchmark: Qwen3-Embedding-0.6B/4B (local) vs. Voyage / Gemini / OpenAI / Cohere (API) | BGE-M3 |
| Sparse | BM25 (via the DB) → SPLADE / BGE-M3 sparse experiment | — |
| Vector DB | Qdrant (or pgvector if on Postgres) | Weaviate, Vespa, OpenSearch |
| Reranker | BGE-reranker-v2-m3 → Qwen3-Reranker / Cohere Rerank experiment | — |
| LLMs | Tiered: small (route/condense), mid (answer), frontier (agent/complex) via a gateway (LiteLLM) | — |
| Evaluation | RAGAS + DeepEval + custom judges; pytest gates | ARES, promptfoo |
| Observability | Langfuse or Arize Phoenix (OpenTelemetry) | LangSmith |
| Guardrails | Llama Guard / Prompt Guard + a custom groundedness check | NeMo Guardrails |
| UI | Streamlit/Gradio for internal use → Next.js chat UI with citations | Slack bot |

## Proposed repo layout

```
Chatbot/
  docs/                      # this curriculum
  src/chatbot/
    ingest/      connectors.py  parse.py  clean.py  dedupe.py
    chunking/    base.py  fixed.py  recursive.py  structure.py  semantic.py
                 parent_child.py  proposition.py  late.py  code_ast.py  table.py
    enrich/      contextual.py  questions.py  metadata.py  summaries.py
    embed/       base.py  providers/*.py  quantize.py
    index/       qdrant_store.py  pgvector_store.py  bm25.py  schema.py
    query/       condense.py  rewrite.py  hyde.py  multi_query.py  router.py  filters.py
    retrieve/    dense.py  sparse.py  hybrid.py  fusion.py  mmr.py  expand.py
    rerank/      cross_encoder.py  llm_rerank.py  colbert.py
    generate/    prompts/  context_builder.py  answer.py  citations.py
    agent/       graph.py  tools.py  crag.py
    guard/       input.py  output.py  acl.py  pii.py
    memory/      session.py  long_term.py
    api/         app.py  schemas.py
    obs/         tracing.py
  eval/
    datasets/    golden_dev.jsonl  golden_test.jsonl  adversarial.jsonl  unanswerable.jsonl
    metrics/     retrieval.py  generation.py  judges/
    experiments/ chunking_grid.py  embedder_bakeoff.py  rerank_ablation.py
    reports/
  configs/       pipeline.yaml  experiments/*.yaml
  tests/         unit/  integration/  acl_matrix/  regression/
```

---

## Phase 0: Foundations (week 1)
**Modules:** 00, 12
- [ ] Pick a pilot corpus (e.g. 500–5,000 public documents: product docs, policies, or a public 10-K set) and a target user persona.
- [ ] Write the **design decisions doc** (Module 00 §4): corpus profile, query profile, SLOs, security model, answer contract.
- [ ] Build **golden set v0**: 150 questions (50 SME-written + 100 synthetic, filtered), including 15% unanswerable and 15% multi-hop.
- [ ] Evaluation harness: retrieval metrics (Recall@k, nDCG, MRR) + RAGAS faithfulness / answer correctness + latency and cost capture.
- **Gate:** the evaluation runs end-to-end on a dummy pipeline and produces a report.

## Phase 1: Naive baseline (week 2)
**Modules:** 01, 02, 04, 05, 07, 10
- [ ] Ingestion with Docling → typed blocks → canonical schema.
- [ ] Recursive 512-token chunking; one embedding model; Qdrant; top-5 dense; a simple grounded prompt with citations.
- [ ] FastAPI `/chat` with streaming; minimal UI.
- **Gate:** baseline numbers recorded (this is the bar every later change must beat).

## Phase 2: Chunking & enrichment lab (week 3)
**Modules:** 02, 03
- [ ] Implement chunkers: fixed, recursive, structure-aware, semantic, page, parent-child, sentence-window, proposition (sample only), late chunking.
- [ ] Run the grid (size × overlap × method) **normalized by context tokens**; per-document-type results.
- [ ] Add breadcrumbs → contextual retrieval → hypothetical questions; ablate each.
- **Gate:** the chosen config beats baseline Recall@10 by ≥ 10% relative, with no slice regression. Results written to `eval/reports/chunking.md`.

## Phase 3: Retrieval & ranking (week 4)
**Modules:** 04, 06, 07, 08
- [ ] Embedder bake-off (4–6 models) → choose by quality per dollar; try int8/binary + rescoring.
- [ ] BM25 + dense hybrid with RRF; tune α / weights; try learned sparse.
- [ ] Reranker ablation (none / BGE / Qwen3 / Cohere) × candidates (25/50/100/150).
- [ ] Query condensing + multi-query + HyDE ablations by intent bucket.
- [ ] MMR / dedup / neighbor expansion; calibrated reranker threshold for abstention.
- **Gate:** Recall@final_k and end-to-end correctness both up significantly (paired bootstrap, 95% CI); p95 within the SLO.

## Phase 4: Conversation, routing & context engineering (week 5)
**Modules:** 06, 10, 11
- [ ] Single structured "query understanding" call (intent, standalone query, filters, sub-queries).
- [ ] Routes: smalltalk / simple RAG / decomposed RAG / SQL tool / escalate.
- [ ] Context builder: token budget, ordering, grouping, compression experiment; prompt caching.
- [ ] Session memory (summary + slots); optional long-term memory with user controls.
- [ ] Citation verification (claim → source check).
- **Gate:** multi-turn evaluation set (50 conversations) passes; citation precision ≥ 0.9.

## Phase 5: Advanced retrieval (week 6)
**Modules:** 09
- [ ] CRAG-style self-correction loop (grade → rewrite → retry, capped).
- [ ] Agentic RAG with hierarchical tools (search / read section / list docs) for the "complex" route only.
- [ ] Experiment: LightRAG or HippoRAG 2 on the multi-hop slice vs. the agentic baseline.
- [ ] Experiment: ColPali / ColQwen on slide- and chart-heavy documents vs. OCR + text.
- **Gate:** adopt only what improves its target slice without exceeding the cost/latency budget for that route.

## Phase 6: Safety & security (week 7)
**Modules:** 13
- [ ] ACLs at ingest + query-time pre-filter; **ACL matrix tests** in CI.
- [ ] Input guardrails (injection / jailbreak / PII); output groundedness + PII scan; markdown and URL sanitization.
- [ ] Adversarial suite (garak / promptfoo): indirect injection documents, poisoning documents, ACL probes.
- **Gate:** zero ACL leaks; attack success rate below target; no faithfulness regression.

## Phase 7: Production hardening (week 8)
**Modules:** 14, 12
- [ ] Tracing for every stage; dashboards (quality, latency, cost, freshness).
- [ ] Incremental ingestion + delete propagation + blue/green re-index.
- [ ] Model gateway with fallbacks; semantic cache (permission-scoped); budgets.
- [ ] CI evaluation gate on every PR; nightly full evaluation; sampled production judging; feedback → golden-set pipeline.
- [ ] Load test at 2× expected QPS.
- **Gate:** SLOs met under load; runbooks written; a staged rollout plan.

## Phase 8: Continuous improvement (ongoing)
- Weekly error review with F-codes; content-gap reports to document owners.
- Quarterly: re-run the embedder/reranker bake-off with new models; fine-tune the embedder or reranker on accumulated feedback; refresh the golden set from production.
- Track the frontier (Module 17) and run experiments through the same gates.

---

## Experiment log template (`eval/reports/<date>-<name>.md`)
```
Hypothesis:
Change (config diff):
Dataset/split: golden_dev v3 (n=412)
Metrics (baseline → new, 95% CI): Recall@10, nDCG@10, Faithfulness, Correctness, p95 latency, $/query
Slice results:
Failure analysis (F-codes, 20 samples):
Decision: adopt / reject / iterate
```

## Capstone: proof you're in the top 0.1%
1. Defend every design choice in the pipeline with an experiment from your own evaluation.
2. Given any bad answer, locate the failing stage within 5 minutes using traces and the retrieval funnel.
3. Explain the trade-offs among all 20 chunking methods, 6 retrieval paradigms, and 8 reranker types without notes.
4. Ship a change through offline eval → canary → A/B with statistically valid results.
5. Pass the ACL and red-team suites with zero leaks.
