# 14 — Production & LLMOps

## 1. Latency budget (example: p95 target 3.5 s to the first useful token)

| Stage | Typical | Optimization |
|---|---|---|
| Auth + guardrail input | 30–150 ms | Small classifiers, run in parallel with condensing |
| Condense / route (small LLM) | 150–400 ms | One structured call; cache; skip for single-turn |
| Query embedding | 10–80 ms | Self-host or cache; batch |
| Hybrid retrieval | 20–150 ms | Parallel legs; tuned ef_search; filters indexed |
| Rerank (100 docs) | 50–300 ms | GPU, smaller reranker, fewer candidates, cascade |
| LLM time-to-first-token | 300–1500 ms | Prompt caching, smaller model per route, streaming |
| Output guardrail | 100–500 ms | Stream with async checks, or check on chunks; tiered |

Parallelize everything you can (guardrails ‖ condensing; BM25 ‖ dense). Agentic routes get a separate, looser SLO.

## 2. Cost levers
- **Model routing / cascades:** a small model for most traffic; escalate on low confidence.
- **Prompt caching** of the system prompt + stable context (large savings on cached tokens with major providers).
- **Semantic caching** of answers (GPTCache / Redis semantic cache) for FAQ-like traffic. Key by permission scope; invalidate on document updates; use a high similarity threshold.
- **Context budget:** fewer, better chunks (reranking + compression).
- **Batch APIs** for ingest enrichment and offline evaluation.
- **Embedding quantization** (int8 / binary) to shrink vector DB cost.
- Track **$/resolved conversation**, not just $/call.

## 3. Observability
- **Tracing:** OpenTelemetry GenAI semantic conventions; tools such as Langfuse, Arize Phoenix, LangSmith, Braintrust, Helicone, Datadog LLM Observability.
- Log per request: query, condensed query, route, filters, retrieved IDs + scores (each stage), reranked IDs, final context IDs, prompt version, model, tokens, latency per stage, answer, citations, guardrail verdicts, user feedback.
- **Replay:** ability to rerun any production trace against a new pipeline version (powerful for regression testing).
- Privacy: redact PII in logs; retention policies.

## 4. Index and knowledge lifecycle
- Incremental ingestion with change detection (`content_hash`); deletes propagate.
- Freshness SLO dashboard (source update → searchable).
- Stale-content detection: documents not updated in N months, conflicting duplicates, low-authority copies.
- Coverage analytics: cluster unanswered or low-score queries → content gaps → tell content owners.
- Re-embedding and re-enrichment migrations with blue/green aliases.
- Re-run the evaluation after major corpus changes.

## 5. Deployment architecture
- Stateless API service (FastAPI or similar) + async workers for ingestion (queue: SQS, Kafka, Celery, or Temporal for durable workflows).
- Separate the **indexing plane** from the **serving plane**.
- Model gateway (LiteLLM, Portkey, or a cloud gateway) for provider failover, keys, budgets, and rate limits.
- Config-driven pipelines (prompt and pipeline versions as artifacts); feature flags for A/B tests.
- Multi-region and data-residency needs; on-prem or VPC options for regulated customers.

## 6. Team and process
- Weekly **error review**: 30–50 sampled conversations, F-code tagging, prioritized backlog.
- Content owners in the loop (the bot is only as good as the docs).
- Release train: offline eval gate → canary → A/B → full rollout.
- Incident playbooks: bad answer reported → trace → root cause → regression test added.

## Expert checks
- Where does semantic caching become a security bug?
- Which three numbers would you put on the executive dashboard?
- How do you replay production traffic safely against a new pipeline?
