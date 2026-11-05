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

---

# Deep dive: worked cost & caching math (illustrative prices; plug in your own)

Assume a mid-tier model at **$3 per 1M input tokens** and **$15 per 1M output tokens**, with cached input at 10% of the base price.

Per query: 6,000 input tokens (of which a 1,500-token system/tool prefix is cacheable) + 400 output tokens.
- No caching: 6,000 × $3/1M = $0.0180 + 400 × $15/1M = $0.0060 → **$0.0240**
- Prefix cached: 1,500 × $0.30/1M = $0.00045 + 4,500 × $3/1M = $0.0135 + $0.0060 → **$0.0200** (−17%)
- Halve the retrieved context with reranking and compression (4,500 → 2,250 input): $0.00045 + $0.00675 + $0.006 = **$0.0132** (−45%)
- Route 60% of traffic to a small model at 1/10 the price: blended ≈ 0.4 × 0.0132 + 0.6 × 0.00132 = **$0.0061** (−75% vs. the start)
At 50k queries/day: $1,200/day → ~$305/day. Retrieval-side optimizations (better reranking → fewer tokens) often save more than negotiating model prices.

## Semantic cache, mechanics and pitfalls
```
key = embed(normalize(standalone_query)); scope = (tenant, permission_hash, corpus_version, language)
hit = cache.search(key, scope=scope, min_cosine=0.95)
if hit: return hit.answer (+ "cached" flag in the trace)
```
- **Threshold too low** → wrong answers ("reset my password" vs. "reset my PIN" at cos 0.93).
- Invalidate on `corpus_version` bump or by `doc_id` dependency tracking.
- Never share entries across permission scopes.

## Latency trick: speculative parallelism
Start retrieval with the *raw* query at the same time as condensing. If the condensed query ≈ the raw one (cos > 0.9), reuse those results; otherwise re-retrieve. This saves one serial LLM hop for most single-turn traffic.
