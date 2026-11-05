# 08 — Reranking

Reranking re-scores the first-stage candidates with a more expensive model that sees the **query and document together**. It is usually the **single biggest quality gain per engineering hour** after hybrid search.

## 1. Reranker types

| Type | How | Examples | Pros | Cons | Latency (100 docs) |
|---|---|---|---|---|---|
| **Cross-encoder (pointwise)** | `[CLS] q [SEP] d` → relevance score | BGE-reranker-v2-m3, mxbai-rerank-v2, Jina reranker v2/v3, ms-marco MiniLM, Cohere Rerank 3.5/4, Voyage rerank-2.5 | Big precision gains; calibrated-ish scores; mature | O(K) transformer passes; max length | ~50–300 ms on GPU / API |
| **LLM-based pointwise** | Small LLM outputs P("yes") relevance | Qwen3-Reranker (0.6B/4B/8B), bge-reranker-v2-gemma, RankLLaMA | Highest open quality; instruction-following ("prefer official policy") | GPU heavy | 100–800 ms |
| **Listwise LLM** | LLM reorders a list (sliding window) | RankGPT, RankZephyr, FIRST, general LLMs | Considers documents jointly; strong on reasoning (BRIGHT) | Slow, costly, position bias, no scores | Seconds |
| **Pairwise** | Compare pairs | PRP (pairwise ranking prompting) | Accurate | O(K²) | Slow |
| **Reasoning rerankers** | Reasoning LLM thinks before scoring | Rank1, ReasonRank, and similar (2025–26) | Best on reasoning-intensive retrieval | Very slow | Seconds |
| **Late interaction** | ColBERT MaxSim over candidates | ColBERTv2, Jina-ColBERT, GTE-ModernColBERT | Cheap reranking (precomputed document tokens) | Less accurate than cross-encoders | ~10–30 ms |
| **Learning-to-rank (feature-based)** | GBDT (LightGBM LambdaMART) over features: BM25, dense, CE score, recency, authority, clicks | Combines business signals | Needs labels/clicks | ~1 ms |
| **Rule / business boosts** | Recency, source authority, `is_latest` | Simple, controllable | Not semantic | ~0 |

## 2. Typical 2026 choices
- **Default self-hosted:** BGE-reranker-v2-m3 (multilingual, cheap); Qwen3-Reranker-4B/8B for top open quality.
- **Default API:** Cohere Rerank (v3.5/v4) or Voyage rerank-2.5; Jina reranker.
- **Reasoning-heavy queries:** an LLM listwise or reasoning reranker *on the top 20 only*.
*(2026 comparisons in Redis and FutureAGI guides are vendor-authored; benchmark it yourself.)*

## 3. Design rules
1. **Rerank 50–150 candidates → keep 5–20.** Measure Recall@final_k vs. candidate count. The gain flattens.
2. Rerank on the **embed text with breadcrumbs/context** but return the original text.
3. Respect max length (often 512–8k tokens). Truncate intelligently (keep the heading + the most query-relevant window).
4. **Cascade:** cheap CE on 150 → LLM reranker on the top 20 only for hard queries (router decides).
5. **Score thresholds** for abstention: calibrate on the evaluation set (e.g. "if the max score < τ → say I don't know or ask to clarify").
6. Add business priors *after* semantic reranking: `final = ce + β·recency + γ·authority`.
7. Fine-tune the reranker on your domain (with clicks, thumbs, or LLM-labeled pairs). Often a bigger win than fine-tuning the embedder.

## 4. Context-level post-processing (after reranking)
- **Dedup** near-identical chunks (versions, copies).
- **Ordering:** most relevant first *and* last ("lost in the middle", Liu et al. 2023); or document order for narrative coherence.
- **Compression:** extractive (keep relevant sentences: LongLLMLingua, RECOMP-extractive, cross-encoder sentence scores) or abstractive (summarize). Cuts tokens 2–5×, but risks dropping key details.
- **Grouping** by document with a source header for clean citations.

## Expert checks
- Why do cross-encoders beat bi-encoders, and why can't you use them for first-stage retrieval?
- When is a listwise LLM reranker worth its latency?
- How do you calibrate a reranker threshold for "I don't know"?
