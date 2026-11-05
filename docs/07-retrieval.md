# 07 — Retrieval

## 1. Retrieval paradigms

| Paradigm | What | Pros | Cons | Role |
|---|---|---|---|---|
| **Sparse lexical (BM25 / BM25F)** | Term-frequency scoring over an inverted index; BM25F weights fields (title > body) | Exact IDs, codes, names, acronyms; zero training; explainable; strong out-of-domain baseline (BEIR) | Vocabulary mismatch; no semantics; needs language analyzers (stemming, CJK tokenization) | Always on, in hybrid |
| **Dense (bi-encoder)** | ANN over embeddings | Paraphrase and semantic matching; multilingual | Misses rare terms; domain shift; opaque | Main semantic retriever |
| **Learned sparse (SPLADE, ELSER, BGE-M3 sparse)** | Model predicts term weights + expansions | Best of both on many benchmarks; inverted index efficiency | More compute at ingest; integration | Replaces or augments BM25 |
| **Late interaction (ColBERT/PLAID)** | Token-level MaxSim | High quality, robust out of domain | Storage, serving complexity | First stage (moderate scale) or rerank |
| **Hybrid** | Run sparse + dense in parallel, fuse (RRF or weighted) | Consistently best simple baseline across studies; robust | Two indices; tuning weights | **Production default** |
| **Metadata / structured** | Filters, SQL, knowledge-graph lookups | Precise | Needs structure | Combined with the above |
| **Generative / agentic retrieval** | LLM issues searches iteratively, reads, refines (Module 09) | Multi-hop | Latency | Complex queries |

## 2. Hybrid search, done right
1. Run BM25 (top 50–100) and dense (top 50–100) **in parallel**, both with the **same ACL + metadata pre-filters**.
2. Fuse with **RRF (k=60)**, or a convex combination `α·norm(dense) + (1-α)·norm(bm25)` with α tuned on the evaluation (typically 0.5–0.8 dense). Min-max or z-score normalization per query.
3. Optional third leg: a **hypothetical-question index** (Module 03) or a **title/summary index**.
4. Pass the fused top 50–150 to the reranker (Module 08) → final top 5–20 to context.

Anthropic's contextual retrieval data illustrates the stacking: contextual embeddings alone −35% failures; + contextual BM25 −49%; + reranking −67%.

## 3. Multi-stage retrieval patterns

| Pattern | Description | Benefit |
|---|---|---|
| **Retrieve → rerank** | Cheap high-recall stage → expensive precise stage | The standard |
| **Doc → chunk (hierarchical)** | First retrieve documents via summaries or titles, then chunks within the top documents | Avoids near-duplicate chunks from irrelevant similar documents |
| **Small-to-big** | Match children, return parents | Precision + sufficiency |
| **Neighbor expansion** | Add chunks i−1 and i+1 of top hits | Fix boundary cuts |
| **Graph expansion** | Follow cross-references or KG edges from hits | Multi-hop and completeness |
| **MMR (Maximal Marginal Relevance)** | `λ·sim(q,d) − (1−λ)·max sim(d, selected)` | Diversity; removes near-duplicates |
| **Recency / authority boosting** | Final score += w·freshness + w·authority | Fix stale or draft answers |
| **Iterative (IRCoT, FLARE)** | Interleave retrieval with reasoning steps; FLARE retrieves when generation confidence drops | Multi-hop, long-form generation |

## 4. Choosing K
- First stage: high recall. K=50–200 (cheap).
- After rerank: precision. Choose by **token budget**, not a magic number: e.g. fill up to 8–20k tokens for a long-context model, 3–6k for a cheap or fast model.
- Dynamic K: cut where reranker scores drop sharply or below a calibrated threshold. This allows **abstaining** when nothing is relevant.

## 5. Retrieval metrics (details in Module 12)
- **Recall@k** (did any or all gold chunks appear?) is the key first-stage metric.
- **nDCG@10, MRR, Precision@k**: ranking quality.
- **Context precision / recall** (RAGAS-style, LLM-judged, label-free).
- Latency p50/p95 per stage; filter selectivity effects.

## 6. Common bugs
- Query embedded without the query instruction or prefix.
- Different tokenizers or analyzers for indexing vs. querying in BM25.
- ACL filters applied after top-K (leak + fewer results).
- Duplicate chunks (versions, copies) filling the top-K → dedup by `content_hash` and `doc_id`.
- Cosine metric on unnormalized vectors.

## Expert checks
- Why does hybrid search beat dense-only even with state-of-the-art embedders?
- What does MMR trade off, and when does diversity *hurt*?
- How do you set K to enable principled abstention?
