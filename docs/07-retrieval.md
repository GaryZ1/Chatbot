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

---

# Deep dive: how sparse, dense, and hybrid retrieval work, with numbers

## A. Sparse retrieval: BM25 step by step

**Inverted index:** for each term, a postings list of (doc_id, term frequency). The query "reset E1043" only touches the postings for `reset` and `e1043`, which is why it's fast.

**BM25 formula** (per query term t, summed over the query's terms):
```
score(D, Q) = Σ_t  IDF(t) · [ tf(t,D) · (k1 + 1) ] / [ tf(t,D) + k1 · (1 − b + b · |D| / avgdl) ]
IDF(t)      = ln( (N − n_t + 0.5) / (n_t + 0.5) + 1 )
```
- `tf(t,D)`: how often t appears in D. *Saturates*: the 10th occurrence adds much less than the 1st (controlled by **k1**, typically 1.2–2.0).
- `|D| / avgdl`: length normalization. A long document gets penalized (controlled by **b**, typically 0.75).
- `IDF`: rare terms matter more.

**Worked example.** N = 1,000 docs, avgdl = 200 tokens, k1 = 1.2, b = 0.75. Doc A has 120 tokens, contains "e1043" twice and "reset" once. "e1043" appears in 2 docs and "reset" in 300.

| Term | IDF | TF part | Contribution |
|---|---|---|---|
| e1043 | ln(998.5/2.5 + 1) = ln(400.4) = **5.99** | 2·2.2 / (2 + 1.2·(0.25 + 0.75·0.6)) = 4.4/2.84 = **1.549** | 9.28 |
| reset | ln(700.5/300.5 + 1) = ln(3.331) = **1.20** | 1·2.2 / (1 + 0.84) = **1.196** | 1.44 |
| | | **BM25(A)** | **10.72** |

Notice the rare error code contributes 6× more than "reset". **That is why BM25 is unbeatable for IDs, codes and names**, while a dense model may map "E1043" to some vague region of vector space.

**BM25F** weights fields: e.g. a title match counts 3×, a heading match 2×, body 1×.

## B. Learned sparse (SPLADE): how it's built
- Run the text through BERT. For each token position, the **masked-language-model head** gives logits over the whole vocabulary (~30,522 terms).
- Weight per vocabulary term: `w_j = max_over_positions log(1 + ReLU(logit_ij))`, i.e. max pooling.
- Result: a sparse vector over the vocabulary, with **expansion terms** that weren't in the text (the document "how to reset MFA" also gets weight on "authentication", "password", "2fa").
- A FLOPS regularizer in training keeps it sparse (~100–300 non-zero terms per doc) so an inverted index still works.
- Score = dot product of the query and document sparse vectors.

## C. Dense retrieval: how a score is computed
- `q⃗ = Enc("query: how do I reset mfa")`, `d⃗ = Enc(chunk)`, both L2-normalized → **cosine = dot product**.
- Tiny 2-D example: q⃗ = [0.6, 0.8]; d1⃗ = [0.8, 0.6] → cos = 0.48 + 0.48 = **0.96**; d2⃗ = [1, 0] → cos = **0.60**. So d1 ranks higher.
- ANN (HNSW, Module 05) finds the top-K highest-cosine vectors without scanning all of them.
- **Score scale:** cosines from a given model sit in a narrow band (e.g. 0.55–0.90). BM25 scores are unbounded (0–30+). **You cannot add them directly.** Hence fusion.

## D. Hybrid fusion: combining sparse + dense

Running example. Query: **"reset MFA token error E1043"**. Each retriever returns its own top 4:

| Rank | BM25 (score) | Dense (cosine) |
|---|---|---|
| 1 | A (12.4) | C (0.82) |
| 2 | B (9.1) | A (0.79) |
| 3 | C (7.8) | E (0.74) |
| 4 | D (3.2) | B (0.61) |

A = KB article that mentions E1043 verbatim; C = "re-enrolling your authenticator app" (semantic match, no code); B = an error-code list; D = an unrelated page that says "token" a lot; E = an MFA FAQ.

### D1. Reciprocal Rank Fusion (RRF), rank-based

```
RRF(d) = Σ_r  1 / (k + rank_r(d))        k = 60 (from Cormack et al. 2009); a missing doc contributes 0
```
| Doc | BM25 term | Dense term | **RRF** | Final rank |
|---|---|---|---|---|
| A | 1/(60+1) = 0.016393 | 1/(60+2) = 0.016129 | **0.032522** | 1 |
| C | 1/(60+3) = 0.015873 | 1/(60+1) = 0.016393 | **0.032266** | 2 |
| B | 1/(60+2) = 0.016129 | 1/(60+4) = 0.015625 | **0.031754** | 3 |
| E | — | 1/(60+3) = 0.015873 | **0.015873** | 4 |
| D | 1/(60+4) = 0.015625 | — | **0.015625** | 5 |

**Why RRF works:**
- It uses **only ranks**, so you never need to make BM25's 12.4 comparable with a cosine of 0.82.
- Documents found by **both** retrievers (A, C, B) beat documents found by one (E, D). Agreement is strong evidence.
- **k** dampens the gap between the top ranks: with k = 60, rank 1 vs. rank 2 differ by only ~1.6%, so one retriever's top pick can't dominate. A smaller k (e.g. 10) makes the top ranks matter more.
- **Weighted RRF:** `w_bm25/(k+rank) + w_dense/(k+rank)`, e.g. weights 0.3/0.7 if dense is usually better on your evaluation.
- **Downside:** it throws away score *magnitude*. If dense is extremely confident (0.95 vs. 0.60), RRF can't tell.

### D2. Score normalization + convex combination, score-based

You first map each retriever's scores to a common [0,1] scale with a **norm()** function, then blend:
```
hybrid(d) = α · norm(dense(d)) + (1 − α) · norm(bm25(d))
```

**norm() option 1, min-max** (per query, per retriever): `norm(x) = (x − min) / (max − min)`
- BM25: min 3.2, max 12.4, range 9.2 → A = 1.000, B = (9.1−3.2)/9.2 = 0.641, C = (7.8−3.2)/9.2 = 0.500, D = 0.000
- Dense: min 0.61, max 0.82, range 0.21 → C = 1.000, A = 0.18/0.21 = 0.857, E = 0.13/0.21 = 0.619, B = 0.000
- Missing docs get 0.

With **α = 0.6** (favor dense):

| Doc | 0.6 · norm(dense) | 0.4 · norm(bm25) | **Hybrid** |
|---|---|---|---|
| A | 0.6·0.857 = 0.514 | 0.4·1.000 = 0.400 | **0.914** |
| C | 0.6·1.000 = 0.600 | 0.4·0.500 = 0.200 | **0.800** |
| E | 0.6·0.619 = 0.371 | 0 | **0.371** |
| B | 0.6·0.000 = 0.000 | 0.4·0.641 = 0.256 | **0.256** |
| D | 0 | 0.4·0.000 = 0.000 | **0.000** |

Note the min-max pitfalls in this example:
1. **B's dense cosine of 0.61 became 0.0** just because it was the lowest of the four, not because it's irrelevant.
2. Min-max is **sensitive to outliers**: one huge BM25 score squashes everyone else toward 0.
3. With one result, `max − min = 0` → divide by zero (guard it).

**norm() option 2, z-score:** `norm(x) = (x − μ) / σ` over that query's result list.
- BM25: μ = 8.125, σ = 3.30 → A = +1.30, B = +0.30, C = −0.10, D = −1.49
- Dense: μ = 0.74, σ = 0.080 → C = +1.00, A = +0.62, E = 0.00, B = −1.62
- Better at keeping relative spread; unbounded (you can squash it with a sigmoid); needs enough results to estimate σ.

**norm() option 3, distribution-based score fusion (DBSF)** (used by Qdrant): normalize with `(x − (μ − 3σ)) / (6σ)`, clipped to [0,1]. Robust to outliers.

**norm() option 4, theoretical bounds:** cosine → (cos + 1)/2. BM25 has no fixed upper bound, so you'd divide by the query's max possible score (Elastic/OpenSearch offer `l2_norm` or `min_max` normalization processors).

**Tuning α:** sweep α ∈ {0, 0.1, …, 1.0} on the dev golden set, measure nDCG@10, and pick the plateau. Typical optimum is 0.5–0.8 dense. On ID-heavy traffic (part numbers, error codes) the optimum shifts toward BM25. Some systems route α per query (e.g. lower α when the query contains a code pattern like `[A-Z]\d{4}`).

### D3. Which fusion to use?
| Situation | Use |
|---|---|
| Default, no tuning data | **RRF (k=60)**: robust, zero tuning |
| Have a dev set and score magnitudes matter | Convex combination with tuned α (z-score or DBSF) |
| Many signals (BM25, dense, recency, authority, clicks) | Learning-to-rank (LambdaMART) over the features |
| More than 2 retrievers or multi-query | RRF (extends to any number of lists) |

Many engines implement this natively: Elasticsearch/OpenSearch `rrf` retriever and normalization processors, Weaviate `hybrid(alpha=…)` (relativeScoreFusion or rankedFusion), Qdrant `Fusion.RRF` / `Fusion.DBSF`, Vespa rank profiles, pgvector + SQL CTEs, Azure AI Search hybrid (RRF).

## E. Hybrid search query (pseudo-code)
```python
bm25_hits  = bm25.search(q_text, k=100, filter=acl & meta)          # parallel
dense_hits = vdb.search(embed_query(q_text), k=100, filter=acl & meta)
fused = rrf([bm25_hits, dense_hits], k=60)[:150]
reranked = reranker.rank(q_text, fused)                              # Module 08
context = select_by_token_budget(mmr(reranked, lam=0.7), budget=8000)
```

## F. MMR worked example (diversity)
`MMR(d) = λ·sim(q,d) − (1−λ)·max_{s∈Selected} sim(d,s)`, λ = 0.7.
Candidates: X (sim to q 0.90), Y (0.88, near-duplicate of X: sim(X,Y) = 0.97), Z (0.80, sim(X,Z) = 0.30).
1. Pick X (highest relevance).
2. Y: 0.7·0.88 − 0.3·0.97 = 0.616 − 0.291 = **0.325**. Z: 0.7·0.80 − 0.3·0.30 = 0.560 − 0.090 = **0.470** → pick **Z**.

MMR skips the duplicate Y and gives the LLM new information.
