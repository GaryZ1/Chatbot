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

---

# Deep dive: how rerankers actually work

## A. Bi-encoder vs. cross-encoder, mechanically

```
BI-ENCODER (first-stage retrieval)            CROSS-ENCODER (reranking)
 query ──► [Encoder] ──► q⃗ ─┐                  "[CLS] query [SEP] document [SEP]"
                            ├─ cos(q⃗, d⃗)                     │
 doc   ──► [Encoder] ──► d⃗ ─┘                       [one Transformer]
 (doc vectors precomputed offline)                          │
                                              h_[CLS] ──► Linear(768→1) ──► logit ──► σ ──► score
 Query and doc NEVER see each other.          Every query token attends to every doc token.
```

- **Bi-encoder:** each text is compressed into one vector *before* it knows what it is being compared to. Cost per query = 1 encoder pass + 1 ANN lookup, whatever the corpus size.
- **Cross-encoder:** the query and the document go through the transformer **together**, so self-attention can match "reset" in the query to "re-enroll your authenticator" in the document, token by token. This is much more precise. But nothing can be precomputed: scoring K candidates takes **K full transformer passes**, each over (|q| + |d|) tokens, with attention cost O(L²). That's why it is only used on the top 50–150 candidates.

## B. What `[CLS]`, `q`, `[SEP]`, `d` mean

BERT-family models use special tokens:

| Token | Meaning |
|---|---|
| `[CLS]` | "Classification" token, always first. It has no meaning of its own; after 12–24 layers of self-attention its output vector **summarizes the whole input pair**, and the classification head reads it |
| `q` | The query's wordpiece tokens |
| `[SEP]` | "Separator". It marks the end of segment A (query) and the end of segment B (document) |
| `d` | The document/passage tokens (truncated to fit max length, e.g. 512) |
| *segment (token-type) IDs* | 0 for `[CLS] q [SEP]`, 1 for `d [SEP]`. They tell the model which part is which |

Concrete input (tokenized):
```
[CLS] how do i reset mfa ? [SEP] to reset multi - factor authentication , open settings → security → re - enroll . [SEP]
seg:  0    0   0  0  0     0  0  0     1  1     1    1    1      ...                                                    1
```
RoBERTa/XLM-R models (e.g. **bge-reranker-v2-m3**) use `<s> q </s></s> d </s>` instead. Same idea, different symbols. Decoder-LLM rerankers use a chat/prompt template instead (see D3).

**Scoring head:** `score = σ(w · h_[CLS] + b)`, where `h_[CLS]` ∈ ℝ^768 (base) or ℝ^1024 (large). Example: if the logit is 2.3, then σ(2.3) = 1/(1+e^-2.3) = **0.909**. A logit of −1.1 gives σ = **0.250**. Rerankers sort by the logit (σ is monotonic). σ is applied only to get a 0–1 number for thresholds.

## C. How cross-encoders are built (training)

1. **Backbone:** a pretrained encoder (BERT, MiniLM, ELECTRA, RoBERTa/XLM-R, DeBERTa) or an LLM (T5, Gemma, Qwen, Llama, Mistral).
2. **Data:** (query, positive passage, hard negatives). The classic dataset is **MS MARCO** passage ranking (~500k training queries from Bing). Modern rerankers add many multilingual and domain sets plus synthetic LLM-generated queries.
3. **Hard negatives:** passages that BM25 or a dense retriever ranks highly but that are *not* relevant. Without them the model only learns easy distinctions.
4. **Loss functions:**
   - **Pointwise BCE** (monoBERT, Nogueira & Cho 2019): label 1 for a positive, 0 for a negative. `L = −[y·log σ(s) + (1−y)·log(1−σ(s))]`.
   - **Localized contrastive / listwise softmax (LCE):** one positive + n hard negatives in a group. `L = −log( e^{s+} / Σ_j e^{s_j} )`. This is the most common modern choice (BGE rerankers, Qwen3-Reranker).
   - **Distillation:** match a larger teacher's scores (MarginMSE: `(s_pos − s_neg)_student ≈ (s_pos − s_neg)_teacher`), or distill listwise LLM rankings into a small cross-encoder.
5. **Max length and truncation:** the training length (e.g. 512 or 8192) sets the usable inference length.
6. **Evaluation:** nDCG@10 on BEIR, MIRACL (multilingual), TREC DL, BRIGHT (reasoning).

**Worked LCE example:** a group of 1 positive and 3 negatives with logits [4.0, 1.0, 0.5, −1.0].
`e^4 = 54.60, e^1 = 2.718, e^0.5 = 1.649, e^-1 = 0.368` → sum = 59.33 → p(pos) = 54.60/59.33 = 0.920 → loss = −ln 0.920 = **0.083**. If the model had scored a negative at 4.5, the loss would jump, and the gradient pushes that negative down.

## D. Cross-encoder families (types)

### D1. Encoder-only cross-encoders (BERT-style), the classic
- **How:** `[CLS] q [SEP] d [SEP]` → transformer → linear head on `[CLS]`.
- **Models:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params, very fast, English); `ms-marco-MiniLM-L-12-v2`; `BAAI/bge-reranker-base/large`; **`BAAI/bge-reranker-v2-m3`** (~568M, built on BGE-M3 / XLM-RoBERTa-large, multilingual, up to 8k tokens, Apache-2.0); `jina-reranker-v2-base-multilingual` (~278M, sequence lengths up to 1024, non-commercial license); `mixedbread-ai/mxbai-rerank-base/large-v1`.
- **Pros:** fast on GPU, cheap, easy to fine-tune with `sentence-transformers` `CrossEncoder`.
- **Cons:** smaller capacity; less instruction-following.

### D2. Sequence-to-sequence rerankers (T5-style)
- **monoT5** (Nogueira et al. 2020): input `"Query: {q} Document: {d} Relevant:"`. The decoder produces one token; score = softmax over the logits of the tokens **"true"** vs. **"false"** → P(true).
- **duoT5:** pairwise. `"Query: q Document0: d_i Document1: d_j Relevant:"` → P(d_i more relevant than d_j). Accurate but O(K²).
- **Pros:** strong zero-shot. **Cons:** slower than encoders; mostly superseded by LLM rerankers.

### D3. Decoder-LLM pointwise rerankers (the 2025–26 frontier for open weights)
- **How:** a decoder LLM gets a prompt like
  `"<Instruct>: Given a web search query, retrieve relevant passages <Query>: {q} <Document>: {d}"`, followed by the question "Does the document meet the requirement? Answer only yes or no."
  Score = `softmax([logit("yes"), logit("no")])[yes]`. No generation loop: **one forward pass** that reads the final position's logits.
- **Models:** **Qwen3-Reranker-0.6B/4B/8B** (Apache-2.0, 32k context, instruction-aware: you can tell it "prefer official policy documents"); **bge-reranker-v2-gemma** (Gemma-2B); **bge-reranker-v2-minicpm-layerwise** (can exit at an earlier layer for speed); **mxbai-rerank-v2** (Qwen2.5-based 0.5B/1.5B, Apache-2.0); **RankLLaMA** (Llama-2 with the end-of-sequence token representation → linear head).
- **Pros:** highest open-weight quality; instructions; long context; multilingual.
- **Cons:** 0.5–8B params → needs a GPU; 3–20× the latency of MiniLM.

### D4. Listwise LLM rerankers
- **How:** give the LLM the query and a numbered list of passages and ask for a permutation: `[3] > [1] > [7] > ...`. To handle more passages than fit, use a **sliding window** (e.g. window 20, step 10) from the bottom of the list to the top, so strong documents "bubble up".
- **Models:** RankGPT (prompting GPT-4-class models), **RankZephyr / RankVicuna** (7B open models distilled from GPT-4 rankings), FIRST (uses only the first generated token's logits → faster), and general frontier LLMs.
- **Pros:** compares documents *against each other* (relative judgments are easier than absolute ones); best on reasoning-heavy queries.
- **Cons:** slow (generation), costly, position bias, output parsing failures, **no calibrated scores** (so no threshold-based abstention).

### D5. Pairwise prompting (PRP)
- Asks "which of A or B is more relevant?" for many pairs and aggregates (all-pairs, or sorting-based with O(K log K) comparisons). Very accurate, rarely used in production because of cost.

### D6. Reasoning rerankers (2025–26)
- A reasoning LLM thinks step by step about relevance before outputting a score or ranking (e.g. Rank1, ReasonRank and similar research models; or prompting a reasoning model). Best on BRIGHT-style tasks ("find the theorem that solves this problem"). Seconds per query, so cascade them on the top ~20.

### D7. Late-interaction rerankers (ColBERT)
- Document token vectors are **precomputed**; at query time only the query is encoded, and scoring is a **MaxSim**:
  `score(q,d) = Σ_{i∈q tokens} max_{j∈d tokens} (q_i · d_j)`
- **Worked example:** query tokens q1 ("reset"), q2 ("mfa"); doc tokens d1 ("re-enroll"), d2 ("authenticator"), d3 ("settings").
  ```
            d1    d2    d3
  q1      0.82  0.31  0.40   → max 0.82
  q2      0.35  0.77  0.20   → max 0.77
                              score = 0.82 + 0.77 = 1.59
  ```
- **Models:** ColBERTv2 (BERT-base, 128-dim token vectors with residual compression to ~2 bits/dim), `answerdotai/answerai-colbert-small-v1` (~33M), Jina-ColBERT-v2 (multilingual, 8k), GTE-ModernColBERT (via PyLate).
- **Pros:** ~10–30 ms to rerank 100 documents; good out of domain. **Cons:** storage of per-token vectors; slightly below cross-encoders.

### D8. Managed reranker APIs
- **Cohere Rerank** (3.5 / 4): multilingual, ~4k-token context per document (3.5), semi-structured input (JSON/YAML/tables). **Voyage rerank-2 / 2.5**: long context, instruction following. **Jina reranker API**. **Google Vertex ranking API**, **Azure AI Search semantic ranker**, **Amazon Bedrock rerank** (Cohere/Amazon models).
- Architectures are proprietary. Treat them as cross-encoders behind an API: you pay per search/document, with no GPUs to run yourself.

## E. Reranker model reference *(verify specifics on each model card; versions move fast)*

| Model | Family | Params | Max tokens | Languages | License | Notes |
|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | Encoder CE | ~22M | 512 | EN | Apache-2.0 | CPU-friendly baseline |
| bge-reranker-base / large | Encoder CE | 278M / 560M | 512 | EN/ZH | MIT | Older but solid |
| **bge-reranker-v2-m3** | Encoder CE (XLM-R) | ~568M | 8192 | 100+ | Apache-2.0 | Default self-hosted multilingual |
| bge-reranker-v2-gemma | LLM pointwise | ~2.5B | 8192 | Multi | Gemma terms | Higher quality, slower |
| jina-reranker-v2-base-multilingual | Encoder CE | ~278M | 1024 | 100+ | CC-BY-NC | Fast; function-calling/code aware |
| mxbai-rerank-base/large-v2 | LLM pointwise (Qwen2.5) | 0.5B / 1.5B | long | 100+ | Apache-2.0 | Strong open-source option |
| **Qwen3-Reranker 0.6B/4B/8B** | LLM pointwise (yes/no) | 0.6–8B | 32k | 100+ | Apache-2.0 | Instruction-aware; top open quality |
| monoT5-base/3B | Seq2seq | 220M–3B | 512 | EN (mT5 multi) | Apache-2.0 | Historical strong baseline |
| RankZephyr-7B | Listwise LLM | 7B | ~4k | EN | Open | Listwise, slower |
| ColBERTv2 / answerai-colbert-small | Late interaction | 110M / 33M | 512 | EN | MIT/Apache | Very fast rerank |
| Cohere Rerank 3.5/4, Voyage rerank-2.5 | API | n/a | 4k–32k | Multi | Commercial | Managed |

## F. Worked end-to-end reranking example

Query: *"Can I carry over unused PTO into next year?"*. Five hybrid candidates:

| Cand | Snippet | Hybrid rank | CE logit | σ(logit) |
|---|---|---|---|---|
| A | "PTO accrual rates by tenure…" | 1 | −0.4 | 0.401 |
| B | "Up to 5 unused PTO days roll over to the next calendar year…" | 2 | 3.1 | 0.957 |
| C | "Carry-on luggage policy for business travel…" | 3 | −4.2 | 0.015 |
| D | "Unused vacation is forfeited on Dec 31 except where local law requires…" | 4 | 2.2 | 0.900 |
| E | "Holiday calendar 2026…" | 5 | −2.0 | 0.119 |

After reranking: **B (0.957) > D (0.900) > A (0.401) > E (0.119) > C (0.015)**.
- C ("carry-on") was a *lexical false positive* from BM25 ("carry"). The cross-encoder sees it's about luggage.
- With a calibrated abstention threshold τ = 0.5, keep {B, D} → the LLM gets two tightly relevant sources and can explain the 5-day rollover and the local-law exception.
- If max σ < τ for every candidate → abstain or ask a clarifying question.

## G. Fine-tuning a cross-encoder on your domain (recipe)
```python
from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
model = CrossEncoder("BAAI/bge-reranker-v2-m3", num_labels=1)
# dataset rows: {"query": ..., "positive": ..., "negative_1": ..., ..., "negative_7": ...}
# hard negatives: top-30 from your hybrid retriever minus known positives, filtered by an LLM judge
loss = CachedMultipleNegativesRankingLoss(model, num_negatives=7)
trainer = CrossEncoderTrainer(model=model, train_dataset=train_ds, loss=loss)
trainer.train()
```
Evaluate nDCG@10 on the held-out golden set before and after. Watch for out-of-domain regression.
