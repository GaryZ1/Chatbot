# 04 — Embedding Models

## 1. Fundamentals
- **Bi-encoder:** query and document are encoded *independently* into vectors; relevance = cosine / dot product. Fast (precompute documents), but there is no query-document interaction → less precise than cross-encoders.
- **Training:** contrastive learning (InfoNCE) with in-batch and **hard negatives**, then instruction tuning. Modern top models are **LLM-based** (decoder backbones such as Qwen3 or Mistral, with last-token or mean pooling).
- **Asymmetric retrieval:** queries are short, documents long. Use the model's **query instruction / prefix** (`"query: "`, `"Represent this question for retrieving..."`, `input_type="query"`). Forgetting it is a classic silent bug.

## 2. Embedding types

| Type | Examples | Pros | Cons | Use |
|---|---|---|---|---|
| **Dense single-vector** | OpenAI text-embedding-3, Voyage-3/4, Gemini Embedding, Cohere Embed v4, Qwen3-Embedding, BGE, E5, Nomic, Arctic Embed, Jina v3/v4 | Compact; semantic matching; universal DB support | Misses exact terms, IDs, rare entities; one vector compresses everything | Default |
| **Sparse lexical** | BM25 / BM25F | Exact-term, ID, and acronym matching; no training; explainable | No synonyms; language-specific analyzers | Always, in hybrid |
| **Learned sparse** | SPLADE-v3, BGE-M3 sparse, Elastic ELSER, OpenSearch neural sparse | Term-level precision + learned expansion; inverted index | Index size; fewer DB integrations | High-precision hybrid |
| **Multi-vector / late interaction** | ColBERTv2, Jina-ColBERT-v2, answerai-colbert, GTE-ModernColBERT, ColPali / ColQwen (images) | Token-level MaxSim → near cross-encoder quality; strong out of domain | 10–100× storage (mitigated by PLAID, quantization, token pooling, MUVERA); complex serving | Reranking stage or first stage at moderate scale |
| **Multimodal** | Cohere Embed v4, Voyage multimodal-3, Jina v4, Gemini Embedding 2, ColPali | Text + images in one space | Less mature evaluation | Slides, screenshots, product images |
| **Code** | Voyage-code-3, Codestral Embed, Nomic Embed Code, Qwen3-Embedding | Code semantics | Domain-specific | Code assistants |

## 3. Model landscape, late 2026 *(verify on live leaderboards; many figures are vendor-reported)*
- **Open-weight leaders:** **Qwen3-Embedding** family (0.6B / 4B / 8B; the 8B was reported #1 on MTEB multilingual at ~70.6 in early 2026); **NVIDIA NV-Embed / Nemotron embed** (a Nemotron 8B embedder reported #1 on **RTEB** in July 2026); **BGE-M3** (dense + sparse + multi-vector in one model; the multilingual workhorse); **Jina v3/v4** (long context, late-chunking-friendly, Matryoshka); **Snowflake Arctic Embed 2**, **Nomic Embed v2 (MoE)**, **mxbai**, **E5-mistral**, **gte-Qwen2/3**.
- **API leaders:** **Google gemini-embedding** (~68 MTEB multilingual); **Voyage-3-large / Voyage-4** (strong retrieval-focused, Matryoshka + int8/binary); **Cohere Embed v4** (multimodal, 128k context, multilingual); **OpenAI text-embedding-3-large** (solid, older, widely deployed).
- **Benchmarks:** **MTEB / MMTEB** (broad, *contaminated*: many models train on the in-domain training sets), **RTEB** (2025+, retrieval with *private* held-out datasets to fight overfitting), **BEIR** (zero-shot retrieval), **BRIGHT** (reasoning-intensive retrieval; standard embedders score poorly → motivates reasoning rerankers and query rewriting), **LoCo / LongEmbed** (long documents), **MIRACL / MLDR** (multilingual), **ViDoRe** (visual documents), **CoIR** (code).

**Leaderboards lie (a little):** a top MTEB average can hide weak performance on *your* domain and language. Always run a **domain bake-off** (section 6).

## 4. Key properties to compare

| Property | Why it matters |
|---|---|
| Retrieval nDCG@10 on *your* data | The only number that really matters |
| Max sequence length | Chunks above it get truncated silently |
| Dimensions & **Matryoshka (MRL)** | MRL lets you truncate 3072 → 256 dims with a small loss → 12× less memory |
| Quantization support | float32 → int8 (4× smaller, ~1% loss) → **binary** (32× smaller; rescore with float vectors) |
| Multilingual / cross-lingual | Query in German, document in English |
| Latency & throughput | Query-time embedding latency adds to p95 |
| Cost | $/M tokens (API) or GPU-hours (self-hosted); re-embedding the corpus on every upgrade |
| License & data residency | Apache/MIT vs. restrictive; on-prem needs |
| Instruction support | Task-specific prefixes can add several points |

## 5. Storage math
`N_chunks × dims × bytes`. Example: 10M chunks × 1024 dims × 4 B = **41 GB** float32 → 10 GB int8 → **1.3 GB binary**. Production pattern: **binary or int8 in RAM for the ANN search → rescore the top 100–500 with float vectors on disk**. This recovers ~95–99% of quality.

## 6. Domain bake-off procedure
1. Build 200–1,000 (query, relevant chunk IDs) pairs from real logs or synthetic questions (Module 12), stratified by query type.
2. Fix the chunking. Embed the corpus with 4–6 candidates (2 API, 2–3 open source, the current incumbent).
3. Report Recall@5/@20/@100, nDCG@10, MRR, p95 latency, $/1M tokens, and index size.
4. Evaluate **in the hybrid setting** (with BM25 + reranker). A weaker embedder plus a reranker often ties a stronger one.
5. Choose by quality-per-dollar at your latency SLO.

## 7. Fine-tuning embeddings
- **When:** specialized jargon (legal, medical, internal product names), a measurable gap versus BM25, and more than ~1k training pairs available (synthetic is fine).
- **How:** sentence-transformers v3+ with `MultipleNegativesRankingLoss` (InfoNCE) + **hard negatives** mined with the base model (skip the top 1–3 to avoid false negatives; or filter with a cross-encoder); `MatryoshkaLoss` to keep truncation support; 1–3 epochs; LoRA for large LLM-based embedders.
- **Data:** LLM-generated queries per chunk (like doc2query), filtered by a reranker or judge for answerability.
- **Expected gain:** commonly +5–15% Recall@10 in-domain. Check for regression on out-of-domain queries.
- **Alternatives:** an adapter or linear transform on top of a frozen API embedding (cheap; no re-embedding of the corpus when used on the query side only).

## 8. Operating embeddings in production
- **Version every vector** (`embedding_model`, `version`, `dims`). Never mix models in one index.
- **Migrations:** dual-write a new index → backfill → shadow-evaluate → switch alias → delete the old one. Budget the re-embedding cost.
- **Normalization:** L2-normalize if the model expects cosine; check the DB metric matches.
- **Batching:** batch documents (32–256); cache query embeddings (LRU / Redis) keyed by normalized text.
- **Monitoring:** embedding-distribution drift (centroid shift), increasing query-to-top-1 distance → coverage gaps.

## Expert checks
- Why do bi-encoders lose to cross-encoders, and how does late interaction recover most of the gap?
- What is Matryoshka representation learning and why does it make binary + rescoring practical?
- Why can MTEB rank mislead you, and what does RTEB change?
- How do you mine hard negatives without poisoning training with false negatives?
