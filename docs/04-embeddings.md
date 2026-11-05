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

---

# Deep dive: how embedding models are built and used

## A. The anatomy of an embedding model
```
"query: how do i reset mfa"
   │ 1. Tokenizer (WordPiece / BPE / SentencePiece) → token IDs [101, 2129, 2079, ...]
   ▼
   2. Transformer backbone (encoder: BERT/XLM-R/ModernBERT, or decoder LLM: Qwen/Mistral/Gemma)
      → one contextual vector per token: H ∈ ℝ^{L × hidden}   (e.g. L=9, hidden=1024)
   ▼
   3. Pooling: collapse L token vectors into 1 vector
      • CLS pooling      : take h_[CLS]                    (BGE, many BERT models)
      • Mean pooling     : average all token vectors       (E5, MiniLM sentence-transformers, Nomic)
      • Last-token / EOS : take the final token's vector   (decoder LLM embedders: Qwen3-Embedding, E5-mistral)
                           (causal attention → only the last token has "seen" the whole text)
      • Latent-attention : learned attention pooling       (NV-Embed)
   ▼
   4. (Optional) projection / Matryoshka truncation to the target dims
   ▼
   5. L2 normalization → unit vector, so dot product = cosine
```

**Mean pooling worked example:** three token vectors [1, 0, 2], [3, 2, 0], [2, 1, 1] → mean = [2, 1, 1] → norm = √6 = 2.449 → normalized = [0.816, 0.408, 0.408].

## B. How they're trained (contrastive learning)
1. **Stage 1, weakly supervised pre-training:** hundreds of millions to billions of naturally occurring pairs (title↔body, question↔answer from forums, query↔clicked page, caption↔image).
2. **Stage 2, supervised fine-tuning:** high-quality labeled pairs (MS MARCO, NQ, HotpotQA, NLI, plus synthetic LLM-generated data) with **hard negatives**.
3. **Instruction tuning:** prepend task instructions ("Instruct: Given a question, retrieve passages that answer it\nQuery: …").
4. **Optional:** model merging, distillation from cross-encoder or LLM teachers, Matryoshka loss.

**InfoNCE loss with in-batch negatives:**
```
L = − log  exp(sim(q, d⁺)/τ) / [ exp(sim(q, d⁺)/τ) + Σ_j exp(sim(q, d⁻_j)/τ) ]
```
τ (temperature) ≈ 0.01–0.05 sharpens the distribution. Every other document in the batch serves as a negative, so **bigger batches = more negatives = better models** (top models train with batches of thousands, using gradient caching).

**Worked example:** τ = 0.05, sim(q,d⁺) = 0.80, two negatives at 0.60 and 0.50 →
exp(16) = 8.886e6, exp(12) = 1.628e5, exp(10) = 2.203e4 → p = 8.886e6 / 9.071e6 = 0.980 → **L = 0.020**.
If a hard negative were at 0.78: exp(15.6) = 5.956e6 → p = 8.886e6/(8.886e6 + 5.956e6 + 2.203e4) = 0.598 → **L = 0.515**. A hard negative creates a large gradient, which is why hard-negative mining matters so much.

## C. Matryoshka Representation Learning (MRL)
The training loss is summed over several prefix lengths (e.g. 64, 128, 256, 512, 1024 dims), so the **first n dimensions are a usable embedding by themselves**. In production: store 256 dims (4× less memory), and optionally rescore with the full 1024. Always **re-normalize after truncating**.
Example: v = [0.6, 0.6, 0.4, 0.346] (unit length). Truncate to 2 dims → [0.6, 0.6] → norm 0.849 → [0.707, 0.707].

## D. Quantization worked examples
- **float32 → int8 (scalar):** per dimension, map the observed range [min, max] to [−128, 127]. With range [−0.12, 0.15], scale = 0.27/255 = 0.00106. The value 0.03 → round((0.03 + 0.12)/0.00106) − 128 = 141 − 128 = **13**. 4× smaller, typically ~1% retrieval loss.
- **Binary:** keep only the sign: [0.12, −0.03, 0.40, −0.20] → **1 0 1 0**. Similarity = number of matching bits (Hamming). 1024 dims → 128 bytes (vs. 4096). Typical pattern: retrieve the top 200 with Hamming distance, **rescore** with int8/float32 vectors → recovers ~95%+ of quality.
- **Product quantization:** see Module 05.

## E. Symmetric vs. asymmetric, and prefixes
| Model family | Query side | Document side |
|---|---|---|
| E5 | `"query: ..."` | `"passage: ..."` |
| BGE (en v1.5) | `"Represent this sentence for searching relevant passages: ..."` | (none) |
| Qwen3-Embedding / E5-mistral / NV-Embed | `"Instruct: {task}\nQuery: ..."` | (none) |
| Nomic | `"search_query: ..."` | `"search_document: ..."` |
| Cohere / Voyage / Jina APIs | `input_type="search_query"` / `"query"` / `task="retrieval.query"` | `"search_document"` / `"document"` / `"retrieval.passage"` |
| OpenAI text-embedding-3 | none | none |

Forgetting the prefix typically costs a few nDCG points, silently.

## F. Model reference *(as of late 2026; verify on model cards and live leaderboards)*

| Model | Type / backbone | Params | Dims (MRL) | Max tokens | License | Notes |
|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | Encoder, mean pool | 22M | 384 | 256 | Apache-2.0 | Prototyping only |
| bge-base/large-en-v1.5 | Encoder, CLS | 110M / 335M | 768 / 1024 | 512 | MIT | Solid English baselines |
| **BGE-M3** | XLM-R-large | 568M | 1024 + sparse + multi-vector | 8192 | MIT | One model, three retrieval modes, 100+ languages |
| multilingual-e5-large(-instruct) | XLM-R-large | 560M | 1024 | 512 | MIT | Multilingual workhorse |
| e5-mistral-7b-instruct | Mistral-7B, last token | 7B | 4096 | 4k–32k | MIT | Early LLM embedder |
| NV-Embed-v2 | Mistral-7B + latent attention | 7B | 4096 | 32k | Non-commercial | Topped MTEB in 2024 |
| **Qwen3-Embedding-0.6B / 4B / 8B** | Qwen3 decoder, last token | 0.6–8B | up to 1024 / 2560 / 4096 (MRL) | 32k | Apache-2.0 | Instruction-aware; open-weight leader in 2026 |
| gte-Qwen2-7B-instruct | Qwen2 decoder | 7B | 3584 | 32k | Apache-2.0 | |
| jina-embeddings-v3 | XLM-R + task LoRA adapters | 570M | 1024 (MRL to 32) | 8192 | CC-BY-NC | Late-chunking friendly |
| jina-embeddings-v4 | Qwen2.5-VL | ~3.8B | 2048 single + 128 multi-vector | 32k | Non-commercial (check) | Text + images |
| nomic-embed-text-v1.5 / v2-moe | Encoder / MoE | 137M / 475M | 768 (MRL) | 8192 / 512 | Apache-2.0 | Fully open data |
| snowflake-arctic-embed-l-v2.0 | XLM-R-large | 568M | 1024 (MRL) | 8192 | Apache-2.0 | Multilingual retrieval |
| ModernBERT-based embedders (gte-modernbert, etc.) | ModernBERT | 150–400M | 768–1024 | 8192 | Apache-2.0 | Fast long-context encoders |
| OpenAI text-embedding-3-small / large | API | n/a | 1536 / 3072 (`dimensions` param) | 8191 | Commercial | Widely deployed |
| Google gemini-embedding-001 | API | n/a | 3072 (MRL 768/1536) | 2048 | Commercial | Top multilingual API |
| Cohere Embed v4 | API, multimodal | n/a | 256–1536 (MRL), int8/binary | 128k | Commercial | Text + images, long docs |
| Voyage-3-large / voyage-4 family | API | n/a | 256–2048, int8/binary | 32k | Commercial | Retrieval-focused; code/law/finance variants |
| ColBERTv2 / ColPali | Multi-vector | 110M / 3B | 128 per token/patch | 512 / image | MIT / Gemma | See Modules 08, 09 |

**How to read a model card in 60 seconds:** pooling method, required prefix, max length, dims (and MRL support), training data languages, license, retrieval nDCG on the closest public task to yours, and whether the inference library is supported (sentence-transformers, TEI, vLLM, Infinity).

## G. Serving
- **Hugging Face Text Embeddings Inference (TEI)**, **Infinity**, **vLLM** (`--task embed`), Triton/ONNX. Use fp16/bf16, dynamic batching, and sort inputs by length to cut padding.
- Throughput rule of thumb: small encoders do thousands of chunks/s per GPU; 7–8B LLM embedders do tens to hundreds/s. Budget re-embedding time accordingly (10M chunks at 200/s ≈ 14 GPU-hours).
