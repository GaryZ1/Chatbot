# 02 — Chunking: every method, its purpose, pros and cons

## 1. Why chunking matters

A chunk is the **unit of retrieval** and the **unit of context**. It must be:
1. **Small enough** that its embedding stays specific. One vector averages everything in it, and a 4,000-token chunk "means" nothing in particular.
2. **Large enough** to be self-contained: subject, conditions, numbers, and units together.
3. **Aligned with meaning**: no sentences, tables, or code blocks cut in half.

These goals conflict. **Every chunking method is a different trade-off between them**, and the most advanced methods decouple them: *retrieve on small units, read large units*.

### Key evidence (read these)
- **Vectara / Qu et al. 2024**, *"Is Semantic Chunking Worth the Computational Cost?"* (arXiv 2410.13070): semantic chunking did **not** consistently beat fixed-size chunking on real (non-synthetic) documents.
- **Chroma Research 2024**, *"Evaluating Chunking Strategies for Retrieval"*: introduced token-level IoU / precision / recall for chunking. Recursive splitting at ~200–400 tokens with no overlap was a strong baseline, and LLM-guided chunking scored well on recall.
- **Anthropic 2024**, *"Contextual Retrieval"*: prepending an LLM-written context to each chunk cut top-20 retrieval failures by **35%** (embeddings), **49%** (+ contextual BM25), and **67%** (+ reranking).
- **Jina AI 2024**, *"Late Chunking"* (arXiv 2409.04701): embed the whole document with a long-context encoder, *then* pool per chunk. This gives contextual chunk embeddings without LLM calls (reported nDCG@10 gains on BEIR long-document tasks, e.g. 52.2 → 54.0 for 256-token chunks).
- **Merola & Singh 2025**, *"Reconstructing Context"* (arXiv 2504.19754, ECIR workshop): contextual retrieval preserves coherence better; late chunking is more efficient but can lose relevance and completeness.
- **NVIDIA 2024 chunking study:** page-level chunking had the highest average accuracy and lowest variance across their financial and report datasets.
- **2026 reports** *(vendor / unverified)*: a FloTorch Feb 2026 benchmark reports recursive 512-token chunking beating semantic chunking on academic papers; a Redis 2026 benchmark reports ~3% average gain from late chunking on long documents.

**Takeaway:** start with structure-aware recursive chunking of 256–512 tokens. Add **contextualization** (contextual retrieval or late chunking) and **parent-child retrieval** before reaching for exotic splitters.

---

## 2. The methods

Each method below has: *what it is → purpose → how → pros → cons → when to use → key parameters*.

### 2.1 Fixed-size character chunking
- **What:** split every N characters, optionally with an overlap of M.
- **Purpose:** the simplest possible baseline.
- **How:** `text[i : i+N]` with stride `N-M`.
- **Pros:** trivial, deterministic, fast, language-agnostic.
- **Cons:** cuts words, sentences, and tables; character count ≠ token count; ignores meaning.
- **When:** never in production. Use it only as a floor in experiments.
- **Params:** N=1000–2000 chars, overlap 10–20%.

### 2.2 Fixed-size token chunking
- **What:** split on tokenizer token counts (use the *embedding model's* tokenizer).
- **Purpose:** guarantee chunks fit the embedding model's max length.
- **Pros:** predictable cost and size; no truncation surprises.
- **Cons:** same boundary problems as character chunking.
- **When:** a fallback inside recursive splitting; streaming logs.
- **Params:** 256–512 tokens; overlap 10–15%.

### 2.3 Sentence-based chunking
- **What:** split into sentences (spaCy, NLTK, pySBD, or a multilingual segmenter), then group k sentences or pack sentences up to a token limit.
- **Purpose:** never cut mid-sentence.
- **Pros:** linguistically clean; good for QA over prose.
- **Cons:** sentence segmenters fail on bullets, abbreviations, code, and tables; a single sentence often lacks context.
- **When:** narrative prose, news, transcripts (combined with packing).

### 2.4 Paragraph-based chunking
- **What:** split on blank lines or `<p>`; merge short paragraphs, split long ones.
- **Purpose:** authors already group ideas into paragraphs.
- **Pros:** natural semantic units; cheap.
- **Cons:** paragraph length varies enormously (1 line to 1 page); PDFs often lose paragraph breaks.
- **When:** well-formatted HTML/Markdown prose.

### 2.5 Recursive character/token splitting (the industry default)
- **What:** try separators in priority order (`\n\n` → `\n` → `. ` → ` ` → `""`) and recurse only on pieces still above the size limit. LangChain `RecursiveCharacterTextSplitter`.
- **Purpose:** respect the largest natural boundary that still fits.
- **Pros:** robust, fast, strong baseline in every benchmark; language-aware variants exist (Markdown, Python, etc.).
- **Cons:** structure-unaware beyond separators; headings separated from their content; no context injection.
- **When:** **start here**. Benchmark everything else against it.
- **Params:** 256–512 tokens (factoid QA lower, analytical higher); overlap 0–15%. Chroma found that overlap often adds little.

### 2.6 Document-structure / layout-aware chunking
- **What:** use the parsed hierarchy (H1/H2/H3, sections, list items, table blocks, pages) as boundaries. Examples: `MarkdownHeaderTextSplitter`, `HTMLHeaderTextSplitter`, Docling `HierarchicalChunker` / `HybridChunker`, Unstructured `chunk_by_title`.
- **Purpose:** align chunks with how the author organized knowledge, and carry the **heading path** as metadata or a prefix.
- **Pros:** semantically coherent; heading breadcrumbs add context for free; great for citations ("Section 4.2"); tables kept intact.
- **Cons:** depends on parse quality; sections can be huge (needs recursive sub-splitting) or tiny (needs merging).
- **When:** **the production default** for manuals, policies, wikis, contracts, and specs. Combine with 2.5 for oversize sections.
- **Pattern:** `chunk_text = "<Doc title> > <H1> > <H2>\n\n<section text>"`.

### 2.7 Page-level chunking
- **What:** one page = one chunk (optionally split further).
- **Purpose:** PDFs where the page is the natural unit (financial reports, slides, forms).
- **Pros:** NVIDIA's 2024 study found it the most consistent; trivial citations ("p. 14"); pairs naturally with visual retrieval (ColPali).
- **Cons:** content that spans pages gets split; pages vary in density.
- **When:** reports, 10-Ks, slide decks, scanned documents.

### 2.8 Sliding window (overlapping) chunking
- **What:** fixed windows with a large overlap (e.g. 50%).
- **Purpose:** ensure every fact appears whole in at least one chunk.
- **Pros:** reduces boundary misses.
- **Cons:** index size grows by 1/(1−overlap); duplicate retrieval results (need dedup); diminishing returns.
- **When:** short, dense texts where boundaries are unknown (legal clauses, logs).

### 2.9 Semantic chunking (embedding-breakpoint)
- **What:** embed each sentence (or a window of sentences); compute cosine distance between neighbors; split where distance exceeds a percentile, standard-deviation, or gradient threshold. Greg Kamradt's "5 levels", LlamaIndex `SemanticSplitterNodeParser`, LangChain `SemanticChunker`, Chonkie.
- **Purpose:** split where the *topic* changes.
- **Pros:** topic-coherent chunks for unstructured text without headings (transcripts, chats).
- **Cons:** an embedding call per sentence at ingest; very sensitive to thresholds; chunk sizes vary wildly; **evidence shows no consistent gain over recursive** (Qu et al. 2024; 2026 reports).
- **When:** long unstructured text with clear topic shifts *and* only once you've shown it beats recursive on your evaluation.
- **Params:** breakpoint percentile 90–95; buffer size 1–3 sentences; min/max chunk size guards.

### 2.10 LLM-based / agentic chunking
- **What:** an LLM reads the document and decides boundaries (and optionally titles and summaries). Variants: **LumberChunker** (Duarte et al. 2024: an LLM finds the passage where the content shifts) and **Meta-Chunking** (Zhao et al. 2024: perplexity / margin-sampling-based boundary detection with small LMs).
- **Purpose:** human-like segmentation of messy content.
- **Pros:** high-quality boundaries; can generate titles and summaries at the same time.
- **Cons:** expensive at ingest (cost ∝ corpus tokens); non-deterministic; re-ingestion cost; LLM can drop or alter text (always split by *offsets*, never by regenerated text).
- **When:** high-value, low-volume corpora (legal, clinical guidelines) or messy text where structure is absent.

### 2.11 Proposition chunking (Dense X Retrieval)
- **What:** an LLM decomposes text into atomic, self-contained factoids ("propositions"). Chen et al. 2023, *"Dense X Retrieval: What Retrieval Granularity Should We Use?"*
- **Purpose:** maximize retrieval precision for factoid QA.
- **Pros:** each unit is decontextualized ("The Eiffel Tower is 330 m tall" not "It is 330 m"); improves Recall@k for short-answer QA in the paper.
- **Cons:** very costly; loses narrative and reasoning across sentences; index explodes ~5–10×; poor for "explain why" questions.
- **When:** FAQ and fact lookup; as an *additional* index pointing back to parent chunks.

### 2.12 Small-to-big / parent-child (parent document retrieval)
- **What:** index **small** child chunks (sentences or ~128 tokens) for matching, but return their **parent** (section or ~1–2k tokens) to the LLM. LangChain `ParentDocumentRetriever`, LlamaIndex `AutoMergingRetriever` (merges children into the parent when enough siblings hit).
- **Purpose:** decouple *retrieval precision* from *context sufficiency*.
- **Pros:** one of the highest-ROI upgrades; precise matching plus complete context.
- **Cons:** more context tokens; two-level storage; need dedup when several children share a parent.
- **When:** almost always worth testing, especially for manuals and policies.

### 2.13 Sentence-window retrieval
- **What:** index single sentences; at query time return the sentence ± k neighbors.
- **Purpose:** a lightweight small-to-big.
- **Pros:** precise; cheap.
- **Cons:** window may still miss distant context (the heading 3 paragraphs up).
- **When:** dense prose QA.

### 2.14 Hierarchical / multi-granularity / summary-tree chunking (RAPTOR)
- **What:** build a tree. Leaves = chunks; cluster them (GMM over embeddings) and summarize each cluster with an LLM; recurse. Index *all levels*. Sarthi et al. 2024, *"RAPTOR"* (ICLR 2024).
- **Purpose:** answer both detail questions (leaves) and thematic or global questions (summaries).
- **Pros:** big gains on long-document, thematic QA (NarrativeQA, QuALITY in the paper).
- **Cons:** expensive ingest; summaries can hallucinate; hard incremental updates (re-clustering).
- **When:** books, long reports, research corpora with "what are the main themes" questions. Compare with GraphRAG community summaries (Module 09).

### 2.15 Contextual chunking (Anthropic Contextual Retrieval)
- **What:** for each chunk, ask an LLM (with the *full document* in a cached prompt) for 50–100 tokens situating the chunk ("This chunk is from ACME's Q2 2023 10-Q, discussing revenue growth versus Q1…"). Prepend it before embedding **and** before BM25 indexing.
- **Purpose:** fix context-less chunks (failure F4).
- **Pros:** large measured gains (−35% / −49% / −67% failures, see above); works with any chunker and any embedder; prompt caching makes it cheap (Anthropic estimated about $1 per million document tokens in 2024).
- **Cons:** an LLM call per chunk at ingest; documents longer than the LLM context need windowing; re-run when documents change.
- **When:** **default upgrade #1** once you have a baseline. Technically this is *enrichment* (Module 03), but it is inseparable from chunking.

### 2.16 Late chunking
- **What:** run a long-context embedding model (e.g. jina-embeddings-v3/v4, 8k+) over the *whole document* (or a large window) to get token embeddings, then mean-pool the token spans of each chunk.
- **Purpose:** contextual chunk embeddings with **no LLM cost**. Each chunk's vector "knows" the surrounding document.
- **Pros:** cheap; deterministic; resolves anaphora ("it", "the company"); no extra tokens stored.
- **Cons:** needs a long-context *token-level* embedding model with access to token outputs (not every API exposes this); gains are smaller than contextual retrieval in some studies (Merola & Singh 2025); helps dense only, not BM25.
- **When:** self-hosted embedding stacks; very large corpora where LLM-per-chunk is unaffordable.

### 2.17 Code-aware (AST) chunking
- **What:** parse with tree-sitter; chunk = function, class, or method (+ imports, signature, docstring, file path); oversized classes get split by method. Examples: LlamaIndex `CodeSplitter`, Chonkie `CodeChunker`, the cAST paper (2025).
- **Purpose:** never split code mid-block; keep symbol identity.
- **Pros:** much better code retrieval; enables symbol-graph links (call or import edges).
- **Cons:** a grammar per language; cross-file context still needs a repo map.
- **When:** any codebase or code-documentation assistant.

### 2.18 Table-aware chunking
- **What:** tables are atomic chunks; large tables are split by row groups **with the header row repeated**; a natural-language summary plus a column description goes into the embedding text.
- **Purpose:** keep row/column semantics.
- **Pros:** answers numeric questions correctly; avoids garbage "table soup".
- **Cons:** big tables blow the token budget → prefer routing to text-to-SQL or a dataframe tool.

### 2.19 Conversation / thread chunking
- **What:** chunk by message thread or ticket, with speaker roles; or extract "problem → resolution" pairs (LinkedIn's ticket knowledge graph, Module 15).
- **Pros:** retains Q→A linkage.
- **Cons:** noisy; needs cleaning of signatures and quoted replies.

### 2.20 Query-time / dynamic chunking
- **What:** store large units; at query time, select the relevant sub-spans (extractive compression, sentence scoring with a cross-encoder, or a "zoom in" agent tool such as `read_section(doc, start, end)`).
- **Purpose:** avoid committing to one granularity at ingest.
- **Pros:** flexible; works well with agentic RAG (A-RAG 2026: hierarchical retrieval interfaces let the agent pick the granularity).
- **Cons:** latency at query time; more complex.

---

## 3. Comparison matrix

| Method | Ingest cost | Quality (typical) | Boundary quality | Self-contained? | Index size | Incremental updates |
|---|---|---|---|---|---|---|
| Fixed char/token | ★ | ★★ | ✗ | ✗ | 1× | easy |
| Recursive | ★ | ★★★ | ◐ | ◐ | 1× | easy |
| Structure-aware | ★★ | ★★★★ | ✓ | ◐ (+breadcrumbs) | 1× | easy |
| Page-level | ★ | ★★★ (reports) | ◐ | ◐ | 1× | easy |
| Semantic | ★★★ | ★★–★★★ | ◐/✓ | ◐ | 1× | medium |
| LLM / agentic | ★★★★★ | ★★★★ | ✓ | ◐ | 1× | costly |
| Proposition | ★★★★★ | ★★★★ (factoid) | ✓ | ✓ | 5–10× | costly |
| Parent-child | ★★ | ★★★★ | depends | ✓ (parent) | 1.2–2× | medium |
| RAPTOR tree | ★★★★ | ★★★★ (thematic) | ✓ | ✓ | 1.3–1.5× | hard |
| Contextual retrieval | ★★★ | ★★★★★ | depends | ✓ | 1.1× | per-doc rerun |
| Late chunking | ★★ | ★★★★ | depends | ✓ (vector only) | 1× | per-doc rerun |

## 4. Recommended recipe (the 2026 "boringly excellent" default)

1. Parse into typed blocks (Module 01).
2. **Structure-aware split** on headings → recursive sub-split to **≤ 400–512 tokens**, never splitting tables or code blocks; merge pieces under ~50 tokens with their neighbor.
3. Prefix each chunk with a **breadcrumb** (`Title > H1 > H2`).
4. Add **contextual retrieval** text (LLM, prompt-cached) *or* late chunking if self-hosted and cost-bound.
5. Index children for retrieval, return **parent sections** (up to ~1,500 tokens) for generation.
6. Store `chunk_id, doc_id, parent_id, position, heading_path, page, char_start/char_end` (for citations and highlighting).

## 5. How to tune chunking empirically
- **Metrics:** chunk-level Recall@k / nDCG@10 on the golden set; Chroma-style **token IoU** (how much of the retrieved text is relevant); end-to-end answer correctness.
- **Grid:** size ∈ {128, 256, 512, 1024} × overlap ∈ {0, 10%, 20%} × {recursive, structure, semantic} × {±contextual}. Hold the embedder and K fixed, then vary K *in tokens* (not chunks), so small chunks aren't unfairly penalized.
- **Normalize by context budget:** compare at an *equal number of context tokens* delivered to the LLM.
- **Look at the long tail:** run analysis per document type. The best chunker for slides is not the best for contracts. It's fine to have **per-type chunkers**.

## 6. Anti-patterns
- Choosing the chunk size from a blog post instead of your own evaluation.
- Chunks larger than the embedding model's max sequence length (silent truncation!).
- 50% overlap "just in case" → duplicated results crowd out diversity.
- Splitting tables or code blocks.
- Losing the offsets, so you can't cite or highlight.
- Re-chunking without re-embedding or versioning (index inconsistency).

## Expert checks
- Explain mechanistically why one embedding of a 3,000-token chunk retrieves worse than six 500-token chunks.
- Contextual retrieval vs. late chunking: which helps BM25, which costs LLM calls, which needs token-level embeddings?
- Why must chunking comparisons be normalized by context tokens rather than by K?
- When does proposition indexing hurt?
