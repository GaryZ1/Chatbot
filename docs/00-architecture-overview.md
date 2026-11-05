# 00 — Architecture Overview

## 1. Why RAG at all?

Before building a retrieval system, decide which way of giving the model knowledge you actually need:

| Approach | What it is | Pros | Cons | Use when |
|---|---|---|---|---|
| **Prompt-only** | The LLM answers from its parametric knowledge | Zero infra | Hallucinates, stale, can't see private data | General chit-chat, public knowledge |
| **Long-context stuffing** | Put the whole corpus in the prompt | Simple, no retrieval errors | Cost scales linearly with corpus size; latency; "context rot" (accuracy drops as input grows, see Chroma 2025); no ACLs | Corpus < ~100–200k tokens, low QPS, one document per session |
| **RAG** | Retrieve the relevant pieces, then generate | Fresh, citable, ACL-aware, cheap per query | Retrieval errors, many moving parts | Large or changing private corpora (the default enterprise case) |
| **Fine-tuning** | Bake knowledge or behavior into the weights | Style, format, domain language | Poor at injecting *facts*, costly to refresh, no citations | Tone, output format, domain jargon, tool-use behavior |
| **Agentic RAG** | The LLM plans, calls retrieval and other tools iteratively | Multi-hop, heterogeneous sources | Latency, cost, harder evaluation | Complex research questions, multi-system queries |
| **Hybrid (2026 default)** | Retrieve a *bounded but generous* context (10–50k tokens), then reason with a long-context model; escalate to an agent loop only when needed | Best quality per dollar | Needs routing logic | Most production assistants |

**Rule of thumb:** RAG for *knowledge*, fine-tuning for *behavior*, agents for *multi-step reasoning*, long context for *one large artifact*.

## 2. Reference architecture

```
                          ┌───────────────────── OFFLINE / INDEXING ─────────────────────┐
 Connectors (SharePoint,  │ 1 Ingest → 2 Parse/OCR → 3 Clean/Dedupe → 4 Chunk            │
 Confluence, Drive, S3,   │ → 5 Enrich (context header, summary, Qs, entities, ACL)      │
 DBs, tickets, web)  ───► │ → 6 Embed (dense + sparse [+ multi-vector])                  │
                          │ → 7 Index (vector + BM25 + metadata + KG) with versioning    │
                          └──────────────────────────────────────────────────────────────┘
                          ┌────────────────────── ONLINE / SERVING ──────────────────────┐
 User ──► API gateway ──► │ a AuthN/AuthZ → b Input guardrails (injection, PII, policy)  │
                          │ c Conversation condense → d Query understanding/routing      │
                          │ e Retrieval (hybrid + ACL filter) → f Fusion → g Rerank      │
                          │ h Context assembly (dedupe, order, budget, citations)        │
                          │ i LLM generation (stream) → j Output guardrails (grounding)  │
                          │ k Response + citations → l Logging/tracing → m Feedback      │
                          └──────────────────────────────────────────────────────────────┘
                          ┌────────────────────── EVALUATION / OPS ──────────────────────┐
                          │ Golden sets · component metrics · LLM judges · A/B · drift   │
                          │ Cost/latency dashboards · index freshness · red-teaming      │
                          └──────────────────────────────────────────────────────────────┘
```

## 3. Failure taxonomy (memorize this)

Adapted from "Seven Failure Points When Engineering a RAG System" (Barnett et al., 2024) and extended with what production teams report:

| # | Failure | Stage | Symptom | Typical fix |
|---|---|---|---|---|
| F1 | **Missing content** | Ingestion | Answer isn't in the corpus; the model hallucinates | "I don't know" policy; coverage analysis |
| F2 | **Parse loss** | Parsing | Tables mangled, headers lost, OCR garbage | Better parser (Docling/MinerU/VLM), layout-aware parsing |
| F3 | **Bad chunk boundaries** | Chunking | The right document is retrieved but the key sentence is split | Structure-aware chunking, overlap, parent-child |
| F4 | **Context-less chunks** | Enrichment | "It increased 3%" has no subject | Contextual retrieval, header injection |
| F5 | **Missed top-K** | Retrieval | The relevant chunk is ranked 30th | Hybrid search, better embeddings, fine-tuning, reranking |
| F6 | **Vocabulary mismatch** | Retrieval | Acronyms, part numbers, product codes | BM25 / learned sparse, query expansion |
| F7 | **Not in context** | Assembly | Retrieved but cut by the token budget | Reranking, compression, better K |
| F8 | **Not extracted** | Generation | In context but ignored (lost-in-the-middle, distractors) | Reorder, shorten context, a stronger model |
| F9 | **Wrong format** | Generation | Ignores the table/list/JSON format | Structured output, few-shot examples |
| F10 | **Incorrect specificity** | Generation | Too generic or too detailed | Prompt specification, answer-style routing |
| F11 | **Incomplete** | Generation | Only part of a multi-part question answered | Query decomposition, agentic loop |
| F12 | **Stale / conflicting** | Lifecycle | An old policy version is cited | Versioning, recency boost, deprecation |
| F13 | **Leakage** | Security | User sees documents they shouldn't | ACL filtering *before* scoring |
| F14 | **Injection** | Security | A retrieved document hijacks instructions | Content isolation, instruction hierarchy, filters |
| F15 | **Conversation drift** | Multi-turn | Follow-up "what about Q3?" loses its referent | Query condensing, memory |

**Expert habit:** for every bad answer, label which F-code caused it. Within a week you'll know where to spend effort. It is almost never "use a bigger LLM".

## 4. Design decisions checklist (decide explicitly; write it down)

1. **Corpus profile:** size (docs and tokens), formats, update frequency, languages, share of tables and images.
2. **Query profile:** factoid vs. analytical vs. procedural vs. multi-hop vs. conversational; expected QPS; latency SLO (p50/p95).
3. **Security model:** per-document ACLs? Tenant isolation? Data residency?
4. **Answer contract:** Citations required? Abstain when unsure? Language? Length?
5. **Budget:** $/query ceiling; GPU or self-hosting constraints.
6. **Success metric:** deflection rate, resolution time, CSAT, accuracy on the golden set, or citation precision.

## 5. Build-vs-buy landscape (2026)

| Layer | Managed / buy | Open source / build |
|---|---|---|
| End-to-end RAG | Vertex AI Search, Azure AI Search + Foundry agents, Amazon Bedrock Knowledge Bases / Q Business, Glean, Vectara | RAGFlow, Onyx (formerly Danswer), Kotaemon, R2R, Haystack, LlamaIndex, LangGraph |
| Parsing | Azure Document Intelligence, AWS Textract, LlamaParse, Reducto, Unstructured API | Docling, MinerU, Marker, PaddleOCR-VL, olmOCR, Unstructured OSS |
| Embeddings | OpenAI, Voyage, Cohere, Google Gemini, Jina | Qwen3-Embedding, BGE-M3, Nomic, Snowflake Arctic Embed, NV-Embed |
| Vector DB | Pinecone, Weaviate Cloud, Zilliz, Turbopuffer, Vespa Cloud, Elastic, OpenSearch | Qdrant, Milvus, Weaviate, pgvector/pgvectorscale, LanceDB, Chroma, Vespa |
| Rerank | Cohere Rerank, Voyage rerank, Jina reranker | BGE-reranker, Qwen3-Reranker, mxbai-rerank, ColBERT (PyLate/RAGatouille) |
| Evaluation | LangSmith, Braintrust, Arize Phoenix, Galileo, Patronus | RAGAS, DeepEval, ARES, TruLens, promptfoo, Inspect |

## Expert checks
- Name five failure points that a bigger LLM does **not** fix.
- When is long-context stuffing strictly better than RAG? When is it strictly worse?
- Why must ACL filtering happen before similarity scoring and top-K, not after?

---

# Deep dive: one request, end to end (with timings)

User (California, group `all-employees-us`) asks: **"Can I carry over unused PTO?"**

| # | Stage | What happens | Output | ~ms |
|---|---|---|---|---|
| 1 | Auth | JWT → user_id, groups | `groups=[all-employees-us]` | 5 |
| 2 | Input guard | Injection/PII classifier | pass | 40 |
| 3 | Understand | One small-LLM call | `intent=kb_qa, standalone="PTO carry-over rules", filters={doc_type: policy}` | 250 |
| 4 | Embed query | Qwen3-Embedding / API | 1024-d vector | 20 |
| 5 | Hybrid retrieve | BM25 top-100 ‖ dense top-100, ACL + `is_latest` filter | 2 lists | 60 |
| 6 | Fuse | RRF k=60 | 150 candidates | 1 |
| 7 | Rerank | bge-reranker-v2-m3 on 150 | top 8, max σ = 0.96 | 180 |
| 8 | Expand / assemble | Child→parent, dedupe, order, 12k-token budget | 3 sources | 5 |
| 9 | Generate | Mid-tier LLM, streamed, cached prefix | answer + [1][2] | 600 TTFT |
| 10 | Output guard | Claim-level grounding + PII + URL check | pass | 150 (async on the stream) |
| 11 | Log | Trace with all IDs, scores, versions | — | async |
Total time to first token ≈ **1.2 s**. A thumbs-down on this trace would go into the error-review queue with every intermediate step attached.
