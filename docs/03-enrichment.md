# 03 — Enrichment & Metadata

Enrichment adds information to chunks *at index time*, so retrieval and filtering work better *at query time*. It is the cheapest place to buy quality: you pay once per chunk, not once per query.

## 1. Enrichment techniques

| Technique | What it is | Purpose | Pros | Cons | When |
|---|---|---|---|---|---|
| **Breadcrumb / header injection** | Prepend `Doc title > Section > Subsection` to the chunk text | Free context | Zero cost, big gains on structured documents | Needs good parse | Always |
| **Contextual retrieval** (Anthropic 2024) | LLM writes 50–100 tokens situating the chunk within the full document | Fix pronouns and missing subjects | −35% to −67% retrieval failures (Anthropic) | LLM call per chunk (cheap with prompt caching) | Default upgrade |
| **Document summary** | LLM summary per document, stored as its own vector and as metadata | Document-level routing, "which doc covers X" | Enables two-stage retrieval (doc → chunk) | Summary may miss details | Large corpora, many similar docs |
| **Chunk summary** | Short abstract per chunk, embedded *instead of* or *alongside* raw text | Denoise long or messy chunks | Better matching for verbose content | Lossy; possible hallucination | Transcripts, emails |
| **Hypothetical questions** (doc2query, "HyPE", reverse HyDE) | LLM generates 3–5 questions each chunk answers; embed the questions → point to the chunk | Close the query-document style gap | Large gains for FAQ-style traffic; query-to-query matching | Index ×3–5; questions bias toward the obvious | Support bots, FAQs |
| **Keyword / keyphrase extraction** | KeyBERT, YAKE, or an LLM; add to a BM25 field | Lexical recall for jargon | Cheap | Noise | Technical domains |
| **Named entity extraction** | Products, people, part numbers, customers, dates | Filters, faceting, KG building | Exact-match filtering ("only SKU-123") | NER errors | Catalogs, tickets |
| **Classification / tagging** | Topic, doc type, audience, product line, region | Metadata filtering and routing | Shrinks the search space | Taxonomy upkeep | Multi-domain corpora |
| **Temporal metadata** | Effective date, expiry, version, fiscal period | Recency boosts, "as of" questions | Fixes stale answers (F12) | Dates are hard to extract reliably | Policies, finance |
| **ACL / tenancy metadata** | allow/deny groups, tenant_id, classification level | Security filtering | Required for enterprise | Must stay synced with the source | Always (enterprise) |
| **Quality signals** | Parse score, authority (official vs. draft), popularity, freshness | Ranking priors | Down-rank drafts and duplicates | Needs signal sources | Large, messy intranets |
| **Synonym / acronym expansion** | Domain glossary ("PTO" ↔ "paid time off") | Vocabulary mismatch (F6) | Cheap, high impact | Glossary maintenance | Enterprise jargon |
| **Table and figure descriptions** | VLM or LLM natural-language description of tables and charts | Make non-text retrievable | Unlocks numeric and visual content | Hallucinated numbers | Reports, decks |
| **Entity linking / KG triples** | Map mentions to canonical IDs; extract (subject, relation, object) | GraphRAG and multi-hop | Enables relationship queries | Expensive; error-prone | See Module 09 |
| **Cross-references** | Links between chunks (cited section, next/previous, parent) | Graph expansion at retrieval | Better completeness | Storage | Contracts, standards |

## 2. Prompt templates

**Contextual retrieval (Anthropic style):**
```
<document>
{{WHOLE_DOCUMENT}}            ← cache this prefix
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document
for the purposes of improving search retrieval of the chunk. Answer only with the
succinct context and nothing else.
```

**Hypothetical questions:**
```
Write 4 diverse questions a {{persona}} might ask that are fully answered by this passage.
Use their vocabulary, not the document's. Include one with an acronym and one paraphrase.
Return JSON: {"questions": [...]}
```

**Structured metadata extraction** (use JSON schema / structured outputs):
```json
{"doc_type": "policy|faq|runbook|contract|spec|other",
 "products": ["..."], "effective_date": "YYYY-MM-DD|null",
 "audience": "employee|customer|partner", "entities": [{"name": "...", "type": "..."}]}
```

## 3. Engineering rules
1. **Separate "embedding text" from "display text".** Embed `breadcrumb + context + chunk`; show and cite the original chunk.
2. **Version the enrichment.** Store `enrichment_version` and model ID; you'll re-run it.
3. **Batch + cache.** Use prompt caching (full document as a cached prefix) and batch APIs (often ~50% cheaper).
4. **Use a cheap model.** Small, fast models (Haiku-class, Flash-class, or a local 7–8B) are usually enough for context and questions. Measure it.
5. **Validate with an evaluation.** Each enrichment must show retrieval lift on the golden set, or it gets removed.
6. **Don't let enrichment invent facts.** It shapes *retrieval*; the LLM answers from the *original* text.

## 4. Metadata filtering at query time
- **Pre-filter** (filter, then ANN search): exact, but can hurt ANN recall when the filter is very selective. Modern engines (Qdrant, Weaviate, Milvus, Vespa, pgvectorscale, Turbopuffer) handle filtered HNSW well.
- **Post-filter** (ANN, then filter): fast, but can return fewer than K results. Never use it for ACLs.
- **Self-query / LLM filter extraction:** an LLM converts "policies updated after 2025 for EMEA" into `{updated_at > 2025-01-01, region: EMEA}` plus a semantic query. Validate fields and values against an enum.

## Expert checks
- Why does embedding hypothetical questions work, in terms of the query/document distribution gap?
- Which enrichments help BM25, which help dense retrieval, and which help both?
- Why should ACLs be enforced as a pre-filter and never as a post-filter?
