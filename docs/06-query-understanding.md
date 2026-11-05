# 06 — Query Understanding & Transformation

Users write short, ambiguous, conversational, typo-ridden queries. Documents are long and formal. Query transformation closes that gap **at query time**. Enrichment (Module 03) closes it at index time. A 2026 survey (arXiv 2412.17558, rev. 2026) frames this field as moving from *retrieval-centric* rewriting to *reasoning-centric*, agentic query optimization.

## 1. Techniques

| Technique | What it is | Purpose | Pros | Cons | When |
|---|---|---|---|---|---|
| **Conversational condensing** | LLM rewrites the follow-up plus history into a standalone query ("what about Q3?" → "What was ACME's EMEA revenue in Q3 2025?") | Fix multi-turn drift (F15) | Essential for chat; cheap | Can over-rewrite or inject wrong context | **Always** in multi-turn chat |
| **Spelling / normalization** | Typo fixes, acronym expansion via glossary, unit normalization | Lexical recall | Cheap, deterministic | Can "fix" valid product codes | Consumer traffic |
| **Query rewriting** (Rewrite-Retrieve-Read, Ma et al. 2023) | LLM rewrites into retrieval-friendly form | Better match | Simple | One extra LLM call | Noisy queries |
| **Multi-query / RAG-Fusion** | Generate N paraphrases → retrieve each → fuse with RRF | Recall via diversity | Robust; covers phrasings | N× retrieval cost; drift | Recall-bound systems |
| **HyDE** (Gao et al. 2022) | LLM writes a *hypothetical answer*; embed it to search | Answer-to-answer similarity | Strong zero-shot gains | Hallucinated specifics can mislead; latency; bad for ID or lookup queries | Short conceptual queries with no in-domain training |
| **Query2doc / Query expansion** | Append LLM pseudo-document or keywords to the query | Lexical + semantic expansion | Helps BM25 a lot | Topic drift | Sparse retrieval |
| **Step-back prompting** (Zheng et al. 2023) | Ask a more general question first ("what are the principles of X?") | Retrieve background | Better for reasoning questions | Extra retrieval | Conceptual or why questions |
| **Decomposition / sub-questions** | Break multi-part or multi-hop into sub-queries (Least-to-Most, IRCoT-style) | Completeness (F11) and multi-hop | Big gains on comparisons and aggregations | Latency; error compounding | "Compare A vs B", "how did X change" |
| **Self-query / filter extraction** | LLM extracts structured filters + semantic query | Precision via metadata | Exact constraints (dates, products) | Must validate against schema | Catalogs, policies with dates |
| **Intent classification & routing** | Classify: chit-chat / KB QA / action / SQL analytics / out-of-scope / escalate | Pick the right pipeline or tool | Saves cost; better UX | Classifier errors | Every production bot |
| **Index / source routing** | Choose which collection or tool (HR KB vs. Eng KB vs. SQL vs. web) | Avoid searching everything | Precision, latency | Routing errors are silent → log and evaluate | Multi-domain |
| **Ambiguity detection & clarification** | Detect underspecified queries → ask a clarifying question | Avoid wrong confident answers | Trust | Annoying if overused | High-stakes domains |
| **Adaptive retrieval decision** (Self-RAG, Adaptive-RAG) | Decide whether to retrieve at all, and how much (none / single / iterative) by query complexity | Cost and latency | Skip retrieval for "thanks!" | Misclassification | High-volume bots |
| **Reasoning-model rewriting** | A reasoning LLM produces search plans (used in deep-research agents; helps reasoning-intensive retrieval like BRIGHT) | Hard queries | Big gains on BRIGHT-style tasks | Expensive | Analyst / research assistants |

## 2. Reciprocal Rank Fusion (RRF)
`score(d) = Σ_i 1 / (k + rank_i(d))`, with `k ≈ 60`. Rank-based, so no score normalization is needed. It is robust and the default for fusing multi-query *and* hybrid results. Weighted RRF lets you favor one retriever.

## 3. Production pattern

```
user_msg + history
   → [cheap LLM, structured output, ~200 ms]
     { intent, standalone_query, sub_queries[], filters{}, needs_retrieval, language, clarify? }
   → validate filters against schema enums
   → route: SMALLTALK | RAG(simple) | RAG(decomposed) | SQL_TOOL | AGENT | ESCALATE
```
Combine all of this into **one** structured LLM call to save latency. Use a small fast model. Cache by `(normalized_query, history_hash)`.

## 4. Evaluating query transformation
- Retrieval Recall@k **with vs. without** each transform, *per intent bucket*. Transforms that help multi-hop often hurt lookups.
- Routing accuracy (a confusion matrix over a labeled set).
- Condensing quality: the judge checks that the standalone query preserves intent and entities.
- Latency added at p95.

## Expert checks
- When does HyDE hurt? (Exact identifiers; the model confidently invents wrong specifics; domains the LLM doesn't know.)
- Why is RRF preferred over linear score combination when fusing retrievers?
- How would you detect that your router silently sends 8% of analytics questions to the text KB?
