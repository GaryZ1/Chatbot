# 09 — Advanced RAG: Graph, Agentic, Self-correcting, Multimodal

> Full catalog with runnable code for every method: [Module 20](20-advanced-rag-catalog.md).

Use these **only when the evaluation shows** that a well-tuned hybrid + rerank baseline fails on a specific query class (multi-hop, global or thematic, multi-source, visual).

## 1. Taxonomy of RAG architectures

| Architecture | Core idea | Solves | Cost |
|---|---|---|---|
| Naive RAG | Embed → top-k → generate | Simple lookup | ★ |
| Advanced RAG | + query transform, hybrid, rerank, context engineering | Most production needs | ★★ |
| Modular RAG | Composable pipelines with routing (Gao et al. survey, 2023–24) | Heterogeneous traffic | ★★ |
| Self-correcting RAG (Self-RAG, CRAG) | Model critiques retrieval and its own output; re-retrieves or falls back | Low-quality retrieval | ★★★ |
| GraphRAG family | Knowledge graph or community summaries | Multi-hop, global "sensemaking" | ★★★★ (ingest) |
| Agentic RAG | LLM agent plans, calls search tools iteratively | Multi-hop, multi-source, ambiguous | ★★★★ (query) |
| RL-trained search agents (Search-R1, ReSearch, DeepResearcher) | Policy trained with RL to interleave reasoning and search | Hard multi-hop | ★★★★★ (training) |
| Multimodal / visual RAG (ColPali) | Retrieve page images; VLM answers | Charts, slides, scans | ★★★ |

## 2. Self-correcting patterns
- **Self-RAG** (Asai et al. 2023): the model is trained to emit reflection tokens: *Retrieve?*, *IsRelevant*, *IsSupported*, *IsUseful*. It retrieves adaptively and critiques its output. **Pro:** adaptive and grounded. **Con:** needs a specially trained model. In practice, people imitate it with prompted judges.
- **CRAG – Corrective RAG** (Yan et al. 2024): a lightweight evaluator grades retrieved documents → *Correct* (refine: decompose-then-recompose knowledge strips), *Incorrect* (web search fallback), *Ambiguous* (both). **Pro:** plug-in robustness. **Con:** extra latency; needs a fallback source.
- **Adaptive-RAG** (Jeong et al. 2024): a classifier routes by complexity → no retrieval / single-step / multi-step.
- **Practical implementation:** a LangGraph state machine: `retrieve → grade_docs → (rewrite_query ↺ | generate) → check_grounded → (regenerate | answer)`, with a **max-iterations cap**.

## 3. GraphRAG family

| Method | Idea | Pros | Cons |
|---|---|---|---|
| **Microsoft GraphRAG** (Edge et al. 2024) | LLM extracts entities and relations → Leiden communities → hierarchical community summaries; *local search* (entity neighborhood) and *global search* (map-reduce over summaries); DRIFT search combines both | Answers global "what are the main themes across 10k docs" questions that vector RAG can't | Very expensive indexing; hard incremental updates; extraction errors |
| **LazyGraphRAG** (Microsoft 2024–25) | Defers LLM summarization to query time; cheap NLP noun-phrase graph | ~0.1% of GraphRAG indexing cost (per Microsoft) | Query-time cost |
| **LightRAG** (Guo et al. 2024) | Dual-level (low: entities; high: themes) retrieval over a graph + vectors; incremental updates | Cheaper than GraphRAG; incremental | Default configs underperform on some enterprise tasks (Unique.ai 2026 report) |
| **HippoRAG / HippoRAG 2** (Gutiérrez et al. 2024, 2025) | Hippocampus-inspired: open-IE KG + **Personalized PageRank** seeded by query entities | Efficient single-step multi-hop; matches iterative retrieval at lower cost | OpenIE quality; graph upkeep |
| **KG-from-tickets** (LinkedIn, SIGIR 2024) | Tree-structured per-ticket graph + inter-ticket links | +77.6% MRR, −28.6% median resolution time in production | Domain-specific design |
| **Text-to-Cypher / structured KG QA** | LLM writes graph queries over a curated KG (Neo4j) | Exact relational answers | Needs a curated schema |
| **LogicRAG / query-time graphs** (2025) | Build the reasoning structure per query, no pre-built graph | No offline cost | Query latency |

**2026 evidence check:** *"Do We Still Need GraphRAG?"* (arXiv 2604.09666, Apr 2026, RAGSearch benchmark) finds that **agentic search substantially improves dense RAG and narrows the gap to GraphRAG**, especially with RL-trained agents. GraphRAG keeps an edge on complex multi-hop reasoning *when its offline cost is amortized*. Unique.ai (2026) reports pausing its GraphRAG integration in favor of optimizing agentic search. **Expert stance:** try agentic search over hybrid retrieval first. Use GraphRAG for genuinely relational or global-sensemaking workloads over a stable corpus.

## 4. Agentic RAG
- **Tools:** `search(query, filters)`, `read(doc_id, section)`, `list_sources()`, `sql(query)`, `web_search()`, `calculator`. The **hierarchical interfaces** of A-RAG (arXiv 2602.03442, 2026) let the agent choose granularity (corpus → doc → section → chunk) instead of a fixed retrieval algorithm.
- **Loop:** ReAct (reason → act → observe) or plan-and-execute; parallel tool calls for sub-questions; a *reflection* step ("is the evidence sufficient?").
- **Deep research mode:** a planner → parallel researcher sub-agents → synthesizer with citations (the OpenAI/Gemini/Anthropic "Research" style). Minutes per query.
- **RL-trained search agents:** Search-R1, R1-Searcher, ReSearch, ZeroSearch, DeepResearcher (2025) train LLMs with outcome rewards to decide when and what to search. Surveys: arXiv 2508.05668 (deep search agents) and RL-based agentic search surveys (2025–26).
- **Guardrails for agents:** step and token budgets, tool allow-lists per user, loop detection, timeouts, cost ceilings, and every tool call logged.
- **Pros:** handles multi-hop and multi-source questions and ambiguity; can verify. **Cons:** 3–30× latency and cost; harder to evaluate (trajectory evaluation); non-determinism.
- **Routing is everything:** send only complex queries (often 5–15% of traffic) to the agent.

## 5. Multimodal / visual document RAG
- **ColPali / ColQwen2.5 / ColNomic / Jina v4 multi-vector** (Faysse et al. 2024, *"ColPali"*): embed page *images* with a VLM into patch-level multi-vectors; MaxSim retrieval; benchmark **ViDoRe** (v1–v3).
- **Pipeline:** page image → multi-vector index → retrieve top pages → a VLM (Claude/GPT/Gemini/Qwen-VL) answers from the images (+ optional OCR text).
- **Pros:** no parse loss; charts and layout understood. **Cons:** storage (~1k vectors per page; mitigate with token pooling, binarization, MUVERA); VLM answering cost; weak for exact-text lookup → hybrid with the text index.
- **Alternatives:** caption figures into text (Module 03), or single-vector multimodal embedders (Cohere v4, Voyage multimodal).

## 6. Other noteworthy patterns
- **Cache-Augmented Generation (CAG):** preload a small, static knowledge base into the KV cache; no retrieval. Great for < ~100k tokens of FAQs.
- **Speculative RAG** (Wang et al. 2024): a small drafter model writes multiple drafts from document subsets; a large model verifies.
- **RAG-Fusion, Branched RAG, Multi-index routing:** see Module 06.
- **Memory-augmented RAG:** user or episodic memory as another retrievable store (Module 11).
- **Text-to-SQL (structured RAG):** for analytics questions, generate SQL against a semantic layer; validate and limit; return a table plus an explanation.

## Expert checks
- Explain GraphRAG global search (map-reduce over community summaries) and why vector RAG cannot answer "what are the top themes?".
- What did the 2026 RAGSearch benchmark imply about GraphRAG vs. agentic dense RAG?
- Design a CRAG-style loop and bound its worst-case latency.
- When is ColPali strictly better than OCR + text embedding?

---

# Deep dive: algorithms step by step

## A. Microsoft GraphRAG indexing pipeline
1. **Chunk** documents (e.g. 300–1,200 tokens).
2. **Extract** entities (name, type, description) and relationships (source, target, description, strength) with an LLM per chunk; optionally "gleaning" rounds to catch misses.
3. **Merge** duplicates across chunks; summarize each entity's and relationship's descriptions.
4. **Community detection** with hierarchical **Leiden** on the entity graph → levels C0 (coarse) … Cn (fine).
5. **Community reports:** an LLM writes a summary per community (key entities, claims, findings).
6. **Query:**
   - *Local search:* find the entities matching the query → gather their neighbors, relationships, source chunks, and community reports → answer.
   - *Global search (map-reduce):* every community report at a level answers the question in parallel with a helpfulness score (**map**) → the top partial answers are combined (**reduce**). This is how it answers "What are the main themes across the corpus?"
   - *DRIFT:* starts global, then follows up with local searches.

## B. HippoRAG retrieval: Personalized PageRank
1. Offline: OpenIE triples from every passage → a KG (nodes = noun phrases, edges = relations + synonym edges from embedding similarity).
2. Query: extract the query's named entities → link them to KG nodes → these are the **seed** nodes.
3. Run **Personalized PageRank**: `r = (1 − d)·s + d·Mᵀ r`, where `s` puts probability only on the seed nodes and d ≈ 0.5–0.85. Probability mass spreads along edges to nodes *connected to the query's entities*.
4. Passage score = sum of the PPR scores of the nodes it contains → rank passages.
Why it helps multi-hop: "Which Stanford professor works on Alzheimer's?" seeds {Stanford, Alzheimer's}. A professor node connected to *both* gets mass from both seeds in a single step, with no iterative retrieval needed.

## C. CRAG control flow
```
docs = retrieve(q)
grades = evaluator(q, docs)          # per-doc relevance score in [−1, 1]
if max(grades) > upper:   action = CORRECT    → knowledge = refine(docs)          # split into strips, keep relevant ones
elif max(grades) < lower: action = INCORRECT  → knowledge = web_search(rewrite(q))
else:                     action = AMBIGUOUS  → knowledge = refine(docs) + web_search(rewrite(q))
answer = generate(q, knowledge)
```
In the enterprise: replace "web search" with *broader internal search*, *ask a clarifying question*, or *abstain*.

## D. Agentic RAG loop (ReAct with budget)
```python
state = {"q": q, "evidence": [], "steps": 0}
while state["steps"] < MAX_STEPS and tokens_used < BUDGET:
    thought, action = llm.plan(state)                 # e.g. search("2025 travel policy intl flights")
    if action.name == "final_answer": break
    obs = tools[action.name](**action.args)           # ACL-filtered search / read_section / sql
    state["evidence"].append(obs); state["steps"] += 1
    if llm.sufficient(state): break                   # reflection
return llm.answer_with_citations(state)
```
Evaluation: final-answer correctness **plus** trajectory metrics (number of steps, redundant searches, tool-error rate, cost).

## E. ColPali scoring
Each page image → a ~1,030 patch embeddings (128-d). Each query token → a 128-d vector. Score = **MaxSim** (as in ColBERT): for every query token, the max similarity over all page patches, summed. Storage per page ≈ 1,030 × 128 × 2 bytes ≈ 264 KB in fp16 → 1M pages ≈ 264 GB, hence binarization, token pooling (merge similar patches), or MUVERA.
