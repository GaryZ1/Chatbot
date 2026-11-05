# 18 — Mastery Quiz (Questions)

Answers are in [`19-quiz-answers.md`](19-quiz-answers.md). Questions are numbered the same way in both files.

**Levels:** 🟢 fundamentals · 🟡 practitioner · 🔴 expert (top 0.1%). 🧮 = calculation, 🏗️ = design scenario.
**Scoring:** 🟢 = 1 pt, 🟡 = 2 pts, 🔴 = 3 pts. Above 85% of the maximum with every 🔴 attempted means you've mastered the repo. If you fall below 70% on any module, re-read that module.

---

## Module 00: Architecture
- **Q0.1** 🟢 Name the offline (indexing) stages and the online (serving) stages of a RAG system.
- **Q0.2** 🟢 For what kind of knowledge is fine-tuning the wrong tool, and what is it good for?
- **Q0.3** 🟡 Give three conditions under which long-context stuffing beats RAG.
- **Q0.4** 🟡 Name five failure points (F-codes) that a bigger LLM does **not** fix.
- **Q0.5** 🔴 A user reports a wrong answer. Describe how you'd determine, in under 5 minutes, which stage failed.
- **Q0.6** 🔴 State the "iron law" of RAG and its corollary about the context window.

## Module 01: Ingestion & parsing
- **Q1.1** 🟢 Why is plain PDF text extraction risky for multi-column reports?
- **Q1.2** 🟢 What does OHRBench show?
- **Q1.3** 🟡 Describe the internal stages of a layout-model parser like Docling.
- **Q1.4** 🟡 Compare Markdown, HTML, and row-wise linearization for tables. When would you use each?
- **Q1.5** 🟡 Why shouldn't you embed raw spreadsheet cells? What do you do instead?
- **Q1.6** 🔴 How can a document VLM parser score higher on OmniDocBench yet hurt RAG quality on your corpus?
- **Q1.7** 🧮 Ground truth "Net income 4.2M", parser output "Net lncome 4.2M". Compute the normalized edit distance.
- **Q1.8** 🏗️ Design delete propagation so a document removed from SharePoint disappears from the index within 1 hour.

## Module 02: Chunking
- **Q2.1** 🟢 What are the three properties a good chunk must balance?
- **Q2.2** 🟢 How does recursive character splitting work?
- **Q2.3** 🟡 Explain why one embedding of a 3,000-token chunk retrieves worse than six 500-token chunks.
- **Q2.4** 🟡 How does semantic chunking decide boundaries? What did Qu et al. (2024) find about it?
- **Q2.5** 🟡 Explain parent-child (small-to-big) retrieval and the problem it solves.
- **Q2.6** 🟡 What is proposition chunking, and when does it hurt?
- **Q2.7** 🟡 How does RAPTOR build its tree, and what questions does it help with?
- **Q2.8** 🔴 Contextual retrieval vs. late chunking: which helps BM25? Which needs LLM calls? Which needs token-level embeddings? Which is cheaper at 100M chunks?
- **Q2.9** 🔴 Why must chunking comparisons be normalized by context tokens rather than by K?
- **Q2.10** 🟡 What are the Anthropic contextual-retrieval failure-reduction numbers for (a) contextual embeddings, (b) + contextual BM25, (c) + reranking?
- **Q2.11** 🧮 Semantic chunking distances between consecutive sentences: S1–S2 0.12, S2–S3 0.15, S3–S4 0.62, S4–S5 0.10, S5–S6 0.55. With threshold 0.5, what are the chunks?
- **Q2.12** 🧮 A relevant excerpt is 80 tokens. You retrieve 600 tokens of chunks containing 72 excerpt tokens. Compute token recall, precision, and IoU.
- **Q2.13** 🟡 How should code be chunked, and why?
- **Q2.14** 🏗️ Design a chunking strategy for a corpus of 40% policies (with headings), 30% slide decks, 20% support tickets, and 10% Python code.
- **Q2.15** 🟢 Name three chunking anti-patterns.

## Module 03: Enrichment
- **Q3.1** 🟢 What is breadcrumb injection and why is it "free"?
- **Q3.2** 🟡 Why does indexing hypothetical questions improve retrieval?
- **Q3.3** 🟡 Why separate `embed_text`, `bm25_text`, and `display_text`?
- **Q3.4** 🟡 Pre-filter vs. post-filter: which is safe for ACLs, and why?
- **Q3.5** 🔴 Which enrichments help BM25 only, dense only, or both?
- **Q3.6** 🧮 10,000 docs × 8,000 tokens, 20 chunks per doc. Estimate the input tokens for contextual retrieval without vs. with prompt caching (cache reads at 10%).
- **Q3.7** 🟡 What is self-query, and what must you validate?

## Module 04: Embeddings
- **Q4.1** 🟢 What is a bi-encoder?
- **Q4.2** 🟢 Name four pooling strategies and which model families use them.
- **Q4.3** 🟡 Why do decoder-LLM embedders use last-token pooling?
- **Q4.4** 🟡 Write the InfoNCE loss and explain the role of temperature and in-batch negatives.
- **Q4.5** 🧮 τ = 0.05, sim(q, d⁺) = 0.80, negatives at 0.60 and 0.50. Compute the loss. Then recompute it with one negative at 0.78 instead of 0.60.
- **Q4.6** 🟡 What is Matryoshka Representation Learning? What must you do after truncating?
- **Q4.7** 🧮 Storage for 10M chunks × 1024 dims in float32, int8, and binary.
- **Q4.8** 🟡 Binarize [0.12, −0.03, 0.40, −0.20]. How do you recover quality after binary search?
- **Q4.9** 🟡 Why can MTEB rank mislead you? What does RTEB change?
- **Q4.10** 🟡 Give the query prefix conventions for E5, Nomic, and Qwen3-Embedding. What happens if you forget them?
- **Q4.11** 🔴 How do you mine hard negatives for fine-tuning without introducing false negatives?
- **Q4.12** 🟡 Name three model families that offer dense + sparse + multi-vector, or multimodal, embeddings.
- **Q4.13** 🏗️ Describe a zero-downtime embedding-model migration.
- **Q4.14** 🧮 Mean-pool [1,0,2], [3,2,0], [2,1,1], then L2-normalize.

## Module 05: Indexing
- **Q5.1** 🟢 What do HNSW's `M`, `ef_construction`, and `ef_search` control?
- **Q5.2** 🟡 Describe how an HNSW search proceeds through the layers.
- **Q5.3** 🟡 How does IVF work? What does `nprobe` trade off?
- **Q5.4** 🧮 Product quantization: 1024-dim float32, m = 64 sub-vectors, 256 centroids each. Bytes per vector and compression ratio?
- **Q5.5** 🔴 Why can a highly selective filter collapse HNSW recall, and how do engines mitigate it?
- **Q5.6** 🟡 When would you pick DiskANN over HNSW?
- **Q5.7** 🟡 How do you measure ANN recall properly?
- **Q5.8** 🏗️ Choose a vector store for (a) 2M chunks already in Postgres, (b) a 5B-vector multi-tenant SaaS, (c) heavy hybrid with custom ranking. Justify each.
- **Q5.9** 🟡 Per-tenant collections vs. a shared collection with `tenant_id`: trade-offs?

## Module 06: Query understanding
- **Q6.1** 🟢 What is conversational condensing? Give an example.
- **Q6.2** 🟡 How does HyDE work, and when does it hurt?
- **Q6.3** 🟡 Multi-query / RAG-Fusion: mechanism, benefit, cost.
- **Q6.4** 🟡 When should you decompose a query? Give an example.
- **Q6.5** 🔴 How would you detect that your router silently sends 8% of analytics questions to the text KB?
- **Q6.6** 🟡 What is step-back prompting?
- **Q6.7** 🏗️ Design a single structured "query understanding" call: fields and validation.

## Module 07: Retrieval
- **Q7.1** 🟢 Why is BM25 better than dense retrieval for error codes and part numbers?
- **Q7.2** 🟡 Explain the role of k1 and b in BM25.
- **Q7.3** 🧮 N = 1,000, the term appears in 2 docs. Compute BM25 IDF.
- **Q7.4** 🧮 tf = 2, k1 = 1.2, b = 0.75, |D| = 120, avgdl = 200. Compute the BM25 TF component.
- **Q7.5** 🟡 How does SPLADE produce a sparse vector? Why can it match words that aren't in the document?
- **Q7.6** 🟢 Why can't you simply add BM25 scores and cosine similarities?
- **Q7.7** 🟢 Write the RRF formula. What is k, and what is its usual value?
- **Q7.8** 🧮 BM25 ranks: A1, B2, C3, D4. Dense ranks: C1, A2, E3, B4. Compute RRF (k=60) for all docs and give the final order.
- **Q7.9** 🧮 BM25 scores A 12.4, B 9.1, C 7.8, D 3.2; dense C 0.82, A 0.79, E 0.74, B 0.61. Min-max normalize both, then compute the hybrid with α = 0.6 (dense weight). Order?
- **Q7.10** 🔴 Name two pitfalls of min-max normalization visible in Q7.9 and two alternatives.
- **Q7.11** 🟡 When would you choose RRF vs. convex combination vs. learning-to-rank?
- **Q7.12** 🧮 MMR with λ = 0.7. X is selected (sim to q 0.90). Candidates: Y (rel 0.88, sim(X,Y) = 0.97), Z (rel 0.80, sim(X,Z) = 0.30). Which is picked next?
- **Q7.13** 🟡 List four common retrieval bugs.
- **Q7.14** 🔴 How do you choose K after reranking so the system can abstain in a principled way?

## Module 08: Reranking
- **Q8.1** 🟢 What do `[CLS]`, `[SEP]`, and segment IDs mean in a cross-encoder input?
- **Q8.2** 🟢 Why are cross-encoders more accurate than bi-encoders?
- **Q8.3** 🟡 Why can't cross-encoders be used for first-stage retrieval? Give the cost argument.
- **Q8.4** 🧮 A cross-encoder outputs logits 2.3 and −1.1. Convert them to probabilities.
- **Q8.5** 🟡 Describe how a cross-encoder is trained: data, negatives, two loss functions.
- **Q8.6** 🧮 LCE loss for logits [4.0 (positive), 1.0, 0.5, −1.0].
- **Q8.7** 🟡 How does monoT5 produce a relevance score?
- **Q8.8** 🟡 How does an LLM pointwise reranker like Qwen3-Reranker produce a score in one forward pass?
- **Q8.9** 🟡 How do listwise rerankers handle more passages than fit in context? What are their downsides?
- **Q8.10** 🧮 ColBERT MaxSim: q1 sims to [d1, d2, d3] = [0.82, 0.31, 0.40]; q2 = [0.35, 0.77, 0.20]. Score?
- **Q8.11** 🟡 Name the reranker family and rough size of: bge-reranker-v2-m3, ms-marco-MiniLM-L-6-v2, Qwen3-Reranker-4B, RankZephyr.
- **Q8.12** 🔴 Design a cascade reranking setup and say where you'd put business priors.
- **Q8.13** 🔴 How do you calibrate a reranker threshold for "I don't know"?
- **Q8.14** 🟡 Why does the reranked "carry-on luggage" document drop in the PTO example? What does that show about BM25?

## Module 09: Advanced RAG
- **Q9.1** 🟢 Explain Self-RAG's reflection tokens.
- **Q9.2** 🟡 Describe CRAG's three actions.
- **Q9.3** 🟡 Walk through GraphRAG indexing and global search.
- **Q9.4** 🔴 What did the 2026 "Do We Still Need GraphRAG?" (RAGSearch) benchmark conclude?
- **Q9.5** 🟡 How does HippoRAG use Personalized PageRank for multi-hop questions?
- **Q9.6** 🟡 What guardrails does an agentic RAG loop need?
- **Q9.7** 🟡 How does ColPali retrieve, and what is its storage challenge?
- **Q9.8** 🔴 When is ColPali strictly better than OCR + text embeddings, and when is it worse?
- **Q9.9** 🏗️ Your evaluation shows 12% of queries are multi-hop comparisons that fail. Propose and justify an architecture change.
- **Q9.10** 🟢 What is Cache-Augmented Generation and when does it fit?

## Module 10: Generation & context engineering
- **Q10.1** 🟢 What is context rot?
- **Q10.2** 🟢 What is "lost in the middle"?
- **Q10.3** 🟡 How does prompt caching influence prompt layout?
- **Q10.4** 🟡 Why repeat the user question at the end of a long prompt?
- **Q10.5** 🟡 Four ways to produce citations, with pros and cons.
- **Q10.6** 🟡 How should the model handle conflicting sources?
- **Q10.7** 🧮 Window target 32k; system 600, memory 300, turns 1,200, question 50, output reserve 1,000. What's the source budget (ignoring tools)? How many 1,500-token parents fit?
- **Q10.8** 🟡 What is RAFT and when should you consider it?

## Module 11: Conversation & memory
- **Q11.1** 🟢 Name the five memory tiers.
- **Q11.2** 🟡 Why does retrieval use the condensed query while generation sees the raw recent turns?
- **Q11.3** 🟡 Describe the ADD/UPDATE/DELETE/NOOP memory write policy.
- **Q11.4** 🔴 How do you stop long-term memory from becoming a source of stale hallucinations?

## Module 12: Evaluation
- **Q12.1** 🟢 Why is first-stage Recall@k the ceiling for the whole system?
- **Q12.2** 🧮 Gold = {d3, d7}; ranking = [d1, d3, d5, d7, d9]. Compute Recall@3, Recall@5, P@5, MRR, AP, and nDCG@5 (binary).
- **Q12.3** 🧮 The same ranking with grades d3 = 3 and d7 = 1. Compute graded nDCG@5 (gain 2^rel − 1).
- **Q12.4** 🧮 An answer has 4 claims; 3 are supported by context. Faithfulness? If the 3 supported ones are cited and the 4th isn't, what are citation precision and citation recall?
- **Q12.5** 🧮 Judge vs. human on 100 items: agree 85; human pass 60%, judge pass 55%. Compute Cohen's κ.
- **Q12.6** 🟡 List five LLM-as-judge biases or pitfalls and their mitigations.
- **Q12.7** 🟡 How does ARES differ from RAGAS?
- **Q12.8** 🟡 What should a golden set contain, and how big should it be?
- **Q12.9** 🟡 Describe the paired bootstrap test.
- **Q12.10** 🔴 Interpret this failure funnel: not in corpus 12, not in top-100 18, not in top-k 25, in context but wrong 30, label noise 15. What do you prioritize?
- **Q12.11** 🟡 Name three implicit online signals of dissatisfaction.
- **Q12.12** 🏗️ Design release gates for a policy bot with 20% unanswerable questions.

## Module 13: Guardrails & security
- **Q13.1** 🟢 Direct vs. indirect prompt injection?
- **Q13.2** 🟡 How can a markdown image exfiltrate data?
- **Q13.3** 🟡 How does PoisonedRAG work, and how do you defend against it?
- **Q13.4** 🔴 How would you *prove* your system never retrieves unauthorized chunks?
- **Q13.5** 🟡 What is spotlighting/datamarking?
- **Q13.6** 🟡 Rank the hallucination controls by ROI.
- **Q13.7** 🟡 Why are embeddings sensitive data?
- **Q13.8** 🟡 Which retrieval paths must all carry the ACL filter?

## Module 14: Production
- **Q14.1** 🟢 Which stages can run in parallel in the serving path?
- **Q14.2** 🧮 $3/M input, $15/M output, cache reads at 10%. A query has 6,000 input tokens (1,500 cacheable) and 400 output tokens. Compute the cost without and with caching.
- **Q14.3** 🟡 Where does semantic caching become a security bug or a correctness bug?
- **Q14.4** 🟡 What must a trace log per request?
- **Q14.5** 🟡 What is speculative parallelism for condensing + retrieval?
- **Q14.6** 🟡 Name four index-lifecycle concerns.

## Module 15: Enterprise case studies
- **Q15.1** 🟢 What were LinkedIn's reported gains, and what was the key idea?
- **Q15.2** 🟢 Describe DoorDash's quality architecture.
- **Q15.3** 🟡 What is NVIDIA's FACTS framework?
- **Q15.4** 🟡 What is the lesson of Moffatt v. Air Canada?
- **Q15.5** 🔴 Across all the case studies, what are the three most transferable lessons?

## Module 16: Build plan
- **Q16.1** 🟢 Why must Phase 0 (evaluation) precede all tuning?
- **Q16.2** 🟡 What gate must chunking changes pass in Phase 2?
- **Q16.3** 🔴 Why are all components defined behind protocols/interfaces, and how does that change experiments?

## Capstone scenarios (🔴🏗️)
- **QC.1** Design a RAG assistant for a 2M-document, multilingual (EN/DE/JA) engineering knowledge base with per-team ACLs, a p95 latency of 3 s, and $0.01/query. Specify every stage.
- **QC.2** After switching embedding models, Recall@10 rose 6 points but user thumbs-down rose 10%. Give five hypotheses and how you'd test each.
- **QC.3** A legal team needs answers with clause-level citations across 50k contracts, including "which contracts lack a limitation-of-liability clause?" Which parts of the standard RAG pipeline break, and what do you add?
- **QC.4** Explain to a VP, in 5 bullets, why "just use a bigger LLM" won't fix your bot's accuracy.
