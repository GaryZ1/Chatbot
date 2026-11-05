# 12 — Evaluation (the most important module)

> If you can't measure it, every change is a guess. Top teams build the evaluation **before** tuning, and gate every change on it.

## 1. Evaluation layers

| Layer | Question | Methods |
|---|---|---|
| **Component: parsing** | Is the text faithful? | Edit distance, table TEDS, human spot checks |
| **Component: retrieval** | Did we fetch the evidence? | Recall@k, nDCG@10, MRR, context precision/recall |
| **Component: rerank** | Did the best evidence rise to the top? | nDCG@5 after rerank, Recall@final_k |
| **Component: routing / condense** | Right pipeline, right standalone query? | Accuracy, confusion matrix, judge |
| **End-to-end: answer** | Correct, faithful, complete, cited? | Correctness vs. reference, faithfulness/groundedness, answer relevance, citation precision/recall |
| **Behavioral** | Abstains when appropriate? Safe? On-brand? | Unanswerable set, red-team set, style rubric |
| **Agentic** | Good trajectory? | Tool-call accuracy, steps, cost, success rate |
| **Online / business** | Does it help users? | Thumbs, CSAT, deflection/resolution rate, escalation rate, time-to-answer, retention, A/B tests |
| **Operational** | Fast and affordable? | p50/p95 latency per stage, $/query, error rates, freshness lag |

## 2. Metrics reference

**Retrieval (need relevance labels):**
- `Recall@k = |relevant ∩ top_k| / |relevant|`. The ceiling for everything downstream.
- `Precision@k`, `MRR = mean(1/rank_first_relevant)`, `nDCG@k` (graded relevance, position-discounted), `Hit@k`.

**Label-free / LLM-judged (RAGAS, DeepEval, TruLens "RAG triad"):**
- **Context relevance / precision:** share of retrieved context that is relevant to the query.
- **Context recall:** does the context contain everything needed for the reference answer?
- **Faithfulness / groundedness:** share of answer claims supported by the context (claim decomposition → NLI or judge per claim).
- **Answer relevance:** does it address the question?
- **Answer correctness:** semantic and factual match to a reference answer.
- **Citation precision / recall** (ALCE).
- **Noise sensitivity, abstention accuracy** (unanswerable questions → should refuse).

**Frameworks:** **RAGAS** (popular, metric library + synthetic test generation), **DeepEval** (pytest-style, G-Eval, CI-friendly), **ARES** (Stanford: fine-tunes lightweight judges on synthetic data + **prediction-powered inference** for confidence intervals using a small human-labeled set), **TruLens**, **Arize Phoenix**, **LangSmith**, **Braintrust**, **promptfoo**, **Inspect**. No single framework dominates in 2026. A common pattern is RAGAS/DeepEval in CI plus a custom calibrated judge for production monitoring.

**Benchmarks to know (to understand the field, not to choose a model):** BEIR, MTEB/RTEB, BRIGHT (reasoning retrieval), KILT, Natural Questions, HotpotQA / 2WikiMultiHop / MuSiQue (multi-hop), FRAMES (multi-hop + reasoning, Google 2024), CRAG benchmark (Meta, KDD Cup 2024), RGB (robustness), MultiHop-RAG, ViDoRe (visual), FinanceBench, LegalBench-RAG, RAGSearch (2026 agentic/graph), BrowseComp (deep research).

## 3. Building the golden dataset
1. **Sources of questions:**
   - Real logs (best): sample and cluster production queries; stratify by intent, product, and difficulty.
   - SME-written: 50–200 high-value questions with reference answers and gold sources.
   - **Synthetic:** LLM generates Q/A from chunks (RAGAS `TestsetGenerator`, evolution-style: simple, reasoning, multi-context, conditional). **Filter** with a judge for answerability and realism; humans review a sample.
2. **Include:** unanswerable questions (~10–15%), multi-hop, comparisons, time-sensitive, ambiguous, adversarial/injection, multilingual, typos.
3. **Labels:** `question, reference_answer, gold_doc_ids, gold_chunk_ids (optional), intent, difficulty, must_cite, should_abstain`.
4. **Size:** 200–500 to start (enough to detect differences of ~5 points); grow to 1–3k.
5. **Versioning:** treat it like code. Freeze a *test* split and never tune on it; keep a *dev* split for iteration.
6. **Refresh:** add every production failure (thumbs-down, escalations) after triage → a living regression suite.

## 4. LLM-as-judge done right
- **Use rubrics** with explicit criteria and a 1–5 or binary scale; ask for reasoning before the verdict; JSON output.
- **Prefer pairwise or binary** judgments over fine-grained scores (more reliable).
- **Calibrate:** label 100–300 items by hand; measure judge–human agreement (Cohen's κ, accuracy). Iterate the prompt until κ > ~0.6–0.8.
- **Biases:** position bias (randomize the order in pairwise), verbosity bias, self-preference (judge with a different model family), leniency.
- **Claim-level faithfulness** beats holistic "is this grounded?" scoring.
- **Cost control:** small judges fine-tuned on your labels (ARES-style), sampling in production.
- DoorDash's production pattern: an LLM judge across five quality dimensions + human review of random transcripts to keep it calibrated (Module 15).

## 5. Experimentation workflow
```
change (chunker / embedder / prompt / model)
   → run offline eval (dev split) → per-slice metrics + cost + latency
   → compare to baseline with confidence intervals (bootstrap / paired tests)
   → error analysis: sample 30 failures, assign F-codes (Module 00)
   → if better & no slice regression → test split → shadow/canary → online A/B
```
- **Statistical rigor:** paired bootstrap over questions; report 95% CIs; beware multiple comparisons.
- **Slice analysis:** a +2% overall can hide −15% on "policy questions in German".
- **CI gate:** a PR fails if faithfulness or Recall@k drops more than X points on the regression set.

## 6. Online evaluation and monitoring
- Implicit signals: copy events, follow-up rephrases (a dissatisfaction signal), escalation, session abandonment.
- Explicit: thumbs + reason codes; periodic SME review queues.
- Sampled production judging (faithfulness, relevance) with dashboards and alerts.
- **Drift:** query-distribution shift (new topics → cluster unseen queries), retrieval-score drift, corpus changes (re-run the evaluation after big ingests).
- A/B tests on business metrics; interleaving for ranking changes.

## 7. Error analysis (the expert's superpower)
For each failure, record: query → retrieved IDs → was gold in the top-100 / top-k / context? → answer → F-code. Then aggregate:
- Gold not in the top-100 → parsing, chunking, embedding, or coverage problem.
- In the top-100 but not the top-k → reranking or fusion problem.
- In context but wrong answer → generation or prompt problem (or a context-length problem).
This **retrieval funnel** tells you exactly where to invest.

## Expert checks
- Why is Recall@k of the first stage the ceiling of the whole system?
- How does ARES use prediction-powered inference, and why does it matter?
- How do you validate an LLM judge before trusting it?
- Design a golden set for a policy bot with 20% unanswerable questions. What metrics gate a release?
