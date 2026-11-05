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

---

# Deep dive: computing every metric by hand

Running example: gold relevant chunks for query Q are **{d3, d7}**. The system returns the ranked list **[d1, d3, d5, d7, d9]**.

| Metric | Formula | Calculation | Value |
|---|---|---|---|
| Hit@1 | any relevant in the top 1? | d1 not relevant | 0 |
| Hit@3 | any relevant in the top 3? | d3 at rank 2 | 1 |
| Recall@3 | relevant in top 3 / total relevant | {d3} / 2 | **0.50** |
| Recall@5 | | {d3, d7} / 2 | **1.00** |
| Precision@5 | relevant in top 5 / 5 | 2/5 | **0.40** |
| MRR | 1 / rank of first relevant | 1/2 | **0.50** |
| AP (average precision) | mean of precision at each relevant hit | (P@2 = 1/2 + P@4 = 2/4)/2 | **0.50** |
| DCG@5 (binary) | Σ rel_i / log2(i+1) | 1/log2(3) + 1/log2(5) = 0.631 + 0.431 | 1.062 |
| IDCG@5 | DCG of the ideal order [d3, d7, …] | 1/log2(2) + 1/log2(3) = 1 + 0.631 | 1.631 |
| **nDCG@5** | DCG / IDCG | 1.062 / 1.631 | **0.651** |

**Graded nDCG:** with relevance grades (d3 = 3 "perfect", d7 = 1 "partial"), use gain `2^rel − 1`: DCG = 7/1.585 + 1/2.322 = 4.416 + 0.431 = 4.847; IDCG = 7/1 + 1/1.585 = 7.631 → nDCG = **0.635**.

Over a dataset, report the **mean** across queries (MRR = mean of the reciprocal ranks, MAP = mean of AP).

## Faithfulness (claim-level)
Answer: *"Up to 5 PTO days roll over [1]. Rolled-over days expire March 31 [1]. California employees get unlimited rollover [2]. Requests need manager approval."*
1. Decompose into claims: C1 (5 days), C2 (expire March 31), C3 (California unlimited), C4 (manager approval).
2. Check each against the context (NLI or judge): C1 ✓, C2 ✓, C3 ✓ (source 2 says CA has no forfeiture), C4 ✗ (not in any source).
3. Faithfulness = 3/4 = **0.75**.
4. Citation precision: C1–C3 citations support their claims → 3/3 = 1.0. Citation recall: 3 of the 4 claims carry citations → 0.75; C4 is unsupported *and* uncited → flag it as a hallucination.

## Context precision / recall (RAGAS-style)
- **Context recall:** split the *reference answer* into sentences; the share attributable to the retrieved context. Reference has 4 statements, 3 found → 0.75.
- **Context precision:** for each retrieved chunk, is it relevant? It's computed rank-aware (like AP over relevance flags). Chunks [rel, irrel, rel] → (1/1 + 2/3)/2 = **0.833**.

## Judge calibration: Cohen's κ
100 answers labeled pass/fail by a human and the judge. They agree on 85. The human says "pass" for 60, the judge for 55.
- Observed agreement p_o = 0.85
- Chance agreement p_e = 0.60·0.55 + 0.40·0.45 = 0.33 + 0.18 = 0.51
- κ = (p_o − p_e) / (1 − p_e) = 0.34 / 0.49 = **0.69** ("substantial"; aim for ≥ 0.6–0.8)
Also look at the confusion matrix: a judge that is too lenient (passes what the human fails) is more dangerous than one that is too strict.

## Paired bootstrap significance (is B really better than A?)
```python
import numpy as np
diff = scores_B - scores_A                 # per-question metric, same questions
boots = [np.mean(np.random.choice(diff, len(diff), replace=True)) for _ in range(10_000)]
lo, hi = np.percentile(boots, [2.5, 97.5])  # 95% CI of the mean improvement
# if lo > 0 → B is significantly better
```
Rule of thumb: with ~300 questions you can reliably detect ~4–6-point differences in accuracy-type metrics.

## Retrieval funnel example (error analysis)
| Bucket | Count (of 100 failures) | Meaning | Action |
|---|---|---|---|
| Gold not in corpus | 12 | F1 coverage | Content team |
| Gold not in top-100 | 18 | Parsing, chunking, embedding | Fix parse/chunk; hybrid; fine-tune |
| In top-100, not in top-k | 25 | Ranking | Better reranker; more candidates |
| In context, answer wrong | 30 | Generation | Prompt, ordering, model |
| Correct, but judged wrong | 15 | Evaluation noise | Fix labels or judge |
→ This team should prioritize ranking and generation, not a new embedding model.
