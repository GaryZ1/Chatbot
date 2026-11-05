# 15 — Enterprise Case Studies & Lessons

Figures are **as reported by the companies or authors**. Treat them as directional, not audited.

## 1. LinkedIn: knowledge-graph RAG for customer service (SIGIR 2024, arXiv 2404.17723)
- **Problem:** treating historical support tickets as flat text loses their structure and the links between tickets.
- **Approach:** parse each ticket into a **tree** (title, description, steps, resolution) using rules plus LLM extraction guided by a YAML template; link tickets into a graph (explicit references + semantic similarity); at query time, parse the query into entities and intents → retrieve subgraphs → generate.
- **Results:** **+77.6% MRR** retrieval improvement; **−28.6% median per-issue resolution time** after deployment.
- **Lesson:** *structure-aware chunking and enrichment of your dominant document type* (tickets) beats generic chunking. Domain structure is a moat.

## 2. DoorDash: Dasher support RAG with guardrail and judge (DoorDash engineering blog, 2024)
- **Approach:** conversation summarization → retrieve similar resolved cases and KB articles → generate. A **two-tier LLM Guardrail** (cheap check → LLM groundedness/coherence/compliance check) runs before responses ship. An **LLM Judge** scores five quality dimensions (including retrieval correctness and response accuracy). Humans review random transcripts to calibrate. A quality loop feeds KB improvements.
- **Results:** ~**90% fewer hallucinations**, ~**99% fewer severe compliance issues** (company-reported).
- **Lesson:** guardrails and continuous judging are *product features*, not afterthoughts. The judge identifies KB gaps to fix at the source.

## 3. Merck Electronics: R&D knowledge chatbot over 50k+ diverse documents (PyCon DE & PyData 2025)
- **Approach:** a production ingestion pipeline for heterogeneous R&D documents, a chatbot workflow, and a custom evaluation framework.
- **Lesson (verbatim theme):** a LangChain prototype is easy; production at 50k heterogeneous documents is about ingestion quality, evaluation, and lifecycle.

## 4. NVIDIA: FACTS framework for enterprise RAG chatbots (arXiv 2407.07858, 2024)
- From building three internal chatbots (IT/HR, financial earnings, general content): the **FACTS** dimensions are **F**reshness, **A**rchitecture, **C**ost, **T**esting, **S**ecurity.
- **15 control points** across the pipeline; lessons include hybrid search, metadata-aware chunking, the need for strong evaluation, and ACLs.
- **Lesson:** the hard parts are operational (freshness, testing, security), not the model.

## 5. Morgan Stanley Wealth Management: advisor assistant (OpenAI case study, 2023–24)
- **Approach:** GPT-4 assistant over ~100k internal research and procedure documents; extensive **evaluation suites with advisors and SMEs** before rollout; later a meeting-summary tool ("Debrief").
- **Results:** very high adoption among advisor teams (OpenAI reported ~98% of advisor teams using it); document retrieval efficiency reportedly improved substantially.
- **Lesson:** an expert-driven evaluation framework (graded by domain experts, per task) was the gating factor for trust and adoption.

## 6. Uber: Genie on-call copilot (Uber engineering blog, 2024)
- **Approach:** RAG over internal wikis, Stack Overflow-style Q&A, and runbooks, answering engineers in Slack support channels; an evaluation and feedback loop; later "enhanced agentic RAG" for better accuracy on engineering-security and policy documents.
- **Results:** reported many thousands of engineering hours saved and a large volume of questions answered.
- **Lesson:** meet users where they work (Slack); the feedback loop drives document-quality improvements.

## 7. Klarna: customer-service assistant (press release, Feb 2024 → 2025 follow-ups)
- **Reported:** 2.3M conversations in its first month, two-thirds of customer-service chats handled, equivalent to ~700 agents' work, resolution time from ~11 to ~2 minutes.
- **2025 follow-up:** Klarna publicly re-emphasized human agents for quality and customer experience.
- **Lesson:** optimize for **resolution quality and CSAT**, not only deflection. Keep humans in the loop for complex or emotional cases.

## 8. Cautionary tale: Moffatt v. Air Canada (BC Civil Resolution Tribunal, 2024)
- The airline's website chatbot gave wrong bereavement-fare information; the tribunal held the company **liable** for its chatbot's statements.
- **Lesson:** grounding, citations to authoritative policy, abstention, and policy-document freshness are legal requirements, not nice-to-haves.

## 9. Unique.ai: GraphRAG vs. agentic search (2026 research notes)
- Its agentic workflows outperformed default LightRAG configurations on sequential-logic and data-mapping tasks; it paused graph-based retrieval to focus on agentic search (April 2026).
- **Lesson:** consistent with arXiv 2604.09666. Strong agentic search over a good hybrid index is often a better investment than a knowledge graph.

## 10. Cross-cutting lessons (what the top teams agree on)
1. **Retrieval quality > model size.** Most bad answers are retrieval misses.
2. **Evaluation is the product.** Expert-graded golden sets + calibrated judges + online feedback.
3. **Data and lifecycle work dominate:** parsing, dedup, freshness, deprecation, ownership of content.
4. **Structure is leverage:** use the native structure of your documents (tickets, sections, tables).
5. **Security is table stakes:** ACL-aware retrieval and injection defenses from day one; multi-tenant isolation.
6. **Route, don't over-engineer:** a simple path for most queries; agents only for the hard tail.
7. **Demo accuracy ≠ production accuracy:** the query-distribution shift is real. Mine production logs continuously.
8. **Humans in the loop** for escalation, calibration, and content fixes.

## Sources
- LinkedIn: <https://arxiv.org/abs/2404.17723>
- DoorDash summaries: <https://www.zenml.io/llmops-database/llm-based-dasher-support-automation-with-rag-and-quality-controls>
- Merck (PyCon DE 2025): <https://pycon.de/archive/2025/talks/lessons-learned-in-bringing-a-rag-chatbot-with-access-to-50k-diverse-documents-to-production/>
- NVIDIA FACTS: <https://arxiv.org/abs/2407.07858>
- Field study, deploying LLMs with RAG: <https://arxiv.org/abs/2411.11895>
- Unique.ai GraphRAG research: <https://docs.unique.ai/administrators/research/rag-evaluations/graph-based-rag-research>
- GraphRAG vs. agentic benchmark: <https://arxiv.org/abs/2604.09666>
- Morgan Stanley, Uber Genie, Klarna, Air Canada: public company blog posts, press releases, and tribunal decision (search the titles above; figures as reported).

---

# Deep dive: mapping each case study to the pipeline

| Case | Parsing / chunking | Retrieval | Quality control | Main lever |
|---|---|---|---|---|
| LinkedIn | Tickets → section trees (LLM + rules) | Entity/intent parse → subgraph retrieval + embeddings | Offline MRR/BLEU-style + production A/B on resolution time | **Domain structure** (Modules 02, 03, 09) |
| DoorDash | KB articles + resolved cases | Conversation summary → similarity search | Two-tier guardrail + LLM judge + human sampling | **Guardrails + continuous judging** (12, 13) |
| Merck | Heterogeneous R&D documents, 50k+ | Production RAG pipeline | Custom evaluation framework | **Ingestion + evaluation** (01, 12) |
| NVIDIA FACTS | Metadata-aware chunking | Hybrid search | Testing as a first-class dimension | **Ops: freshness, testing, security** (14) |
| Morgan Stanley | Curated research corpus | Retrieval over ~100k documents | SME-graded evaluation suites | **Expert evaluation → trust** (12) |
| Uber Genie | Wikis, Q&A, runbooks | RAG → agentic RAG | User feedback in Slack | **Distribution channel + feedback** (11, 14) |
| Klarna | Policies, order data (tools) | RAG + tool calls | Business metrics (CSAT, resolution) | **Measure resolution quality, keep humans** |
| Air Canada | — | — | None effective | **Grounding + freshness are legal duties** (10, 13) |

**Exercise:** for your own use case, fill in this row before writing code.
