# 17 — Reading List & Repos

★ = must read. Arrange your study to follow the modules.

## Foundations
- ★ Lewis et al. 2020, *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (arXiv 2005.11401)
- ★ Gao et al. 2023–24, *Retrieval-Augmented Generation for Large Language Models: A Survey* (arXiv 2312.10997)
- ★ Barnett et al. 2024, *Seven Failure Points When Engineering a RAG System* (arXiv 2401.05856)
- Karpukhin et al. 2020, *Dense Passage Retrieval* (DPR)
- Robertson & Zaragoza 2009, *The Probabilistic Relevance Framework: BM25 and Beyond*

## Chunking & enrichment
- ★ Anthropic 2024, *Introducing Contextual Retrieval* (anthropic.com/news/contextual-retrieval)
- ★ Günther et al. 2024, *Late Chunking* (arXiv 2409.04701)
- ★ Qu et al. 2024, *Is Semantic Chunking Worth the Computational Cost?* (arXiv 2410.13070)
- ★ Chroma Research 2024, *Evaluating Chunking Strategies for Retrieval* (research.trychroma.com)
- Merola & Singh 2025, *Reconstructing Context* (arXiv 2504.19754)
- Chen et al. 2023, *Dense X Retrieval: What Retrieval Granularity Should We Use?* (arXiv 2312.06648)
- Sarthi et al. 2024, *RAPTOR* (arXiv 2401.18059)
- Duarte et al. 2024, *LumberChunker* (arXiv 2406.17526); Zhao et al. 2024, *Meta-Chunking* (arXiv 2410.12788)
- Nogueira et al. 2019, *Document Expansion by Query Prediction* (doc2query)

## Parsing
- OHRBench, *OCR Hinders RAG* (arXiv 2412.02592)
- OmniDocBench (opendatalab); Docling technical report (arXiv 2408.09869); MinerU / MinerU2.5; olmOCR (Ai2)

## Embeddings & retrieval
- ★ Muennighoff et al., *MTEB* (arXiv 2210.07316) and MMTEB (2025); RTEB (2025)
- ★ Khattab & Zaharia 2020, *ColBERT*; Santhanam et al. 2022, *ColBERTv2*
- Formal et al., *SPLADE* / SPLADE-v3
- Chen et al. 2024, *BGE M3-Embedding* (arXiv 2402.03216)
- Kusupati et al. 2022, *Matryoshka Representation Learning*
- Zhang et al. 2025, *Qwen3 Embedding* technical report
- Su et al. 2024, *BRIGHT* (reasoning-intensive retrieval)
- Thakur et al. 2021, *BEIR*
- Malkov & Yashunin, *HNSW*; Subramanya et al. 2019, *DiskANN*; Dhulipala et al. 2024, *MUVERA*

## Query transformation & reranking
- Gao et al. 2022, *HyDE* (arXiv 2212.10496); Ma et al. 2023, *Query Rewriting for RAG*; Zheng et al. 2023, *Step-Back Prompting*
- Query optimization survey (arXiv 2412.17558, revised 2026)
- Cormack et al. 2009, *Reciprocal Rank Fusion*
- Nogueira & Cho 2019, *Passage Re-ranking with BERT*; Sun et al. 2023, *RankGPT*; Pradeep et al., *RankZephyr*
- Lin, Nogueira, Yates, *Pretrained Transformers for Text Ranking: BERT and Beyond* (arXiv 2010.06467)

## Advanced RAG
- ★ Asai et al. 2023, *Self-RAG* (arXiv 2310.11511); Yan et al. 2024, *Corrective RAG* (arXiv 2401.15884); Jeong et al. 2024, *Adaptive-RAG*
- ★ Edge et al. 2024, *From Local to Global: A GraphRAG Approach* (arXiv 2404.16130)
- Guo et al. 2024, *LightRAG* (arXiv 2410.05779); Gutiérrez et al. 2024/2025, *HippoRAG / HippoRAG 2* (arXiv 2405.14831, 2502.14802)
- ★ *Do We Still Need GraphRAG? Benchmarking RAG and GraphRAG for Agentic Search* (arXiv 2604.09666, 2026)
- *A-RAG: Hierarchical Retrieval Interfaces for Agentic RAG* (arXiv 2602.03442, 2026)
- *GraphSearch* (arXiv 2509.22009); spreading activation for KG-RAG (arXiv 2512.15922)
- Jin et al. 2025, *Search-R1*; deep search agents survey (arXiv 2508.05668)
- Trivedi et al. 2023, *IRCoT*; Jiang et al. 2023, *FLARE*
- Faysse et al. 2024, *ColPali* (arXiv 2407.01449)
- Zhang et al. 2024, *RAFT*; Wang et al. 2024, *Speculative RAG*

## Generation, context & long context
- ★ Liu et al. 2023, *Lost in the Middle* (arXiv 2307.03172)
- ★ Chroma Research 2025, *Context Rot*
- Hsieh et al. 2024, *RULER*; Anthropic, *Effective context engineering for AI agents* (2025)
- Gao et al. 2023, *ALCE: Enabling LLMs to Generate Text with Citations*
- Databricks 2024, *Long Context RAG Performance of LLMs*

## Evaluation
- ★ Es et al. 2023, *RAGAS* (arXiv 2309.15217); ★ Saad-Falcon et al. 2023, *ARES* (arXiv 2311.09476)
- Zheng et al. 2023, *Judging LLM-as-a-Judge* (MT-Bench); Angelopoulos et al. 2023, *Prediction-Powered Inference*
- Yang et al. 2024, *CRAG – Comprehensive RAG Benchmark* (Meta); Krishna et al. 2024, *FRAMES*
- Chen et al. 2023, *RGB: Benchmarking LLMs in RAG*

## Security
- ★ OWASP Top 10 for LLM Applications (2025)
- Greshake et al. 2023, *Indirect Prompt Injection*; Zou et al. 2024, *PoisonedRAG* (arXiv 2402.07867)
- Morris et al. 2023, *Text Embeddings Reveal (Almost) As Much As Text* (Vec2Text)

## Enterprise
- ★ Xu et al. 2024 (LinkedIn), *RAG with Knowledge Graphs for Customer Service QA* (arXiv 2404.17723)
- ★ NVIDIA 2024, *FACTS About Building RAG-based Chatbots* (arXiv 2407.07858)
- DoorDash Engineering 2024, *Path to high-quality LLM-based Dasher support automation*
- ZenML LLMOps Database (hundreds of production case studies): zenml.io/llmops-database

## Repos to study (read the code, not just the README)
| Repo | Why |
|---|---|
| microsoft/graphrag | Reference GraphRAG implementation |
| HKUDS/LightRAG | Lightweight graph RAG |
| OSU-NLP-Group/HippoRAG | PPR-based memory retrieval |
| infiniflow/ragflow | Full open-source RAG engine with deep document understanding |
| onyx-dot-app/onyx | Enterprise search/chat with connectors and permissions |
| docling-project/docling, opendatalab/MinerU | Document parsing |
| run-llama/llama_index, langchain-ai/langgraph, deepset-ai/haystack | Orchestration patterns |
| stanfordnlp/dspy | Programmatic prompt and pipeline optimization |
| explodinggradients/ragas, confident-ai/deepeval, stanford-futuredata/ARES | Evaluation |
| embeddings-benchmark/mteb | Embedding evaluation |
| stanford-futuredata/ColBERT, lightonai/pylate, illuin-tech/colpali | Late interaction |
| FlagOpen/FlagEmbedding | BGE embedders and rerankers |
| chonkie-inc/chonkie | Chunking library to compare methods |
| qdrant/qdrant, milvus-io/milvus, vespa-engine/vespa | Vector and search engines |
| langfuse/langfuse, Arize-ai/phoenix | Observability |
| NVIDIA/garak, promptfoo/promptfoo | Red-teaming |
| PeterGriffinJin/Search-R1 | RL-trained search agents |

## Staying current
- arXiv cs.IR / cs.CL (search "retrieval-augmented", "agentic search", "reranking")
- SIGIR, ECIR, ACL, EMNLP, NeurIPS (datasets & benchmarks) proceedings
- MTEB / RTEB, ViDoRe, OmniDocBench, BRIGHT leaderboards
- Engineering blogs: Anthropic, OpenAI, Google, Cohere, Voyage, Jina, Vespa, Qdrant, Weaviate, LlamaIndex, LangChain, Databricks, Pinecone
