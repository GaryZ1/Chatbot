# 05 — Indexing & Vector Stores

## 1. ANN index algorithms

| Index | How it works | Pros | Cons | When |
|---|---|---|---|---|
| **Flat (brute force)** | Exact scan | 100% recall; simple | O(N) per query | < ~100k vectors; ground truth for recall tests |
| **HNSW** | Multi-layer proximity graph; greedy search | Excellent recall/latency; incremental inserts | RAM-heavy (vectors + graph); deletes degrade the graph; filtered search is tricky | Default up to ~10–100M vectors in RAM |
| **IVF (IVF-Flat)** | k-means clusters; probe `nprobe` clusters | Lower memory; fast build | Recall depends on nprobe; retrain when data shifts | Large static corpora |
| **IVF-PQ / OPQ** | IVF + product quantization (compress vectors into codes) | Billions of vectors, small RAM | Recall loss; needs rescoring | Very large scale (FAISS, Milvus) |
| **ScaNN** | Anisotropic quantization + reordering | Excellent speed/recall | Google ecosystem | Large scale (Vertex, AlloyDB) |
| **DiskANN / Vamana** (and SPANN) | Graph index on SSD with compressed vectors in RAM | Billion-scale at low RAM cost | SSD IOPS dependence | Huge corpora, cost-sensitive (pgvectorscale StreamingDiskANN, Azure, Milvus) |
| **Binary / RaBitQ quantized** | 1-bit codes + Hamming distance + rescore | 32× compression | Needs rescoring | Memory-bound deployments |
| **Inverted index (BM25 / SPLADE)** | Term → postings list | Exact-term search; mature | No semantics (BM25) | Always, for hybrid |
| **Multi-vector (PLAID, MUVERA)** | Centroid pruning for ColBERT; MUVERA turns multi-vector sets into single fixed vectors for ANN | Late interaction at scale | Complexity | ColBERT/ColPali first stage |

**HNSW tuning:** `M` (16–64; graph degree → recall and memory), `ef_construction` (100–400; build quality), `ef_search` (50–500; the query-time recall/latency knob). Always measure **recall@k against flat search** on a held-out query set. Aim for ≥ 0.95.

## 2. Vector database selection (2026)

| Option | Type | Strengths | Watch-outs |
|---|---|---|---|
| **pgvector (+ pgvectorscale)** | Postgres extension | Your data is already there; SQL joins, transactions, row-level security for ACLs; DiskANN via pgvectorscale | Tuning at very large scale; hybrid search needs extra work (pg_search / ParadeDB) |
| **Qdrant** | Dedicated, Rust | Fast filtered HNSW, payload indexes, sparse vectors, quantization, multi-vector | Separate system to operate |
| **Weaviate** | Dedicated | Built-in hybrid (BM25F + vector), modules, multi-tenancy | Memory footprint |
| **Milvus / Zilliz** | Dedicated, distributed | Billion scale, many index types (GPU CAGRA), sparse + dense | Operational complexity |
| **Vespa** | Search engine | Best-in-class hybrid, ranking expressions, ColBERT/multi-vector native, huge scale | Learning curve |
| **Elasticsearch / OpenSearch** | Search engine | Mature BM25 + kNN + ELSER/neural sparse; enterprise security | JVM ops; vector performance is improving but heavier |
| **Pinecone** | Managed serverless | Zero ops, sparse-dense, namespaces | Cost at scale; vendor lock-in |
| **Turbopuffer** | Object-storage-backed serverless | Very cheap at large multi-tenant scale | Higher cold-query latency |
| **LanceDB** | Embedded / columnar | Multimodal, local-first, cheap | Younger ecosystem |
| **Chroma** | Embedded / dev | Easy prototyping | Not the first choice for heavy production |
| **Azure AI Search / Vertex AI Search / OpenSearch Serverless / Bedrock KB** | Cloud managed | Integrated connectors, semantic ranker, security | Less control; cost |

**Decision rule:** < 5M chunks and already on Postgres → **pgvector**. Hybrid-heavy, high-QPS, complex ranking → **Vespa / Elastic / OpenSearch / Weaviate / Qdrant**. Massive multi-tenant SaaS → **Turbopuffer / Pinecone serverless / Milvus**. Don't pick from vendor benchmarks; test *your* filters and *your* QPS.

## 3. Index design

```
collection: chunks_v3   (alias: chunks_live)
  id                (chunk_id)
  dense_vector      (1024, int8, HNSW)  + float rescoring store
  sparse_vector     (SPLADE / BGE-M3 sparse)   or   bm25 text field (analyzer per language)
  text, embed_text, heading_path, doc_id, parent_id, page, char_span
  tenant_id, acl_groups[], classification
  doc_type, product[], language, effective_date, updated_at, version, is_latest
  quality_score, embedding_model, enrichment_version
```

- **Multi-tenancy:** a separate collection/namespace per tenant (strong isolation, simple deletes) vs. a shared collection with a `tenant_id` filter (efficient). Pick isolation for regulated customers.
- **Versioning:** keep `is_latest`; default-filter `is_latest = true`; allow "as of" queries.
- **Aliases** for zero-downtime re-indexing (blue/green).
- **Payload indexes** on every field you filter by (ACL, tenant, dates).

## 4. Index lifecycle
- Upserts keyed by `chunk_id = hash(doc_id, version, position)`; delete by `doc_id` on update.
- HNSW deletes leave tombstones → periodic compaction or rebuild.
- Track **freshness lag** (source update → searchable) as an SLO.
- Snapshots and backups; restore drills.

## Expert checks
- What do HNSW `M`, `ef_construction`, and `ef_search` each trade off?
- Why can a highly selective metadata filter collapse HNSW recall, and how do engines mitigate it?
- When would you pick DiskANN over HNSW?
