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

---

# Deep dive: how ANN indexes work

## A. HNSW (Hierarchical Navigable Small World)
```
Layer 2:   A ─────────────── K                       (few nodes, long-range links)
Layer 1:   A ──── F ──── K ──── P                    (more nodes)
Layer 0:   A─B─C─D─E─F─G─H─I─J─K─L─M─N─O─P ...        (ALL nodes, short-range links, ≤ 2M neighbors)
```
- **Insert:** each vector gets a random top layer (probability decays exponentially, like a skip list). On each layer it links to its `M` nearest neighbors, found with a beam search of width `ef_construction`.
- **Search:** start at the entry point on the top layer → greedily move to the neighbor closest to the query → when there's no improvement, drop down a layer → on layer 0, run a beam search keeping the `ef_search` best candidates → return the top-k.
- **Why it's fast:** about O(log N) hops. Upper layers act as "highways", layer 0 as "local streets".
- **Memory:** vectors + about `M × 2 × 4 bytes` of links per node at layer 0. With 10M × 1024-d float32 (41 GB) plus links (~1.3 GB at M = 16), everything must sit in RAM, hence quantization.
- **Knob example:** `ef_search` = 40 → recall@10 = 0.91 at 1.1 ms; 128 → 0.97 at 2.6 ms; 512 → 0.995 at 8 ms (illustrative). Pick the knee of the curve.
- **Filtered search problem:** if a filter keeps only 0.5% of nodes, the greedy walk keeps hitting filtered-out nodes and the graph becomes "disconnected" for that query. Mitigations: engines switch to brute force below a selectivity threshold (Qdrant, Weaviate), filter-aware graph construction (ACORN, Qdrant payload-based subgraphs), or partitioning by tenant.

## B. IVF (inverted file)
1. Run k-means on the vectors → `nlist` centroids (e.g. √N to 4√N; 4,096 for 10M).
2. Each vector is stored in the list of its nearest centroid.
3. Query: find the `nprobe` nearest centroids (e.g. 16 of 4,096) → scan only those lists (~0.4% of the data).
Trade-off: a bigger `nprobe` means higher recall and slower search. Vectors near cluster borders are the ones that get missed.

## C. Product Quantization (PQ), worked example
- 1024-d vector → split into **m = 64 sub-vectors** of 16 dims each.
- For each sub-space, learn a codebook of **256 centroids** with k-means → each sub-vector is replaced by a 1-byte centroid ID.
- Storage: 64 bytes per vector instead of 4,096 → **64× compression**.
- Distance: precompute a 64 × 256 lookup table of (query sub-vector ↔ centroid) distances; the distance to any vector = the sum of 64 table lookups ("asymmetric distance computation"). Very fast.
- Cost: approximation error → retrieve 10× candidates and **rescore** with the original vectors.

## D. DiskANN (Vamana)
A single-layer graph built with pruning that keeps long-range edges (parameter α > 1). Compressed PQ vectors live in RAM; the full vectors and the graph live on SSD. A search does a few dozen SSD reads. Serves billions of vectors from one machine with ~64–128 GB RAM.

## E. Recall measurement script (always do this)
```python
gt   = flat_index.search(Q, k=10)          # exact ground truth
ann  = hnsw_index.search(Q, k=10, ef=128)
recall_at_10 = mean(len(set(a) & set(g)) / 10 for a, g in zip(ann, gt))
```
Measure it **with your production filters on**. Filtered recall is often much worse than unfiltered.
