# Advanced RAG examples

One small, self-contained file per method in
[`docs/20-advanced-rag-catalog.md`](../../docs/20-advanced-rag-catalog.md). Only `numpy` is
required.

```bash
python examples/advanced_rag/run_all.py
```

```bash
python examples/advanced_rag/ircot.py
```

```bash
pip install anthropic && RAG_LLM=claude python examples/advanced_rag/crag.py
```

| File | Role |
|---|---|
| `rag_kit.py` | corpus, BM25, hashing embedder, hybrid RRF retriever, Claude client, cited generation |
| `offline_llm.py` | deterministic heuristic LLM double (default), so every method runs in CI |
| `run_all.py` | runs every method's `demo()` |
| `build_doc.py` | inlines these files into the catalog doc (`--check` is enforced by a test) |
| everything else | one method each; see the doc for the explanation |

The offline double only shows the shape of each method's outputs. Compare methods with a
real model and the evaluation harness (`docs/12-evaluation.md`). After editing an example,
run `python examples/advanced_rag/build_doc.py` to refresh the doc.
