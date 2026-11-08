"""HippoRAG (Gutierrez et al., 2024) and HippoRAG 2 (2025): a hippocampus-inspired index.
Offline, OpenIE builds a knowledge graph (+ synonym edges between similar entity names).
At query time, link the query to graph nodes and run Personalized PageRank (PPR) from them;
probability mass flows to nodes connected to SEVERAL seeds, so a multi-hop answer is found
in ONE retrieval step. HippoRAG 2 adds passage nodes to the graph and seeds them with dense
scores, which keeps simple factual recall as strong as plain dense retrieval."""

import numpy as np
from graphrag import build_graph
from rag_kit import DenseIndex, HashingEmbedder, entities, generate_answer, get_llm, show


def build_hippo_graph(synonym_threshold: float = 0.8):
    g = build_graph()
    nodes = sorted(g.entities) + sorted(g.chunks)             # entity nodes + passage nodes
    idx = {n: i for i, n in enumerate(nodes)}
    adj = np.zeros((len(nodes), len(nodes)))
    for (a, b), descs in g.edges.items():                     # relation edges
        adj[idx[a], idx[b]] = adj[idx[b], idx[a]] = len(descs)
    for name, info in g.entities.items():                     # entity <-> passage edges
        for cid in info["chunks"]:
            adj[idx[name], idx[cid]] = adj[idx[cid], idx[name]] = 1.0
    ents = sorted(g.entities)
    vecs = HashingEmbedder().embed(ents)                      # synonym edges
    sim = vecs @ vecs.T
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            if sim[i, j] >= synonym_threshold:
                adj[idx[ents[i]], idx[ents[j]]] = adj[idx[ents[j]], idx[ents[i]]] = sim[i, j]
    return g, nodes, idx, adj


def personalized_pagerank(adj: np.ndarray, seed: np.ndarray, damping: float = 0.5,
                          iters: int = 50) -> np.ndarray:
    """r = (1 - d) * s + d * M^T r, with M the row-normalised adjacency."""
    deg = adj.sum(1, keepdims=True)
    m = np.divide(adj, deg, out=np.zeros_like(adj), where=deg > 0)
    s = seed / seed.sum()
    r = s.copy()
    for _ in range(iters):
        r = (1 - damping) * s + damping * m.T @ r
    return r


def hipporag_retrieve(question: str, k: int = 3, passage_weight: float = 0.05):
    g, nodes, idx, adj = build_hippo_graph()
    seed = np.zeros(len(nodes))
    ents = sorted(g.entities)
    qvecs = HashingEmbedder().embed(entities(question) or [question])
    evecs = HashingEmbedder().embed(ents)
    for qv in qvecs:                                  # link query entities -> graph nodes
        j = int((evecs @ qv).argmax())
        seed[idx[ents[j]]] += 1.0
    chunk_ids = sorted(g.chunks)
    dense = DenseIndex([g.chunks[c].text for c in chunk_ids]).scores(question)
    for c, s in zip(chunk_ids, dense, strict=True):  # HippoRAG 2: dense-seeded passages
        seed[idx[c]] += passage_weight * max(s, 0)
    ranks = personalized_pagerank(adj, seed)
    top = sorted(chunk_ids, key=lambda c: -ranks[idx[c]])[:k]
    return [g.chunks[c] for c in top], [ents[int((evecs @ qv).argmax())] for qv in qvecs]


def demo() -> None:
    q = "Which European Space Agency veteran is CTO at Orbital Systems?"
    hits, seeds = hipporag_retrieve(q)
    show("Seed nodes", seeds)
    show("PPR-ranked passages", [h.id for h in hits])
    show("Answer", generate_answer(get_llm(), q, hits))


if __name__ == "__main__":
    demo()
