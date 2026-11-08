"""RAPTOR (Sarthi et al., 2024): recursively cluster chunk embeddings and summarise each
cluster, building a tree from leaves (chunks) to root (corpus summary). Retrieve over all
levels at once ("collapsed tree"), so broad questions hit summaries and narrow ones hit
leaves. (The paper uses UMAP + Gaussian mixtures; k-means keeps this dependency-free.)"""

import numpy as np
from rag_kit import Chunk, DenseIndex, HashingEmbedder, chunk_docs, get_llm, show


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = x[rng.choice(len(x), size=k, replace=False)]
    for _ in range(iters):
        labels = np.argmax(x @ centers.T, axis=1)              # cosine (rows are unit norm)
        for j in range(k):
            if (labels == j).any():
                c = x[labels == j].mean(0)
                centers[j] = c / (np.linalg.norm(c) or 1)
    return labels


def build_tree(leaves: list[Chunk], branching: int = 4, max_levels: int = 3) -> list[Chunk]:
    llm, emb = get_llm(), HashingEmbedder()
    nodes, level = list(leaves), list(leaves)
    for depth in range(1, max_levels + 1):
        if len(level) <= 1:
            break
        k = max(1, len(level) // branching)
        labels = kmeans(emb.embed(c.text for c in level), k)
        parents = []
        for j in range(k):
            members = [c for c, lab in zip(level, labels, strict=True) if lab == j]
            if not members:
                continue
            docs = "\n".join(f"<document>{m.text}</document>" for m in members)
            summary = llm(f"{docs}\nSummarise the key facts in these documents.",
                          task="summarize")
            parents.append(Chunk(f"L{depth}-{j}", "raptor", f"summary level {depth}", summary,
                                 {"level": depth, "children": [m.id for m in members]}))
        nodes += parents
        level = parents
    return nodes


def collapsed_tree_search(nodes: list[Chunk], query: str, k: int = 4) -> list[Chunk]:
    scores = DenseIndex([n.text for n in nodes]).scores(query)
    return [nodes[i] for i in scores.argsort()[::-1][:k]]


def demo() -> None:
    tree = build_tree(chunk_docs())
    show("Tree size", {"leaves": len(chunk_docs()), "all_nodes": len(tree)})
    hits = collapsed_tree_search(tree, "Give an overview of Orbital Systems")
    show("Collapsed-tree hits", [(h.id, h.meta.get("level", 0)) for h in hits])


if __name__ == "__main__":
    demo()
