"""Microsoft GraphRAG (Edge et al., 2024). Indexing: LLM extracts entities + relations per
chunk -> merge into a graph -> detect communities (hierarchical Leiden in the paper) ->
an LLM writes a report per community. Query: LOCAL search (entity neighbourhood + source
chunks + reports) for specific questions; GLOBAL search (map-reduce over community reports)
for corpus-wide "sensemaking" questions vector RAG cannot answer. DRIFT search starts
global, then fans out into local follow-ups."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rag_kit import (
    Chunk,
    DenseIndex,
    chunk_docs,
    format_context,
    generate_answer,
    get_llm,
    parse_json,
    show,
)

EXTRACT = """Extract all entities (people, organisations, products, places, events) and the
relationships between them from the text.
<text>{text}</text>
Return JSON {{"entities": [{{"name", "type", "description"}}],
"relations": [{{"source", "target", "description"}}]}}."""


@dataclass
class Graph:
    entities: dict[str, dict] = field(default_factory=dict)        # name -> info
    edges: dict[tuple[str, str], list[str]] = field(default_factory=lambda: defaultdict(list))
    chunks: dict[str, Chunk] = field(default_factory=dict)

    def neighbours(self, name: str) -> set[str]:
        return {b if a == name else a for a, b in self.edges if name in (a, b)}


def build_graph(chunks: list[Chunk] | None = None) -> Graph:
    llm, g = get_llm(), Graph()
    for c in chunks or chunk_docs():
        g.chunks[c.id] = c
        out = parse_json(llm(EXTRACT.format(text=c.text), task="extract_graph"), {})
        for e in out.get("entities", []):
            info = g.entities.setdefault(e["name"], {"type": e.get("type", ""),
                                                     "descriptions": [], "chunks": set()})
            info["descriptions"].append(e.get("description", ""))
            info["chunks"].add(c.id)
        for r in out.get("relations", []):
            key = tuple(sorted((r["source"], r["target"])))
            g.edges[key].append(r.get("description", ""))       # merged across chunks
    return g


def communities(g: Graph, iters: int = 10) -> dict[str, int]:
    """Weighted label propagation (stand-in for graspologic's hierarchical_leiden)."""
    labels = {n: i for i, n in enumerate(sorted(g.entities))}
    for _ in range(iters):
        changed = False
        for n in sorted(g.entities):
            weights = Counter()
            for (a, b), descs in g.edges.items():
                if n in (a, b):
                    weights[labels[b if a == n else a]] += len(descs)
            if weights:
                best = max(sorted(weights), key=lambda lab: weights[lab])
                changed |= best != labels[n]
                labels[n] = best
        if not changed:
            break
    return labels


def community_reports(g: Graph, labels: dict[str, int]) -> list[dict]:
    llm, groups = get_llm(), defaultdict(list)
    for n, lab in labels.items():
        groups[lab].append(n)
    reports = []
    for lab, members in sorted(groups.items()):
        facts = {d for n in members for d in g.entities[n]["descriptions"]}
        facts |= {d for (a, b), ds in g.edges.items() if a in members and b in members
                  for d in ds}
        docs = "\n".join(f"<document>{f}</document>" for f in sorted(facts))
        summary = llm(f"{docs}\nWrite a community report: the key entities, how they relate, "
                      "and the most important findings.", task="summarize")
        reports.append({"id": lab, "entities": members, "report": summary})
    return reports


def local_search(g: Graph, reports: list[dict], question: str, k_entities: int = 3) -> str:
    names = list(g.entities)
    index = DenseIndex([f"{n}: {' '.join(g.entities[n]['descriptions'])}" for n in names])
    seeds = [names[i] for i in index.scores(question).argsort()[::-1][:k_entities]]
    ctx_ids = {cid for n in seeds for cid in g.entities[n]["chunks"]}
    relations = [f"{a} -- {b}: {ds[0]}" for (a, b), ds in g.edges.items()
                 if a in seeds or b in seeds][:8]
    reps = [r["report"] for r in reports if set(r["entities"]) & set(seeds)][:2]
    extra = "<relations>\n" + "\n".join(relations) + "\n</relations>\n<reports>\n" + \
        "\n".join(reps) + "\n</reports>\n"
    return generate_answer(get_llm(), question, [g.chunks[c] for c in sorted(ctx_ids)], extra)


def global_search(reports: list[dict], question: str, top: int = 5) -> dict:
    llm, points = get_llm(), []
    for r in reports:                                                    # MAP (parallelisable)
        out = parse_json(llm(f"<report>{r['report']}</report>\n<question>{question}</question>"
                             "\nList the key points from this report that help answer the "
                             'question, each with a helpfulness score 0-100. Return JSON '
                             '{"points": [{"text", "score"}]}.', task="map"), {})
        points += [p | {"community": r["id"]} for p in out.get("points", []) if p.get("text")]
    points.sort(key=lambda p: -p["score"])
    ctx = format_context([Chunk(f"c{p['community']}", "", "community", p["text"])
                          for p in points[:top]])
    answer = llm(f"<context>{ctx}</context>\n<question>{question}</question>\nCombine the "
                 "points into a comprehensive answer.", task="reduce")       # REDUCE
    return {"n_points": len(points), "top_points": [p["text"] for p in points[:top]],
            "answer": answer}


def demo() -> None:
    g = build_graph()
    labels = communities(g)
    reports = community_reports(g, labels)
    show("Graph", {"entities": len(g.entities), "edges": len(g.edges),
                   "communities": len(reports)})
    show("Local search", local_search(g, reports, "What did Theo Brandt do before Orbital?"))
    show("Global search", global_search(reports, "What are the main themes about Orbital "
                                                 "Systems across all documents?"))


if __name__ == "__main__":
    demo()
