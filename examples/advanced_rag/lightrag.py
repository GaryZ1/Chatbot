"""LightRAG (Guo et al., 2024): graph + vectors with DUAL-LEVEL retrieval. The LLM extracts
low-level keywords (specific entities) and high-level keywords (themes) from the query.
Low-level keys are matched against entity vectors, high-level keys against relation
vectors; both are expanded one hop and merged with their source chunks. New documents are
merged into the graph incrementally (no community re-computation as in GraphRAG)."""

from graphrag import Graph, build_graph
from rag_kit import Chunk, DenseIndex, generate_answer, get_llm, parse_json, show

KEYWORDS = """<question>{q}</question>
Extract keywords. "low_level": specific entities, names, numbers; "high_level": broad
concepts or themes. Return JSON {{"low_level": [...], "high_level": [...]}}."""


class LightRAG:
    def __init__(self, g: Graph | None = None) -> None:
        self.g = g or build_graph()
        self.reindex()

    def reindex(self) -> None:
        self.names = sorted(self.g.entities)
        self.rels = sorted(self.g.edges)
        self.entity_index = DenseIndex(
            [f"{n} {' '.join(self.g.entities[n]['descriptions'])}" for n in self.names])
        self.relation_index = DenseIndex(
            [f"{a} {b} {' '.join(self.g.edges[(a, b)])}" for a, b in self.rels])

    def insert(self, chunks: list[Chunk]) -> None:
        """Incremental update: extract the new chunks and union them into the graph."""
        new = build_graph(chunks)
        for name, info in new.entities.items():
            cur = self.g.entities.setdefault(name, {"type": info["type"], "descriptions": [],
                                                    "chunks": set()})
            cur["descriptions"] += info["descriptions"]
            cur["chunks"] |= info["chunks"]
        for key, ds in new.edges.items():
            self.g.edges[key] += ds
        self.g.chunks.update(new.chunks)
        self.reindex()

    def query(self, question: str, k: int = 3) -> dict:
        kw = parse_json(get_llm()(KEYWORDS.format(q=question), task="keywords"), {})
        low, high = " ".join(kw.get("low_level", [])), " ".join(kw.get("high_level", []))
        ents = [self.names[i] for i in self.entity_index.scores(low or question)
                .argsort()[::-1][:k]]
        rels = [self.rels[i] for i in self.relation_index.scores(high or question)
                .argsort()[::-1][:k]]
        chunk_ids = {c for n in ents for c in self.g.entities[n]["chunks"]}
        chunk_ids |= {c for a, b in rels for n in (a, b) for c in self.g.entities[n]["chunks"]}
        cands = [self.g.chunks[c] for c in sorted(chunk_ids)]
        order = DenseIndex([c.text for c in cands]).scores(question).argsort()[::-1]
        rel_text = "\n".join(f"{a} -- {b}: {self.g.edges[(a, b)][0]}" for a, b in rels)
        answer = generate_answer(get_llm(), question, [cands[i] for i in order[:6]],
                                 extra=f"<relations>\n{rel_text}\n</relations>\n")
        return {"low_level_entities": ents, "high_level_relations": rels, "answer": answer}


def demo() -> None:
    rag = LightRAG()
    show("LightRAG", rag.query("How is Helios Aerospace connected to Orbital Systems?"))
    rag.insert([Chunk("n01#0", "n01", "Partnership", "Helios Aerospace and Lumen Labs signed "
                      "a star tracker supply deal in 2025.")])
    show("After incremental insert", rag.query("What did Helios Aerospace sign with Lumen "
                                               "Labs?"))


if __name__ == "__main__":
    demo()
