"""Think-on-Graph (ToG, Sun et al., 2023; ToG-2, 2024): the LLM walks a knowledge graph with
beam search. From the topic entities it scores candidate relations/neighbours, keeps the
top-N paths, and at each depth asks whether the paths already answer the question. ToG-2
also pulls the text of the entities it visits (tight graph + text coupling)."""

from graphrag import build_graph
from rag_kit import entities, get_llm, overlap, parse_json, show

SCORE = """<question>{q}</question>
<paths>
{paths}
</paths>
Score how useful each reasoning path is for answering the question (0-1).
Return a JSON list of numbers in the same order."""
ENOUGH = """<question>{q}</question>
<context>{paths}</context>
Can the question be answered from these reasoning paths? Return JSON
{{"verdict": "yes"|"no", "score": 0-1}}."""


def think_on_graph(question: str, width: int = 3, depth: int = 3) -> dict:
    llm, g = get_llm(), build_graph()
    topic = [e for e in g.entities if any(overlap(e, q) == 1 for q in entities(question))]
    beams = [[t] for t in topic[:width]] or [[next(iter(g.entities))]]
    for _ in range(depth):
        candidates = []
        for path in beams:                                   # expand the frontier entity
            tail = path[-1]
            for (a, b), descs in g.edges.items():
                if tail in (a, b):
                    nxt = b if a == tail else a
                    if nxt not in path[::2]:
                        candidates.append([*path, descs[0], nxt])
        if not candidates:
            break
        rendered = "\n".join(f"{i}. " + " -> ".join(p) for i, p in enumerate(candidates))
        default = [overlap(question, " ".join(p)) for p in candidates]
        scores = parse_json(llm(SCORE.format(q=question, paths=rendered), task="rank_paths"),
                            default)
        if not (isinstance(scores, list) and len(scores) == len(candidates)):
            scores = default
        beams = [p for _, p in sorted(zip(scores, candidates, strict=True),
                                      key=lambda x: -x[0])[:width]]   # prune to top-N
        facts = " ".join(dict.fromkeys(s for p in beams for s in p[1::2]))
        if parse_json(llm(ENOUGH.format(q=question, paths=facts), task="judge"),
                      {}).get("verdict") == "yes":
            break
    facts = " ".join(dict.fromkeys(s for p in beams for s in p[1::2]))
    answer = llm(f"<context>{facts}</context>\n<question>{question}</question>", task="answer")
    return {"paths": [" -> ".join(p) for p in beams], "answer": answer}


def demo() -> None:
    show("Think-on-Graph", think_on_graph("Where is the Lumen Labs star tracker factory?"))


if __name__ == "__main__":
    demo()
