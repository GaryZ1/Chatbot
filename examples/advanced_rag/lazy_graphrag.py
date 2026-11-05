"""LazyGraphRAG (Microsoft Research, 2024-25): NO LLM at indexing. Build a cheap concept
graph from noun phrases (NLP) and their co-occurrence, cluster it, and defer all LLM work
to query time: rank communities by similarity to the query, then spend a fixed
"relevance-test budget" on LLM checks of their chunks (iterative deepening), and answer
from the chunks that pass. Indexing cost ~ vector RAG; the budget sets the quality/cost."""

from collections import Counter, defaultdict
from itertools import combinations

from rag_kit import (
    Chunk,
    DenseIndex,
    chunk_docs,
    content_terms,
    entities,
    generate_answer,
    get_llm,
    parse_json,
    show,
)


def noun_phrases(text: str) -> list[str]:
    """Cheap concept extraction: capitalised phrases + frequent content words.
    Production: spaCy noun chunks or a fine-tuned small extractor."""
    return entities(text) + [t for t in content_terms(text) if len(t) > 6]


def build_concept_index(chunks: list[Chunk], hub_share: float = 0.2):
    concept_chunks, cooc = defaultdict(set), Counter()
    for c in chunks:
        for x in set(noun_phrases(c.text)):
            concept_chunks[x].add(c.id)
    hubs = {x for x, ids in concept_chunks.items() if len(ids) > hub_share * len(chunks)}
    for c in chunks:                       # hub concepts would glue everything together
        cooc.update(combinations(sorted(set(noun_phrases(c.text)) - hubs), 2))
    # Communities: connected components over strong co-occurrence edges (Leiden in prod).
    parent = {x: x for x in concept_chunks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cooc:
        parent[find(a)] = find(b)
    comms = defaultdict(set)
    for x in concept_chunks.keys() - hubs:
        comms[find(x)] |= concept_chunks[x]
    return [sorted(ids) for ids in comms.values()]


def lazy_graphrag(question: str, budget: int = 6) -> dict:
    llm, chunks = get_llm(), chunk_docs()
    by_id = {c.id: c for c in chunks}
    comms = build_concept_index(chunks)
    q_scores = DenseIndex([" ".join(by_id[i].text for i in ids) for ids in comms]) \
        .scores(question)
    chunk_score = dict(zip(by_id, DenseIndex([c.text for c in chunks]).scores(question),
                           strict=True))
    relevant, tests, seen = [], 0, set()
    for ci in q_scores.argsort()[::-1]:                     # best communities first
        ranked = sorted(set(comms[ci]) - seen, key=lambda i: -chunk_score[i])
        for cid in ranked:                                  # iterative deepening
            seen.add(cid)
            if tests >= budget:
                break
            tests += 1
            grade = parse_json(llm(f"<question>{question}</question><document>"
                                   f"{by_id[cid].text}</document>\nIs the document relevant? "
                                   'Return JSON {"relevant": bool, "score": 0-1}.',
                                   task="grade"), {})
            if grade.get("relevant"):
                relevant.append(by_id[cid])
        if tests >= budget:
            break
    return {"communities": len(comms), "relevance_tests": tests,
            "relevant": [c.id for c in relevant],
            "answer": generate_answer(llm, question, relevant)}


def demo() -> None:
    show("LazyGraphRAG", lazy_graphrag("How did Orbital Systems respond when telemetry logs were exposed?"))


if __name__ == "__main__":
    demo()
