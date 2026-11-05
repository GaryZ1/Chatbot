from pathlib import Path

import pytest

from chatbot.evaluation.dataset import GoldenExample, GoldenSet, Split, split_of

DATA = Path(__file__).resolve().parents[3] / "eval" / "datasets"


def _ex(i: int, intent: str = "lookup", **kw) -> GoldenExample:
    return GoldenExample(
        id=f"q{i}",
        question=f"question {i}?",
        reference_answer="ans",
        gold_doc_ids=["file:a.md"],
        intent=intent,
        **kw,
    )


def test_roundtrip_and_hash(tmp_path):
    gs = GoldenSet([_ex(1), _ex(2, slices=["x"])], name="t")
    p = tmp_path / "g.jsonl"
    gs.save(p)
    loaded = GoldenSet.load(p)
    assert [e.id for e in loaded] == ["q1", "q2"]
    assert loaded.version_hash() == gs.version_hash()
    # Order-independent; any label change creates a new version.
    assert GoldenSet([_ex(2, slices=["x"]), _ex(1)]).version_hash() == gs.version_hash()
    assert GoldenSet([_ex(1), _ex(2)]).version_hash() != gs.version_hash()


def test_split_is_frozen_and_deterministic():
    ids = [f"id-{i}" for i in range(400)]
    first = [split_of(i) for i in ids]
    assert first == [split_of(i) for i in ids]
    share = sum(s is Split.TEST for s in first) / len(ids)
    assert 0.2 < share < 0.4
    gs = GoldenSet(_ex(i) for i in range(50))
    test_ids = {e.id for e in gs.test()}
    bigger = GoldenSet([*gs, *(_ex(i) for i in range(50, 100))])
    assert test_ids <= {e.id for e in bigger.test()}  # adding data never moves old examples
    assert not test_ids & {e.id for e in gs.dev()}


def test_stratified_sample():
    gs = GoldenSet(
        [*(_ex(i, "lookup") for i in range(80)), *(_ex(i, "procedure") for i in range(80, 100))]
    )
    s = gs.stratified_sample(10, by="intent", seed=3)
    assert len(s) == 10
    intents = [e.intent for e in s]
    assert intents.count("lookup") == 8 and intents.count("procedure") == 2
    assert [e.id for e in gs.stratified_sample(10, seed=3)] == [e.id for e in s]


def test_validate_flags_label_problems():
    gs = GoldenSet(
        [
            _ex(1),
            GoldenExample(
                id="u1", question="question 1?", should_abstain=True, gold_doc_ids=["file:a.md"]
            ),
            GoldenExample(id="n1", question="no gold?"),
        ]
    )
    issues = gs.validate(corpus_doc_ids={"file:b.md"})
    msgs = [(i.severity, i.example_id, i.message) for i in issues]
    assert any(s == "error" and e == "u1" for s, e, _ in msgs)
    assert any(e == "u1" and "duplicate question" in m for _, e, m in msgs)
    assert any(s == "error" and "not in corpus" in m for s, _, m in msgs)
    assert any(e == "n1" and "no gold evidence" in m for _, e, m in msgs)


def test_duplicate_ids_and_bad_grades():
    gs = GoldenSet([_ex(1)])
    with pytest.raises(ValueError):
        gs.add(_ex(1))
    with pytest.raises(ValueError):
        GoldenExample(id="x", question="q", graded_relevance={"a": 9})


def test_relevance_levels_and_slices():
    ex = GoldenExample(
        id="x",
        question="q",
        gold_doc_ids=["d1"],
        gold_chunk_ids=["c1"],
        graded_relevance={"d1": 3, "c1": 2},
        slices=["multi_hop"],
        language="de",
    )
    assert ex.relevance("doc") == {"d1": 3}
    assert ex.relevance("chunk") == {"c1": 2}
    keys = ex.slice_keys()
    assert {"all", "intent:lookup", "tag:multi_hop", "lang:de", "answerable"} <= set(keys)


def test_sample_dataset_files_are_valid():
    gs = GoldenSet.load_many(sorted(DATA.glob("*.jsonl")))
    assert len(gs) >= 15
    assert not [i for i in gs.validate() if i.severity == "error"]
    docs = {d for e in gs for d in e.gold_doc_ids}
    assert docs <= {
        "file:refund-policy.md",
        "file:pto-policy.md",
        "file:security-policy.md",
        "file:expense-policy.md",
    }
    assert all(e.should_abstain for e in GoldenSet.load(DATA / "unanswerable.jsonl"))
