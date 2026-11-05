import math

import pytest

from chatbot.core.tokens import RegexTokenizer
from chatbot.evaluation.metrics import (
    ClaimSupport,
    CitedStatement,
    abstention_report,
    alce_citation_scores,
    average_precision,
    citation_scores,
    exact_match,
    hit_at_k,
    is_abstention,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieval_metrics,
    split_statements,
    token_f1,
    token_iou,
)


def test_ndcg_binary_known_value():
    ranked = ["x", "a", "y", "b"]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, {"a", "b"}, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(ranked, {"a", "b"}, 4) == pytest.approx(0.650921, abs=1e-6)


def test_ndcg_graded_known_value():
    ranked = ["b", "a"]
    grades = {"a": 3, "b": 1}
    dcg = 1 + 7 / math.log2(3)
    idcg = 7 + 1 / math.log2(3)
    assert ndcg_at_k(ranked, grades, 2) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], grades, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["z"], grades, 2) == 0.0
    assert ndcg_at_k(["a"], {}, 2) == 0.0


def test_recall_precision_hit_mrr():
    ranked = ["d1", "d2", "d1", "d3"]  # duplicate d1 removed
    rel = {"d2", "d3", "d9"}
    assert recall_at_k(ranked, rel, 2) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, rel, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, rel, 2) == pytest.approx(0.5)
    assert hit_at_k(ranked, rel, 1) == 0.0
    assert hit_at_k(ranked, rel, 2) == 1.0
    assert reciprocal_rank(ranked, rel) == pytest.approx(0.5)
    assert mrr([["a", "b"], ["b"]], [{"b"}, {"b"}]) == pytest.approx(0.75)
    assert average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx((1 + 2 / 3) / 2)
    with pytest.raises(ValueError):
        recall_at_k(ranked, rel, 0)


def test_retrieval_metrics_keys():
    m = retrieval_metrics(["a", "b"], {"a": 2}, ks=[1, 5])
    assert m["recall@1"] == 1.0 and m["ndcg@5"] == 1.0 and m["mrr"] == 1.0
    assert m["precision@5"] == pytest.approx(0.2)


def test_token_iou_chars_and_tokens():
    # gold chars 0-10, retrieved 5-15 → intersection 5, union 15
    res = token_iou([("d", 5, 15)], [("d", 0, 10)])
    assert res.iou == pytest.approx(5 / 15)
    assert res.recall == pytest.approx(0.5)
    assert res.precision == pytest.approx(0.5)
    # Redundant overlapping chunks are penalised in the denominator.
    red = token_iou([("d", 0, 10), ("d", 0, 10)], [("d", 0, 10)])
    assert red.iou == pytest.approx(10 / 20)
    nored = token_iou([("d", 0, 10), ("d", 0, 10)], [("d", 0, 10)], count_redundant=False)
    assert nored.iou == pytest.approx(1.0)
    text = "alpha beta gamma delta"
    tok = token_iou([("d", 0, 10)], [("d", 6, 16)], doc_texts={"d": text}, tokenizer=RegexTokenizer())
    # retrieved tokens: alpha, beta; gold: beta, gamma → IoU 1/3
    assert tok.iou == pytest.approx(1 / 3)


def test_lexical_answer_metrics():
    assert exact_match("The 30 days.", "30 days") == 1.0
    assert token_f1("refund within 30 days [1]", "30 days") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert token_f1("", "x") == 0.0


def test_abstention():
    assert is_abstention("I don't know based on the documents.")
    assert is_abstention("There is not enough information in the context.")
    assert not is_abstention("Refunds are allowed within 30 days [1].")
    rep = abstention_report([(True, True), (True, False), (False, True), (False, False)])
    assert rep.accuracy == 0.5 and rep.precision == 0.5 and rep.recall == 0.5
    assert rep.false_refusal_rate == 0.5


def test_split_statements_markers():
    st = split_statements("Refunds take 30 days [1]. PTO is 20 days [2, 3]. [4]")
    assert [s.markers for s in st] == [[1], [2, 3, 4]]
    assert st[0].text == "Refunds take 30 days ."


def test_alce_from_support():
    claims = [
        ClaimSupport(claim="a", cited_ids=["1", "2"], supporting_ids={"1"}),  # 2 is irrelevant
        ClaimSupport(claim="b", cited_ids=["3"], supporting_ids=set()),  # unsupported
        ClaimSupport(claim="c", cited_ids=["4", "5"], supporting_ids=set(), jointly_supported=True),
        ClaimSupport(claim="d", cited_ids=[], supporting_ids=set()),
    ]
    s = citation_scores(claims)
    assert s.citation_recall == pytest.approx(2 / 4)
    # precise: 1 (alone), 4 and 5 (needed jointly) → 3 of 5 citations
    assert s.citation_precision == pytest.approx(3 / 5)


async def test_alce_with_entailment_oracle():
    sources = {1: "refund 30 days", 2: "pto 20 days"}

    async def entails(claim: str, passages):
        return any(claim.split()[0] in p for p in passages)

    statements = [CitedStatement(text="refund window", markers=[1, 2]), CitedStatement(text="pto", markers=[1])]
    s = await alce_citation_scores(statements, sources, entails)
    assert s.citation_recall == pytest.approx(0.5)
    assert s.citation_precision == pytest.approx(1 / 3)
