import pytest

from chatbot.evaluation.judges import JudgeCalibration, cohen_kappa
from chatbot.evaluation.stats import holm_correction, mean_ci, paired_bootstrap


def test_paired_bootstrap_deterministic_with_seed():
    base = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0] * 5
    cand = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0] * 5
    a = paired_bootstrap(base, cand, n_boot=2000, seed=7)
    b = paired_bootstrap(base, cand, n_boot=2000, seed=7)
    assert a == b
    assert a.delta == pytest.approx(0.3)
    assert a.low > 0 and a.significant
    c = paired_bootstrap(base, cand, n_boot=2000, seed=8)
    assert c.delta == a.delta


def test_paired_bootstrap_no_difference():
    x = [0.5, 0.2, 0.9]
    r = paired_bootstrap(x, x)
    assert r.delta == 0 and r.p_value == 1.0 and not r.significant
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])


def test_mean_ci_contains_mean():
    ci = mean_ci([0.1, 0.4, 0.5, 0.9, 0.3], n_boot=1000, seed=1)
    assert ci.low <= ci.mean <= ci.high
    assert mean_ci([0.1, 0.4, 0.5, 0.9, 0.3], n_boot=1000, seed=1) == ci


def test_holm():
    res = holm_correction({"a": 0.01, "b": 0.04, "c": 0.03}, alpha=0.05)
    assert res.adjusted["a"] == pytest.approx(0.03)
    assert res.adjusted["c"] == pytest.approx(0.06)
    assert res.adjusted["b"] == pytest.approx(0.06)  # monotone
    assert res.reject == {"a": True, "c": False, "b": False}


def test_cohen_kappa_known_values():
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)
    assert cohen_kappa(["y", "n"], ["y", "n"]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1], [1, 1]) == 1.0
    assert cohen_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], weights="quadratic") == pytest.approx(1.0)
    assert cohen_kappa([1, 2, 3], [3, 2, 1], weights="linear") < 0


def test_judge_calibration():
    cal = JudgeCalibration.compute([True, True, False, False, None], [True, False, False, False, True])
    assert cal.n == 4
    assert cal.accuracy == pytest.approx(0.75)
    assert cal.kappa == pytest.approx(0.5)
    assert not cal.passes or cal.kappa >= 0.6
    assert cal.confusion["False"]["True"] == 1
