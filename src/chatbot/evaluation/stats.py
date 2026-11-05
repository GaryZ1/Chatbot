"""Statistics for offline comparisons (docs/12-evaluation.md §5, "Statistical rigor").

* :func:`mean_ci` — bootstrap percentile CI of a mean.
* :func:`paired_bootstrap` — paired bootstrap over questions for ``candidate -
  baseline``; resampling the *same* question indices for both systems removes
  between-question variance, which is what makes ~5-point differences detectable
  on a 200-500 question set.
* :func:`holm_correction` — Holm-Bonferroni step-down adjustment when many metrics
  or slices are compared at once ("beware multiple comparisons").

All functions are deterministic for a given ``seed``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel

DEFAULT_N_BOOT = 10_000
DEFAULT_CONFIDENCE = 0.95


class MeanCI(BaseModel):
    mean: float
    low: float
    high: float
    n: int


class PairedDiff(BaseModel):
    baseline_mean: float
    candidate_mean: float
    delta: float
    low: float
    high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """CI excludes zero."""
        return self.low > 0 or self.high < 0


def _arr(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if np.isnan(a).any():
        raise ValueError("values contain NaN; drop missing examples first")
    return a


def percentile(values: Sequence[float], q: float) -> float:
    if not len(values):
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def mean_ci(
    values: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> MeanCI:
    a = _arr(values)
    n = len(a)
    if n == 0:
        return MeanCI(mean=0.0, low=0.0, high=0.0, n=0)
    if n == 1:
        v = float(a[0])
        return MeanCI(mean=v, low=v, high=v, n=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = a[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    return MeanCI(
        mean=float(a.mean()),
        low=float(np.quantile(means, alpha)),
        high=float(np.quantile(means, 1 - alpha)),
        n=n,
    )


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    confidence: float = DEFAULT_CONFIDENCE,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
) -> PairedDiff:
    """Bootstrap CI and two-sided p-value for ``mean(candidate - baseline)``.

    The p-value is the two-sided bootstrap estimate ``2 * min(P(d* <= 0), P(d* >= 0))``
    (clipped to 1), a standard approximation for paired bootstrap tests."""
    b, c = _arr(baseline), _arr(candidate)
    if len(b) != len(c):
        raise ValueError("paired samples must have equal length")
    n = len(b)
    if n == 0:
        return PairedDiff(
            baseline_mean=0, candidate_mean=0, delta=0, low=0, high=0, p_value=1.0, n=0
        )
    d = c - b
    delta = float(d.mean())
    if n == 1 or np.allclose(d, d[0]):
        p = 1.0 if delta == 0 else 0.0
        return PairedDiff(
            baseline_mean=float(b.mean()),
            candidate_mean=float(c.mean()),
            delta=delta,
            low=delta,
            high=delta,
            p_value=p if n > 1 else 1.0,
            n=n,
        )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    p = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return PairedDiff(
        baseline_mean=float(b.mean()),
        candidate_mean=float(c.mean()),
        delta=delta,
        low=float(np.quantile(boots, alpha)),
        high=float(np.quantile(boots, 1 - alpha)),
        p_value=min(1.0, p),
        n=n,
    )


class HolmResult(BaseModel):
    adjusted: dict[str, float]
    reject: dict[str, bool]


def holm_correction(
    p_values: Mapping[str, float] | Sequence[float], alpha: float = 0.05
) -> HolmResult:
    """Holm-Bonferroni step-down. Adjusted p-values are monotone and capped at 1."""
    items = (
        list(p_values.items())
        if isinstance(p_values, Mapping)
        else [(str(i), p) for i, p in enumerate(p_values)]
    )
    m = len(items)
    order = sorted(items, key=lambda kv: kv[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(order):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        adjusted[key] = running
    return HolmResult(adjusted=adjusted, reject={k: v <= alpha for k, v in adjusted.items()})
