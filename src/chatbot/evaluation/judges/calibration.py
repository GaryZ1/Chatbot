"""Judge calibration against human labels (docs/12-evaluation.md §4).

"Label 100-300 items by hand; measure judge-human agreement (Cohen's κ,
accuracy). Iterate the prompt until κ > ~0.6-0.8." Supports nominal κ and
linear/quadratic weighted κ for ordinal 1-5 scales.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from typing import Literal

from pydantic import BaseModel

DEFAULT_KAPPA_TARGET = 0.6


def cohen_kappa(
    a: Sequence[Hashable],
    b: Sequence[Hashable],
    weights: Literal["none", "linear", "quadratic"] = "none",
) -> float:
    """Cohen's κ between two raters. Weighted variants require numeric labels.

    Returns 1.0 when both raters are constant and identical (perfect, degenerate
    agreement) and 0.0 when expected agreement is 1 but observed is not."""
    if len(a) != len(b):
        raise ValueError("label sequences must have equal length")
    n = len(a)
    if n == 0:
        raise ValueError("need at least one labelled pair")
    if weights == "none":
        po = sum(x == y for x, y in zip(a, b, strict=True)) / n
        ca, cb = Counter(a), Counter(b)
        pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
        if pe == 1.0:
            return 1.0 if po == 1.0 else 0.0
        return (po - pe) / (1 - pe)
    xs = [float(x) for x in a]  # type: ignore[arg-type]
    ys = [float(y) for y in b]  # type: ignore[arg-type]
    cats = sorted(set(xs) | set(ys))
    if len(cats) == 1:
        return 1.0
    idx = {c: i for i, c in enumerate(cats)}
    span = len(cats) - 1
    power = 1 if weights == "linear" else 2

    def w(i: int, j: int) -> float:
        return (abs(i - j) / span) ** power

    obs = sum(w(idx[x], idx[y]) for x, y in zip(xs, ys, strict=True)) / n
    cx, cy = Counter(xs), Counter(ys)
    exp = sum(cx[ci] * cy[cj] * w(idx[ci], idx[cj]) for ci in cx for cj in cy) / (n * n)
    if exp == 0:
        return 1.0
    return 1 - obs / exp


class JudgeCalibration(BaseModel):
    n: int
    accuracy: float
    kappa: float
    weights: str = "none"
    confusion: dict[str, dict[str, int]]
    target_kappa: float = DEFAULT_KAPPA_TARGET

    @property
    def passes(self) -> bool:
        return self.kappa >= self.target_kappa

    @classmethod
    def compute(
        cls,
        judge_labels: Sequence[Hashable],
        human_labels: Sequence[Hashable],
        weights: Literal["none", "linear", "quadratic"] = "none",
        target_kappa: float = DEFAULT_KAPPA_TARGET,
    ) -> JudgeCalibration:
        """Agreement of judge vs. human labels. Pairs where the judge returned
        ``None`` (failed) are dropped and should be reported separately."""
        pairs = [(j, h) for j, h in zip(judge_labels, human_labels, strict=True) if j is not None]
        if not pairs:
            raise ValueError("no valid judge/human pairs")
        js, hs = [p[0] for p in pairs], [p[1] for p in pairs]
        confusion: dict[str, dict[str, int]] = {}
        for j, h in pairs:
            row = confusion.setdefault(str(h), {})
            row[str(j)] = row.get(str(j), 0) + 1
        return cls(
            n=len(pairs),
            accuracy=sum(j == h for j, h in pairs) / len(pairs),
            kappa=cohen_kappa(js, hs, weights),
            weights=weights,
            confusion=confusion,
            target_kappa=target_kappa,
        )
