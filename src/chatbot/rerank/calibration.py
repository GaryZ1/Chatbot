"""Abstention threshold calibration (docs/08-reranking.md §3 rule 5, docs/07 §4).

Fit τ on labelled ``(max_score, answerable)`` pairs from the evaluation set: the
system *answers* when the best reranker score is ``≥ τ`` and abstains ("I don't
know" / clarify) otherwise. Objectives:

* ``"f1"`` — maximise F1 of the answer decision (positive = answerable);
* ``"precision"`` — the lowest τ (max recall) whose precision ≥ ``target_precision``,
  for deployments where a wrong answer is far costlier than an abstention.

Thresholds are model-specific: re-fit whenever the reranker changes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from chatbot.core.types import ScoredChunk

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdFit:
    tau: float
    precision: float
    recall: float
    f1: float
    answer_rate: float
    support: int
    objective: str


def _metrics(pairs: Sequence[tuple[float, bool]], tau: float) -> tuple[float, float, float, float]:
    tp = sum(1 for s, a in pairs if s >= tau and a)
    fp = sum(1 for s, a in pairs if s >= tau and not a)
    fn = sum(1 for s, a in pairs if s < tau and a)
    answered = tp + fp
    precision = tp / answered if answered else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, answered / len(pairs)


def fit_threshold(
    pairs: Iterable[tuple[float, bool]],
    *,
    objective: Literal["f1", "precision"] = "f1",
    target_precision: float = 0.9,
) -> ThresholdFit:
    """Choose τ among observed scores (plus "always abstain") for the given objective."""
    data = [(float(s), bool(a)) for s, a in pairs]
    if not data:
        raise ValueError("need at least one labelled pair")
    if any(math.isnan(s) for s, _ in data):
        raise ValueError("scores must not be NaN")
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    scores = sorted({s for s, _ in data})
    candidates = [*scores, math.nextafter(scores[-1], math.inf)]  # last = always abstain
    fits = []
    for tau in candidates:
        p, r, f1, rate = _metrics(data, tau)
        fits.append(ThresholdFit(tau, p, r, f1, rate, len(data), objective))
    if objective == "f1":
        # Highest F1; ties → higher τ (more conservative).
        return max(fits, key=lambda f: (f.f1, f.tau))
    if objective != "precision":
        raise ValueError(f"unknown objective {objective!r}")
    ok = [f for f in fits if f.precision >= target_precision and f.answer_rate > 0]
    if not ok:
        log.warning(
            "no threshold reaches precision %.2f; calibrated to always abstain", target_precision
        )
        return fits[-1]
    return max(ok, key=lambda f: (f.recall, -f.tau))


@dataclass(frozen=True)
class GateDecision:
    abstain: bool
    max_score: float | None
    tau: float
    reason: str


class ThresholdGate:
    """Decides abstention from the best (reranker) score of the final candidates."""

    def __init__(self, tau: float, *, stage: str | None = None) -> None:
        self.tau = tau
        self.stage = stage

    @classmethod
    def from_fit(cls, fit: ThresholdFit, *, stage: str | None = None) -> ThresholdGate:
        return cls(fit.tau, stage=stage)

    def _score(self, hit: ScoredChunk) -> float | None:
        if self.stage is None:
            return hit.score
        return hit.stage_scores.get(self.stage)

    def max_score(self, hits: Sequence[ScoredChunk]) -> float | None:
        vals = [s for s in (self._score(h) for h in hits) if s is not None]
        return max(vals) if vals else None

    def decide(self, hits: Sequence[ScoredChunk]) -> GateDecision:
        best = self.max_score(hits)
        if best is None:
            return GateDecision(True, None, self.tau, "no candidates")
        if best < self.tau:
            return GateDecision(True, best, self.tau, f"max score {best:.3f} < tau {self.tau:.3f}")
        return GateDecision(False, best, self.tau, "ok")

    def should_abstain(self, hits: Sequence[ScoredChunk]) -> bool:
        return self.decide(hits).abstain

    def filter(self, hits: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Keep only hits at or above τ."""
        return [h for h in hits if (s := self._score(h)) is not None and s >= self.tau]
