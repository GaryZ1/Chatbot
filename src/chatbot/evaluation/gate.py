"""CI regression gate (docs/12-evaluation.md §5: "a PR fails if faithfulness or
Recall@k drops more than X points on the regression set").

Rules are expressed in the metric's own units (``max_drop: 0.02`` = 2 points on a
0-1 metric). For lower-is-better metrics (latency, cost, error rate) a "drop" is
an *increase*. Configure from a dict or YAML::

    rules:
      - {metric: "recall@10", max_drop: 0.02}
      - {metric: faithfulness, max_drop: 0.02, required: true}
      - {metric: abstention_accuracy, slice: unanswerable, min_value: 0.8}
      - {metric: latency_p95_ms, max_relative_drop: 0.2}
    fail_on_slice_regression: true
    slice_max_drop: 0.10

or the shorthand ``{"recall@10": {"max_drop": 0.02}, ...}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from chatbot.evaluation.report import LOWER_IS_BETTER, MIN_SLICE_N, Comparison, compare
from chatbot.evaluation.runner import EvalResults

logger = logging.getLogger(__name__)


class GateRule(BaseModel):
    metric: str
    slice: str = "all"
    min_value: float | None = None
    max_value: float | None = None
    #: Max allowed absolute degradation vs. baseline (metric units).
    max_drop: float | None = None
    #: Max allowed degradation as a fraction of the baseline value.
    max_relative_drop: float | None = None
    #: Only fail a drop rule when the paired bootstrap CI excludes zero ("all" slice only).
    require_significant: bool = False
    #: Fail if the metric is missing from the candidate results.
    required: bool = False
    lower_is_better: bool | None = None

    @model_validator(mode="after")
    def _has_threshold(self) -> GateRule:
        if all(
            v is None
            for v in (self.min_value, self.max_value, self.max_drop, self.max_relative_drop)
        ):
            raise ValueError(f"gate rule for {self.metric!r} has no threshold")
        return self

    @property
    def lower_better(self) -> bool:
        return (
            self.lower_is_better
            if self.lower_is_better is not None
            else self.metric in LOWER_IS_BETTER
        )


class GateConfig(BaseModel):
    rules: list[GateRule] = Field(default_factory=list)
    fail_on_slice_regression: bool = False
    slice_max_drop: float = 0.10
    slice_metrics: list[str] = Field(
        default_factory=lambda: ["recall@10", "faithfulness", "correct", "pass"]
    )
    min_slice_n: int = MIN_SLICE_N
    #: Upper bound on the run error rate (system crashes / timeouts).
    max_error_rate: float | None = 0.05

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateConfig:
        if "rules" in data or any(k in cls.model_fields for k in data):
            return cls.model_validate(data)
        rules = [GateRule(metric=k, **(v or {})) for k, v in data.items()]
        return cls(rules=rules)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GateConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"gate config {path} must be a mapping")
        if "gate" in data and isinstance(data["gate"], dict):
            data = data["gate"]
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> GateConfig:
        """Regression-set defaults: Recall@10 and faithfulness may not drop > 2 points,
        abstention accuracy may not drop > 5 points."""
        return cls(
            rules=[
                GateRule(metric="recall@10", max_drop=0.02),
                GateRule(metric="faithfulness", max_drop=0.02),
                GateRule(metric="abstention_accuracy", max_drop=0.05),
            ]
        )


class GateCheck(BaseModel):
    rule: str
    passed: bool
    detail: str


class GateResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    checks: list[GateCheck] = Field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1


def _describe(rule: GateRule) -> str:
    parts = [rule.metric if rule.slice == "all" else f"{rule.metric}[{rule.slice}]"]
    for name in ("min_value", "max_value", "max_drop", "max_relative_drop"):
        v = getattr(rule, name)
        if v is not None:
            parts.append(f"{name}={v}")
    return " ".join(parts)


class RegressionGate:
    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig.default()

    def evaluate(
        self,
        candidate: EvalResults,
        baseline: EvalResults | None = None,
        comparison: Comparison | None = None,
    ) -> GateResult:
        cfg = self.config
        checks: list[GateCheck] = []
        needs_sig = baseline is not None and any(r.require_significant for r in cfg.rules)
        if needs_sig and comparison is None and baseline is not None:
            comparison = compare(baseline, candidate, metrics=[r.metric for r in cfg.rules])

        for rule in cfg.rules:
            desc = _describe(rule)
            value = candidate.metric(rule.metric, rule.slice)
            if value is None:
                checks.append(
                    GateCheck(
                        rule=desc,
                        passed=not rule.required,
                        detail="metric missing"
                        + (" (required)" if rule.required else " (skipped)"),
                    )
                )
                continue
            lb = rule.lower_better
            if rule.min_value is not None and value < rule.min_value:
                checks.append(
                    GateCheck(rule=desc, passed=False, detail=f"{value:.4f} < min {rule.min_value}")
                )
                continue
            if rule.max_value is not None and value > rule.max_value:
                checks.append(
                    GateCheck(rule=desc, passed=False, detail=f"{value:.4f} > max {rule.max_value}")
                )
                continue
            if baseline is not None and (
                rule.max_drop is not None or rule.max_relative_drop is not None
            ):
                base = baseline.metric(rule.metric, rule.slice)
                if base is None:
                    checks.append(
                        GateCheck(rule=desc, passed=True, detail="no baseline value (skipped)")
                    )
                    continue
                drop = (value - base) if lb else (base - value)
                limit_abs = rule.max_drop if rule.max_drop is not None else float("inf")
                limit_rel = (
                    rule.max_relative_drop * abs(base)
                    if rule.max_relative_drop is not None
                    else float("inf")
                )
                limit = min(limit_abs, limit_rel)
                failed = drop > limit + 1e-12
                if (
                    failed
                    and rule.require_significant
                    and comparison is not None
                    and rule.slice == "all"
                ):
                    mc = comparison.metrics.get(rule.metric)
                    if mc is not None and not mc.significant:
                        checks.append(
                            GateCheck(
                                rule=desc,
                                passed=True,
                                detail=f"drop {drop:.4f} > {limit:.4f} but not significant (p={mc.p_adjusted:.3f})",
                            )
                        )
                        continue
                checks.append(
                    GateCheck(
                        rule=desc,
                        passed=not failed,
                        detail=f"baseline {base:.4f} → {value:.4f} (degradation {drop:+.4f}, limit {limit:.4f})",
                    )
                )
                continue
            checks.append(GateCheck(rule=desc, passed=True, detail=f"value {value:.4f}"))

        if cfg.max_error_rate is not None:
            err = candidate.metric("error_rate") or 0.0
            checks.append(
                GateCheck(
                    rule=f"error_rate max_value={cfg.max_error_rate}",
                    passed=err <= cfg.max_error_rate,
                    detail=f"error rate {err:.2%}",
                )
            )

        if cfg.fail_on_slice_regression and baseline is not None:
            for s, cm in candidate.aggregate.items():
                bm = baseline.aggregate.get(s)
                n = min(candidate.slice_sizes.get(s, 0), baseline.slice_sizes.get(s, 0))
                if bm is None or n < cfg.min_slice_n:
                    continue
                for metric in cfg.slice_metrics:
                    if (
                        metric in cm
                        and metric in bm
                        and bm[metric] - cm[metric] > cfg.slice_max_drop
                    ):
                        checks.append(
                            GateCheck(
                                rule=f"slice {s} {metric} max_drop={cfg.slice_max_drop}",
                                passed=False,
                                detail=f"{bm[metric]:.3f} → {cm[metric]:.3f} (n={n})",
                            )
                        )

        reasons = [f"{c.rule}: {c.detail}" for c in checks if not c.passed]
        result = GateResult(passed=not reasons, reasons=reasons, checks=checks)
        if reasons:
            logger.warning("eval gate FAILED: %s", "; ".join(reasons))
        else:
            logger.info("eval gate passed (%d checks)", len(checks))
        return result
