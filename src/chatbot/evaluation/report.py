"""Reports and baseline comparisons (docs/12-evaluation.md §5, docs/16-build-plan.md
"Experiment log template").

:func:`compare` pairs baseline and candidate per example id and computes a paired
bootstrap CI for each metric (Holm-corrected across metrics), plus per-slice
deltas flagging regressions ("a +2% overall can hide -15% on one slice").
:func:`render_markdown` writes the experiment-log format to
``eval/reports/<date>-<name>.md``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from chatbot.core.types import utcnow
from chatbot.evaluation.runner import EvalResults
from chatbot.evaluation.stats import PairedDiff, holm_correction, paired_bootstrap

logger = logging.getLogger(__name__)

#: Headline metrics in the experiment-log template, in display order.
HEADLINE_METRICS: tuple[str, ...] = (
    "recall@10",
    "ndcg@10",
    "recall@5",
    "mrr",
    "faithfulness",
    "correct",
    "answer_f1",
    "abstention_correct",
    "citation_precision",
    "pass",
)
#: Metrics where lower is better.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "latency_ms",
        "cost_usd",
        "latency_p50_ms",
        "latency_p95_ms",
        "cost_per_query_usd",
        "error_rate",
        "false_refusal_rate",
    }
)
#: Minimum slice size to report a slice regression (tiny slices are noise).
MIN_SLICE_N = 5


class MetricComparison(BaseModel):
    metric: str
    diff: PairedDiff
    p_adjusted: float
    significant: bool
    improved: bool


class SliceDelta(BaseModel):
    slice: str
    metric: str
    n: int
    baseline: float
    candidate: float
    delta: float


class Comparison(BaseModel):
    baseline_name: str
    candidate_name: str
    n_paired: int
    metrics: dict[str, MetricComparison] = Field(default_factory=dict)
    slice_regressions: list[SliceDelta] = Field(default_factory=list)
    operational: dict[str, dict[str, float]] = Field(default_factory=dict)

    def regressed(self) -> list[str]:
        return [k for k, v in self.metrics.items() if v.significant and not v.improved]


def compare(
    baseline: EvalResults,
    candidate: EvalResults,
    metrics: list[str] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    slice_regression_threshold: float = 0.05,
    min_slice_n: int = MIN_SLICE_N,
) -> Comparison:
    """Paired comparison on the intersection of example ids."""
    if baseline.dataset_version != candidate.dataset_version:
        logger.warning(
            "comparing runs on different dataset versions (%s vs %s): only shared ids are paired",
            baseline.dataset_version,
            candidate.dataset_version,
        )
    b_by = {e.id: e for e in baseline.examples if e.error is None}
    c_by = {e.id: e for e in candidate.examples if e.error is None}
    shared = sorted(set(b_by) & set(c_by))
    if metrics is None:
        keys: set[str] = set()
        for i in shared:
            keys |= set(b_by[i].metrics) & set(c_by[i].metrics)
        metrics = sorted(keys)
    diffs: dict[str, PairedDiff] = {}
    for metric in metrics:
        ids = [i for i in shared if metric in b_by[i].metrics and metric in c_by[i].metrics]
        if not ids:
            continue
        diffs[metric] = paired_bootstrap(
            [b_by[i].metrics[metric] for i in ids],
            [c_by[i].metrics[metric] for i in ids],
            n_boot=n_boot,
            seed=seed,
        )
    holm = holm_correction({k: d.p_value for k, d in diffs.items()}, alpha=alpha)
    out = Comparison(
        baseline_name=baseline.name, candidate_name=candidate.name, n_paired=len(shared)
    )
    for k, d in diffs.items():
        better = d.delta < 0 if k in LOWER_IS_BETTER else d.delta > 0
        out.metrics[k] = MetricComparison(
            metric=k,
            diff=d,
            p_adjusted=holm.adjusted[k],
            significant=holm.reject[k] and d.significant,
            improved=better,
        )
    # Slice regressions on aggregate means (descriptive; small slices skipped).
    for s, bm in baseline.aggregate.items():
        cm = candidate.aggregate.get(s)
        n = min(baseline.slice_sizes.get(s, 0), candidate.slice_sizes.get(s, 0))
        if cm is None or n < min_slice_n:
            continue
        for metric in metrics:
            if metric not in bm or metric not in cm:
                continue
            delta = cm[metric] - bm[metric]
            worse = (
                delta > slice_regression_threshold
                if metric in LOWER_IS_BETTER
                else delta < -slice_regression_threshold
            )
            if worse:
                out.slice_regressions.append(
                    SliceDelta(
                        slice=s,
                        metric=metric,
                        n=n,
                        baseline=bm[metric],
                        candidate=cm[metric],
                        delta=delta,
                    )
                )
    out.slice_regressions.sort(key=lambda d: d.delta)
    for key in ("latency_p50_ms", "latency_p95_ms", "cost_per_query_usd", "error_rate"):
        bv, cv = baseline.metric(key), candidate.metric(key)
        if bv is not None and cv is not None:
            out.operational[key] = {"baseline": bv, "candidate": cv, "delta": cv - bv}
    return out


# --------------------------------------------------------------------------- rendering


def _fmt(v: float | None, metric: str = "") -> str:
    if v is None:
        return "–"
    if metric.endswith("_ms"):
        return f"{v:.0f} ms"
    if "usd" in metric:
        return f"${v:.4f}"
    return f"{v:.3f}"


class ExperimentLog(BaseModel):
    """Free-text fields of the experiment-log template."""

    hypothesis: str = ""
    change: str = ""
    decision: str = ""  # adopt / reject / iterate


def render_markdown(
    results: EvalResults,
    comparison: Comparison | None = None,
    baseline: EvalResults | None = None,
    log: ExperimentLog | None = None,
    max_failures: int = 20,
) -> str:
    log = log or ExperimentLog()
    lines: list[str] = [f"# Experiment: {results.name}", ""]
    lines.append(f"Hypothesis: {log.hypothesis or '(not stated)'}")
    lines.append(f"Change (config diff): {log.change or '(not stated)'}")
    lines.append(f"Dataset/split: {results.dataset_name} {results.dataset_version} (n={results.n})")
    lines.append(f"Run: {results.started_at} · {results.duration_s:.1f}s")
    lines.append("")
    lines.append("## Metrics (baseline → new, 95% CI)")
    lines.append("")
    allm = results.aggregate.get("all", {})
    headline = [m for m in HEADLINE_METRICS if m in allm]
    ops = ["latency_p95_ms", "cost_per_query_usd"]
    if comparison is not None and baseline is not None:
        lines.append("| Metric | Baseline | New | Δ (95% CI) | p (Holm) |")
        lines.append("|---|---|---|---|---|")
        for m in headline:
            mc = comparison.metrics.get(m)
            if mc is None:
                lines.append(f"| {m} | {_fmt(baseline.metric(m))} | {_fmt(allm.get(m))} | – | – |")
                continue
            d = mc.diff
            flag = " ✅" if mc.significant and mc.improved else (" ❌" if mc.significant else "")
            lines.append(
                f"| {m} | {_fmt(d.baseline_mean)} | {_fmt(d.candidate_mean)} | "
                f"{d.delta:+.3f} [{d.low:+.3f}, {d.high:+.3f}]{flag} | {mc.p_adjusted:.3f} |"
            )
        for m in ops:
            lines.append(
                f"| {m} | {_fmt(baseline.metric(m), m)} | {_fmt(allm.get(m), m)} | – | – |"
            )
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for m in [*headline, *ops]:
            lines.append(f"| {m} | {_fmt(allm.get(m), m)} |")
    lines.append("")

    lines.append("## Slice results")
    lines.append("")
    slice_metrics = [
        m
        for m in ("recall@10", "recall@5", "faithfulness", "correct", "answer_f1", "pass")
        if m in allm
    ][:4]
    lines.append("| Slice | n | " + " | ".join(slice_metrics) + " |")
    lines.append("|---|---|" + "---|" * len(slice_metrics))
    for s, n in sorted(results.slice_sizes.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        vals = results.aggregate.get(s, {})
        lines.append(
            f"| {s} | {n} | " + " | ".join(_fmt(vals.get(m)) for m in slice_metrics) + " |"
        )
    if comparison is not None and comparison.slice_regressions:
        lines.append("")
        lines.append("**Slice regressions:**")
        for d in comparison.slice_regressions:
            lines.append(
                f"- `{d.slice}` {d.metric}: {d.baseline:.3f} → {d.candidate:.3f} ({d.delta:+.3f}, n={d.n})"
            )
    lines.append("")

    lines.append("## Retrieval funnel")
    lines.append("")
    f = results.funnel
    if f.get("n"):
        lines.append(f"Answerable examples with gold evidence: {int(f['n'])}")
        for key in ("gold_in_candidates", "gold_in_final", "gold_in_context", "answered_correctly"):
            if key in f:
                lines.append(f"- {key}: {f[key]:.1%}")
    else:
        lines.append("No answerable examples with gold evidence.")
    lines.append("")

    lines.append("## Latency & cost")
    lines.append("")
    lines.append("| Stage | p50 | p95 | mean |")
    lines.append("|---|---|---|---|")
    for stage, v in results.latency.items():
        lines.append(f"| {stage} | {v['p50']:.0f} ms | {v['p95']:.0f} ms | {v['mean']:.0f} ms |")
    lines.append(
        f"\n$/query: {_fmt(results.cost.get('per_query_usd'), 'usd')} · total {_fmt(results.cost.get('total_usd'), 'usd')}"
    )
    lines.append("")

    failures = results.failures()
    lines.append(f"## Failure analysis (F-codes, {min(len(failures), max_failures)} samples)")
    lines.append("")
    if results.failure_counts:
        lines.append(", ".join(f"{k}: {v}" for k, v in results.failure_counts.items()))
        lines.append("")
    for e in failures[:max_failures]:
        code = e.error or e.failure_code or e.failure_stage or "?"
        note = f" — {'; '.join(e.notes)}" if e.notes else ""
        lines.append(f"- `{e.id}` [{code}] {e.question}{note}")
    lines.append("")
    lines.append(f"Decision: {log.decision or 'adopt / reject / iterate'}")
    return "\n".join(lines) + "\n"


def write_report(
    results: EvalResults,
    out_dir: str | Path,
    comparison: Comparison | None = None,
    baseline: EvalResults | None = None,
    log: ExperimentLog | None = None,
) -> tuple[Path, Path]:
    """Write ``<date>-<name>.md`` and ``<date>-<name>.json`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    slug = (
        "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in results.name).strip("-")
        or "run"
    )
    stem = f"{utcnow().date().isoformat()}-{slug}"
    md = out / f"{stem}.md"
    md.write_text(render_markdown(results, comparison, baseline, log), encoding="utf-8")
    js = out / f"{stem}.json"
    payload = {"results": results.model_dump(mode="json")}
    if comparison is not None:
        payload["comparison"] = comparison.model_dump(mode="json")
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md, js
