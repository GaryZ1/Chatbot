"""One-call entry point for a ``chatbot eval`` command (docs/12-evaluation.md §5).

``run_eval`` = load golden set → validate labels → pick the frozen split → run the
system → (optional) compare with a baseline → write the experiment-log report →
(optional) apply the CI gate and return an exit code.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from chatbot.evaluation.dataset import GoldenSet, LabelIssue
from chatbot.evaluation.gate import GateConfig, GateResult, RegressionGate
from chatbot.evaluation.report import Comparison, ExperimentLog, compare, write_report
from chatbot.evaluation.runner import (
    EvalConfig,
    EvalResults,
    EvalRunner,
    JudgeSuite,
    SystemUnderTest,
)
from chatbot.llm.base import LLM

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path("eval/reports")


class EvalOutcome(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    results: EvalResults
    comparison: Comparison | None = None
    gate: GateResult | None = None
    label_issues: list[LabelIssue] = []
    report_path: Path | None = None
    json_path: Path | None = None

    @property
    def exit_code(self) -> int:
        return self.gate.exit_code if self.gate is not None else 0


def _load_dataset(dataset: str | Path | GoldenSet | Sequence[str | Path]) -> GoldenSet:
    if isinstance(dataset, GoldenSet):
        return dataset
    if isinstance(dataset, str | Path):
        return GoldenSet.load(dataset)
    paths = list(dataset)
    return GoldenSet.load_many(paths, name="+".join(Path(p).stem for p in paths))


def _load_gate(gate: GateConfig | dict[str, Any] | str | Path | None) -> GateConfig | None:
    if gate is None or isinstance(gate, GateConfig):
        return gate
    if isinstance(gate, dict):
        return GateConfig.from_dict(gate)
    return GateConfig.from_yaml(gate)


async def run_eval(
    system: SystemUnderTest,
    dataset: str | Path | GoldenSet | Sequence[str | Path],
    *,
    name: str = "candidate",
    split: Literal["dev", "test", "all"] = "dev",
    allow_test: bool = False,
    config: EvalConfig | None = None,
    judges: JudgeSuite | None = None,
    judge_llm: LLM | None = None,
    baseline: str | Path | EvalResults | None = None,
    gate: GateConfig | dict[str, Any] | str | Path | None = None,
    out_dir: str | Path | None = DEFAULT_REPORT_DIR,
    sample: int | None = None,
    seed: int = 0,
    log: ExperimentLog | None = None,
    fail_on_label_errors: bool = True,
) -> EvalOutcome:
    """Run an offline evaluation end-to-end.

    * ``split="test"`` requires ``allow_test=True``: the frozen test split is for final
      confirmation only, never for iteration (docs/12-evaluation.md §3.5).
    * ``judges`` overrides ``judge_llm``; with only ``judge_llm`` the default
      faithfulness + correctness judges are used; with neither, only label-based metrics.
    * ``sample`` draws a stratified (by intent) subset for quick smoke runs.
    """
    gs = _load_dataset(dataset)
    issues = gs.validate(corpus_doc_ids=config.corpus_doc_ids if config else None)
    errors = [i for i in issues if i.severity == "error"]
    for issue in issues:
        log_fn = logger.error if issue.severity == "error" else logger.warning
        log_fn(
            "label issue [%s] %s: %s",
            issue.severity,
            issue.example_id or "<dataset>",
            issue.message,
        )
    if errors and fail_on_label_errors:
        raise ValueError(
            f"golden set has {len(errors)} label errors; fix them or pass fail_on_label_errors=False"
        )

    if split == "test" and not allow_test:
        raise ValueError(
            "the test split is frozen; pass allow_test=True for a final confirmation run"
        )
    subset = gs if split == "all" else gs.split(split)
    if sample is not None:
        subset = subset.stratified_sample(sample, by="intent", seed=seed)
    if len(subset) == 0:
        raise ValueError(f"no examples in split {split!r}")

    if judges is None and judge_llm is not None:
        judges = JudgeSuite.default(judge_llm)
    runner = EvalRunner(system, config=config, judges=judges, name=name)
    results = await runner.run(subset)

    base: EvalResults | None
    if baseline is None or isinstance(baseline, EvalResults):
        base = baseline
    else:
        base = EvalResults.load_json(baseline)
    comparison = compare(base, results, seed=seed) if base is not None else None

    gate_cfg = _load_gate(gate)
    gate_result = RegressionGate(gate_cfg).evaluate(results, base, comparison) if gate_cfg else None

    md_path = js_path = None
    if out_dir is not None:
        if gate_result is not None:
            log = (log or ExperimentLog()).model_copy()
            if not log.decision:
                log.decision = (
                    "gate passed"
                    if gate_result.passed
                    else "gate FAILED: " + "; ".join(gate_result.reasons)
                )
        md_path, js_path = write_report(results, out_dir, comparison, base, log)
        logger.info("eval report written to %s", md_path)
    return EvalOutcome(
        results=results,
        comparison=comparison,
        gate=gate_result,
        label_issues=issues,
        report_path=md_path,
        json_path=js_path,
    )
