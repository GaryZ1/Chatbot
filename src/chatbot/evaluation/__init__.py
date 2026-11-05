"""Evaluation harness (docs/12-evaluation.md): golden sets, metrics, judges,
statistics, runner, reports and the CI regression gate."""

from chatbot.evaluation.cli import EvalOutcome, run_eval
from chatbot.evaluation.dataset import (
    Difficulty,
    ExampleSource,
    GoldenExample,
    GoldenSet,
    LabelIssue,
    Split,
    split_of,
)
from chatbot.evaluation.gate import GateConfig, GateResult, GateRule, RegressionGate
from chatbot.evaluation.report import (
    Comparison,
    ExperimentLog,
    compare,
    render_markdown,
    write_report,
)
from chatbot.evaluation.runner import (
    EvalConfig,
    EvalResults,
    EvalRunner,
    EvalTrace,
    ExampleResult,
    JudgeSuite,
    SystemUnderTest,
)
from chatbot.evaluation.stats import holm_correction, mean_ci, paired_bootstrap
from chatbot.evaluation.synthetic import Evolution, SyntheticConfig, SyntheticGenerator

__all__ = [
    "Comparison",
    "Difficulty",
    "EvalConfig",
    "EvalOutcome",
    "EvalResults",
    "EvalRunner",
    "EvalTrace",
    "Evolution",
    "ExampleResult",
    "ExampleSource",
    "ExperimentLog",
    "GateConfig",
    "GateResult",
    "GateRule",
    "GoldenExample",
    "GoldenSet",
    "JudgeSuite",
    "LabelIssue",
    "RegressionGate",
    "Split",
    "SyntheticConfig",
    "SyntheticGenerator",
    "SystemUnderTest",
    "compare",
    "holm_correction",
    "mean_ci",
    "paired_bootstrap",
    "render_markdown",
    "run_eval",
    "split_of",
    "write_report",
]
