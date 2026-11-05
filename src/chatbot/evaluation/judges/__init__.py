"""LLM-as-judge (docs/12-evaluation.md §4)."""

from chatbot.evaluation.judges.base import Judge, JudgeConfig, xml_block
from chatbot.evaluation.judges.calibration import JudgeCalibration, cohen_kappa
from chatbot.evaluation.judges.pairwise import PairwiseJudge, PairwiseResult, win_rate
from chatbot.evaluation.judges.rag import (
    AnswerRelevanceJudge,
    CitationSupportJudge,
    ClaimVerdict,
    ContextJudge,
    CorrectnessJudge,
    CorrectnessVerdict,
    FaithfulnessJudge,
    FaithfulnessResult,
    RelevanceVerdict,
)

__all__ = [
    "AnswerRelevanceJudge",
    "CitationSupportJudge",
    "ClaimVerdict",
    "ContextJudge",
    "CorrectnessJudge",
    "CorrectnessVerdict",
    "FaithfulnessJudge",
    "FaithfulnessResult",
    "Judge",
    "JudgeCalibration",
    "JudgeConfig",
    "PairwiseJudge",
    "PairwiseResult",
    "RelevanceVerdict",
    "cohen_kappa",
    "win_rate",
    "xml_block",
]
