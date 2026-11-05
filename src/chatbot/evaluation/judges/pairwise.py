"""Pairwise preference judge with position-bias control (docs/12-evaluation.md §4).

"Prefer pairwise or binary judgments over fine-grained scores" and "randomize the
order in pairwise". The presentation order is drawn from a seeded RNG keyed by the
item id, so runs are reproducible; ``both_orders=True`` asks twice (A/B and B/A)
and only declares a winner when both orderings agree (otherwise a tie).
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from pydantic import BaseModel

from chatbot.evaluation.judges.base import Judge, JudgeConfig, xml_block
from chatbot.llm.base import LLM

PAIRWISE_SYSTEM = """You compare two answers to the same question and pick the better one.
Criteria, in priority order: (1) factual correctness and faithfulness to the provided reference/context,
(2) completeness on the key points, (3) clear citations where required, (4) concision.
Do NOT prefer an answer because it is longer or appears first. Ties are allowed when they are equally good.
Write your reasoning first, then the verdict.
Return JSON: {"reasoning": "...", "winner": "first|second|tie"}."""

Winner = Literal["a", "b", "tie"]


class PairwiseVerdict(BaseModel):
    reasoning: str = ""
    winner: Literal["first", "second", "tie"]


class PairwiseResult(BaseModel):
    winner: Winner
    consistent: bool = True
    a_first: bool
    reasoning: str = ""


class PairwiseJudge(Judge):
    def __init__(
        self,
        llm: LLM,
        config: JudgeConfig | None = None,
        seed: int = 0,
        both_orders: bool = False,
    ) -> None:
        super().__init__(llm, config)
        self.seed = seed
        self.both_orders = both_orders

    def _a_first(self, item_id: str) -> bool:
        digest = hashlib.sha256(f"{self.seed}|{item_id}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big")).random() < 0.5

    async def _ask(self, question: str, first: str, second: str, reference: str | None) -> PairwiseVerdict | None:
        parts = [xml_block("question", question)]
        if reference:
            parts.append(xml_block("reference", reference))
        parts += [xml_block("answer_first", first), xml_block("answer_second", second)]
        return await self._parse(PAIRWISE_SYSTEM, "\n\n".join(parts), PairwiseVerdict)

    @staticmethod
    def _map(v: PairwiseVerdict, a_first: bool) -> Winner:
        if v.winner == "tie":
            return "tie"
        first_won = v.winner == "first"
        return "a" if first_won == a_first else "b"

    async def compare(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        reference: str | None = None,
        item_id: str = "",
    ) -> PairwiseResult | None:
        a_first = self._a_first(item_id or question)
        first, second = (answer_a, answer_b) if a_first else (answer_b, answer_a)
        v1 = await self._ask(question, first, second, reference)
        if v1 is None:
            return None
        w1 = self._map(v1, a_first)
        if not self.both_orders:
            return PairwiseResult(winner=w1, a_first=a_first, reasoning=v1.reasoning)
        v2 = await self._ask(question, second, first, reference)
        if v2 is None:
            return PairwiseResult(winner=w1, a_first=a_first, reasoning=v1.reasoning, consistent=False)
        w2 = self._map(v2, not a_first)
        if w1 == w2:
            return PairwiseResult(winner=w1, a_first=a_first, reasoning=v1.reasoning)
        return PairwiseResult(winner="tie", a_first=a_first, reasoning=v1.reasoning, consistent=False)


def win_rate(results: list[PairwiseResult | None]) -> dict[str, float]:
    """Win/tie/loss rates of B over A (ties count half in ``b_score``)."""
    valid = [r for r in results if r is not None]
    if not valid:
        return {"b_win": 0.0, "a_win": 0.0, "tie": 0.0, "b_score": 0.0, "n": 0.0}
    n = len(valid)
    b = sum(r.winner == "b" for r in valid)
    a = sum(r.winner == "a" for r in valid)
    t = n - a - b
    return {"b_win": b / n, "a_win": a / n, "tie": t / n, "b_score": (b + 0.5 * t) / n, "n": float(n)}
