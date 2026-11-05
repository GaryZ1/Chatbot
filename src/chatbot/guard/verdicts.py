"""Guardrail verdicts shared by the input, retrieval and output guards.

Implements the decision vocabulary of the guardrail pipeline in
docs/13-guardrails-security.md §2: the input side ends in *block | safe-complete |
proceed* (``GuardAction``), the output side in *deliver | regenerate | fallback |
escalate* (``OutputAction``). Every verdict carries machine-readable ``flags``,
human-readable ``reasons`` and numeric ``scores`` so it can be logged in the trace
(docs/14-production-llmops.md §3) and aggregated in dashboards.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GuardAction(StrEnum):
    """Input / retrieval guard decision, ordered by severity (see :data:`_SEVERITY`)."""

    ALLOW = "allow"
    REDACT = "redact"  # proceed with ``redacted_text`` instead of the raw text
    REGENERATE = "regenerate"
    ESCALATE = "escalate"  # hand off to a human
    BLOCK = "block"  # refuse / safe-complete


_SEVERITY: dict[GuardAction, int] = {
    GuardAction.ALLOW: 0,
    GuardAction.REDACT: 1,
    GuardAction.REGENERATE: 2,
    GuardAction.ESCALATE: 3,
    GuardAction.BLOCK: 4,
}


class OutputAction(StrEnum):
    """Output guard decision (docs/13 §2, OUTPUT line)."""

    DELIVER = "deliver"
    REGENERATE = "regenerate"
    FALLBACK = "fallback"  # "I can't answer that" style safe answer
    ESCALATE = "escalate"


class GuardVerdict(BaseModel):
    """Result of an input or retrieval guard."""

    action: GuardAction = GuardAction.ALLOW
    stage: str = "input"
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    #: Text the pipeline must use downstream (PII-redacted); ``None`` means unchanged.
    redacted_text: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    #: Optional reversible placeholder mapping (placeholder -> original), never logged.
    redaction_map: dict[str, str] = Field(default_factory=dict, exclude=True, repr=False)

    @property
    def allowed(self) -> bool:
        """True when the request may proceed (possibly with redacted text)."""
        return self.action in (GuardAction.ALLOW, GuardAction.REDACT)

    @property
    def blocked(self) -> bool:
        return self.action is GuardAction.BLOCK

    def text_or(self, original: str) -> str:
        """The text to use downstream: the redacted text if any, else ``original``."""
        return self.redacted_text if self.redacted_text is not None else original

    def escalate_to(self, action: GuardAction, reason: str, flag: str | None = None) -> None:
        """Raise the action to ``action`` if it is more severe, recording why."""
        if _SEVERITY[action] > _SEVERITY[self.action]:
            self.action = action
        self.reasons.append(reason)
        if flag and flag not in self.flags:
            self.flags.append(flag)

    def merge(self, other: GuardVerdict) -> GuardVerdict:
        """Combine two verdicts; the most severe action wins, evidence is unioned."""
        action = self.action if _SEVERITY[self.action] >= _SEVERITY[other.action] else other.action
        flags = list(dict.fromkeys([*self.flags, *other.flags]))
        return GuardVerdict(
            action=action,
            stage=self.stage if self.stage == other.stage else f"{self.stage}+{other.stage}",
            reasons=[*self.reasons, *other.reasons],
            flags=flags,
            redacted_text=other.redacted_text
            if other.redacted_text is not None
            else self.redacted_text,
            scores={**self.scores, **other.scores},
            redaction_map={**self.redaction_map, **other.redaction_map},
        )

    def to_log(self) -> dict[str, Any]:
        """Trace-safe representation (no redaction map, no raw text)."""
        return {
            "stage": self.stage,
            "action": self.action.value,
            "flags": list(self.flags),
            "reasons": list(self.reasons),
            "scores": dict(self.scores),
        }


class OutputVerdict(BaseModel):
    """Result of :class:`chatbot.guard.output.OutputGuard`."""

    action: OutputAction = OutputAction.DELIVER
    #: Sanitised (markdown/URL/HTML stripped, PII-redacted) answer text to deliver.
    text: str = ""
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    #: Whether the LLM judge (second tier) was consulted.
    judge_used: bool = False

    @property
    def deliverable(self) -> bool:
        return self.action is OutputAction.DELIVER

    def to_log(self) -> dict[str, Any]:
        return {
            "stage": "output",
            "action": self.action.value,
            "flags": list(self.flags),
            "reasons": list(self.reasons),
            "scores": dict(self.scores),
            "unsupported_claims": len(self.unsupported_claims),
            "invalid_citations": list(self.invalid_citations),
            "judge_used": self.judge_used,
        }
