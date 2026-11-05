"""Configuration models for guard thresholds (docs/13-guardrails-security.md §2).

All thresholds live here so they can be loaded from the pipeline config, versioned
as artifacts and tuned against the red-team suite (docs/13 §5) without code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from chatbot.guard.verdicts import GuardAction


class PIIPolicy(BaseModel):
    """Which entity types to detect and how to treat them."""

    #: Entity type names (see :class:`chatbot.guard.pii.PIIType`); empty = all.
    enabled_types: list[str] = Field(default_factory=list)
    #: Phones are the noisiest detector; require at least this many digits.
    min_phone_digits: int = Field(default=9, ge=7, le=15)
    #: Use a Presidio analyzer in addition to the regex detectors when installed.
    use_presidio: bool = False


class InjectionPolicy(BaseModel):
    """Prompt-injection / jailbreak classifier thresholds."""

    #: Heuristic score at or above which a user message is treated as an attack.
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Lower threshold for retrieved chunks (indirect injection, docs/13 §1).
    chunk_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Heuristic band in which the optional LLM classifier is consulted.
    llm_band: tuple[float, float] = (0.2, 0.75)
    #: ``"borderline"`` (only inside ``llm_band``), ``"always"`` or ``"never"``.
    llm_mode: str = "borderline"
    #: Minimum length of a base64 run to be decoded and re-scanned.
    min_base64_len: int = Field(default=32, ge=16)

    @field_validator("llm_mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in {"borderline", "always", "never"}:
            raise ValueError("llm_mode must be borderline | always | never")
        return v

    @model_validator(mode="after")
    def _band(self) -> InjectionPolicy:
        lo, hi = self.llm_band
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("llm_band must satisfy 0 <= lo <= hi <= 1")
        return self


class TopicPolicy(BaseModel):
    """Scope policy: keywords are matched case-insensitively on word boundaries."""

    #: If non-empty, a message must mention at least one of these to be in scope.
    allow_keywords: list[str] = Field(default_factory=list)
    #: A message mentioning any of these is out of scope.
    deny_keywords: list[str] = Field(default_factory=list)
    #: Short messages (greetings, "thanks") are exempt from the allow list.
    allow_list_min_words: int = Field(default=4, ge=0)
    off_topic_action: GuardAction = GuardAction.BLOCK


class InputGuardPolicy(BaseModel):
    max_chars: int = Field(default=8000, ge=1)
    max_tokens: int = Field(default=2000, ge=1)
    redact_pii: bool = True
    #: Block (rather than redact) when the user pastes a credential.
    block_on_secrets: bool = False
    #: Keep a reversible placeholder map so answers can be re-personalised.
    reversible_redaction: bool = False
    injection: InjectionPolicy = Field(default_factory=InjectionPolicy)
    topics: TopicPolicy = Field(default_factory=TopicPolicy)
    pii: PIIPolicy = Field(default_factory=PIIPolicy)


class GroundednessPolicy(BaseModel):
    """Two-tier groundedness check (DoorDash pattern, docs/13 §2)."""

    #: Per-claim lexical support below which a claim is "unsupported".
    claim_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Mean support at/above which (with no unsupported claims) the answer passes cheaply.
    high: float = Field(default=0.75, ge=0.0, le=1.0)
    #: Mean support below which the answer fails cheaply (no LLM call).
    low: float = Field(default=0.3, ge=0.0, le=1.0)
    #: Max fraction of unsupported claims tolerated when no judge is available.
    max_unsupported_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Claims shorter than this many content words are not checked.
    min_claim_words: int = Field(default=3, ge=1)
    use_llm_judge: bool = True

    @model_validator(mode="after")
    def _order(self) -> GroundednessPolicy:
        if self.low > self.high:
            raise ValueError("groundedness low must be <= high")
        return self


class OutputGuardPolicy(BaseModel):
    #: Domains (and their sub-domains) that links / images may point to.
    allowed_domains: list[str] = Field(default_factory=list)
    #: Remove every markdown/HTML image, even on allow-listed domains.
    strip_all_images: bool = False
    strip_html: bool = True
    #: Replace bare URLs to non-allow-listed domains with a placeholder.
    strip_bare_urls: bool = True
    require_citations: bool = True
    redact_pii: bool = True
    #: PII values that appear verbatim in the sources (e.g. a support e-mail) are kept.
    allow_pii_from_sources: bool = True
    max_regenerations: int = Field(default=1, ge=0)
    escalate_on_fallback: bool = False
    fallback_text: str = (
        "I can't answer that reliably from the available sources. "
        "Please rephrase or contact support."
    )
    groundedness: GroundednessPolicy = Field(default_factory=GroundednessPolicy)


class PoisoningPolicy(BaseModel):
    """Ingest-time content scanning (knowledge-base poisoning, docs/13 §1)."""

    injection_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    #: source_uri prefix (or domain) -> authority score in [0, 1].
    trusted_sources: dict[str, float] = Field(default_factory=dict)
    trusted_authors: list[str] = Field(default_factory=list)
    default_authority: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Documents below this authority with any anomaly flag are quarantined.
    min_authority: float = Field(default=0.3, ge=0.0, le=1.0)
    new_doc_days: int = Field(default=7, ge=0)
    #: Most frequent content word share above which the doc looks keyword-stuffed.
    max_term_share: float = Field(default=0.12, gt=0.0, le=1.0)
    #: Share of leading sentences that are questions above which the doc looks query-stuffed.
    max_leading_question_share: float = Field(default=0.6, gt=0.0, le=1.0)
    quarantine_on_secrets: bool = True


class GuardPolicies(BaseModel):
    """Aggregate guard configuration."""

    input: InputGuardPolicy = Field(default_factory=InputGuardPolicy)
    output: OutputGuardPolicy = Field(default_factory=OutputGuardPolicy)
    poisoning: PoisoningPolicy = Field(default_factory=PoisoningPolicy)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GuardPolicies:
        return cls.model_validate(data or {})

    @classmethod
    def from_yaml(cls, path: str | Path) -> GuardPolicies:
        with Path(path).open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"guard policy file {path} must contain a mapping")
        return cls.from_dict(data.get("guard", data))
