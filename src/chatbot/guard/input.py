"""Input guardrail pipeline (docs/13-guardrails-security.md §2, INPUT line).

    authN/Z -> rate limit -> PII detect/redact -> injection/jailbreak classifier
            -> topic/scope policy -> (block | safe-complete | proceed)

Authentication happens in the API layer; :class:`InputGuard` receives the resolved
:class:`~chatbot.core.types.Principal`. The guard is cheap by default (regex + rule
classifier, ~1 ms) so it can run in parallel with query condensing
(docs/14-production-llmops.md §1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable

from chatbot.core.tokens import Tokenizer, default_tokenizer
from chatbot.core.types import Principal
from chatbot.guard.injection import InjectionDetector
from chatbot.guard.pii import PIIDetector
from chatbot.guard.policies import InputGuardPolicy, TopicPolicy
from chatbot.guard.verdicts import GuardAction, GuardVerdict
from chatbot.llm.base import LLM
from chatbot.ops.ratelimit import BudgetExceeded, LimitExceeded

logger = logging.getLogger(__name__)

#: Called with the principal; raises :class:`~chatbot.ops.ratelimit.LimitExceeded` to reject.
RateLimitHook = Callable[[Principal], Awaitable[None]]


def _kw_pattern(keywords: list[str]) -> re.Pattern[str] | None:
    words = [k.strip().lower() for k in keywords if k.strip()]
    if not words:
        return None
    alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


class TopicClassifier:
    """Keyword scope policy: deny list wins; allow list (if any) must match."""

    def __init__(self, policy: TopicPolicy) -> None:
        self.policy = policy
        self._allow = _kw_pattern(policy.allow_keywords)
        self._deny = _kw_pattern(policy.deny_keywords)

    def check(self, text: str) -> tuple[bool, str]:
        """Return ``(in_scope, reason)``."""
        if self._deny is not None:
            m = self._deny.search(text)
            if m:
                return False, f"deny:{m.group(0).lower()}"
        if self._allow is not None:
            if len(text.split()) < self.policy.allow_list_min_words:
                return True, "short_message"
            if not self._allow.search(text):
                return False, "no_allowed_topic"
        return True, ""


class InputGuard:
    """Runs every input check and returns one :class:`GuardVerdict`.

    The verdict's ``redacted_text`` (always set when the request may proceed) is the
    text that downstream stages and logs must use.
    """

    def __init__(
        self,
        policy: InputGuardPolicy | None = None,
        *,
        pii: PIIDetector | None = None,
        injection: InjectionDetector | None = None,
        rate_limit: RateLimitHook | None = None,
        llm: LLM | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.policy = policy or InputGuardPolicy()
        self.pii = pii or PIIDetector.from_policy(self.policy.pii)
        self.injection = injection or InjectionDetector(self.policy.injection, llm=llm)
        self.rate_limit = rate_limit
        self.topics = TopicClassifier(self.policy.topics)
        self.tokenizer = tokenizer or default_tokenizer()

    async def check(self, text: str, principal: Principal | None = None) -> GuardVerdict:
        verdict = GuardVerdict(stage="input")
        p = self.policy

        # 1. size limits (also a denial-of-wallet control)
        if not text or not text.strip():
            verdict.escalate_to(GuardAction.BLOCK, "empty input", "empty_input")
            return verdict
        if len(text) > p.max_chars:
            verdict.escalate_to(
                GuardAction.BLOCK, f"input exceeds {p.max_chars} characters", "too_long"
            )
            return verdict
        n_tokens = self.tokenizer.count(text)
        verdict.scores["input_tokens"] = float(n_tokens)
        if n_tokens > p.max_tokens:
            verdict.escalate_to(GuardAction.BLOCK, f"input exceeds {p.max_tokens} tokens", "too_long")
            return verdict

        # 2. rate limit / budget hook
        if self.rate_limit is not None and principal is not None:
            try:
                await self.rate_limit(principal)
            except LimitExceeded as exc:
                verdict.scores["retry_after"] = exc.retry_after
                flag = "budget_exceeded" if isinstance(exc, BudgetExceeded) else "rate_limited"
                verdict.escalate_to(GuardAction.BLOCK, str(exc), flag)
                return verdict

        # 3. PII / secrets
        working = text
        red = self.pii.redact(text, reversible=p.reversible_redaction)
        if red.matches:
            for t, n in red.counts().items():
                verdict.scores[f"pii:{t}"] = float(n)
            if red.has_secrets:
                verdict.flags.append("secret_in_input")
                if p.block_on_secrets:
                    verdict.escalate_to(GuardAction.BLOCK, "input contains a credential", "secret")
                    return verdict
            if p.redact_pii or red.has_secrets:
                working = red.text
                verdict.redaction_map = dict(red.mapping)
                verdict.escalate_to(
                    GuardAction.REDACT,
                    f"redacted {len(red.matches)} sensitive value(s)",
                    "pii_redacted",
                )
            else:
                verdict.flags.append("pii_present")

        # 4. injection / jailbreak (scored on the original text: redaction can hide signals)
        inj = await self.injection.classify(text)
        verdict.scores["injection"] = inj.score
        if inj.llm_used:
            verdict.flags.append("injection_llm_checked")
        if inj.is_injection:
            verdict.flags.extend(f for f in inj.flags if f not in verdict.flags)
            verdict.escalate_to(
                GuardAction.BLOCK,
                f"prompt injection suspected (score {inj.score:.2f}: {', '.join(inj.matched)})",
                "prompt_injection",
            )
            logger.warning(
                "input blocked as injection user=%s score=%.2f rules=%s",
                principal.user_id if principal else "-",
                inj.score,
                inj.matched,
            )
            return verdict

        # 5. topic / scope policy
        in_scope, why = self.topics.check(working)
        if not in_scope:
            verdict.escalate_to(self.policy.topics.off_topic_action, f"out of scope ({why})", "off_topic")
            if verdict.blocked:
                return verdict

        verdict.redacted_text = working
        return verdict
