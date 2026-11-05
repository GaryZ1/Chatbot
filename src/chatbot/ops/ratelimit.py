"""Rate limits and per-user budgets against denial of wallet.

Implements docs/13-guardrails-security.md §1 ("Denial of wallet: rate limits,
per-user budgets, agent step caps") and the gateway budgets of
docs/14-production-llmops.md §5. Everything takes an injectable clock so behaviour
is deterministic under test; state is in-process (use one limiter per API worker,
or back it with Redis for a fleet-wide limit).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from chatbot.core.types import Principal

logger = logging.getLogger(__name__)

Clock = Callable[[], float]
WallClock = Callable[[], datetime]


class LimitExceeded(Exception):
    """Base class for all limit errors; ``retry_after`` is in seconds (may be 0)."""

    def __init__(self, message: str, retry_after: float = 0.0, key: str = "") -> None:
        super().__init__(message)
        self.retry_after = max(0.0, float(retry_after))
        self.key = key


class RateLimited(LimitExceeded):
    """Request rate exceeded (HTTP 429)."""


class BudgetExceeded(LimitExceeded):
    """A token or dollar budget is exhausted (daily user budget or request budget)."""

    def __init__(
        self, message: str, retry_after: float = 0.0, key: str = "", kind: str = "budget"
    ) -> None:
        super().__init__(message, retry_after, key)
        self.kind = kind


class TokenBucket:
    """Classic token bucket: ``capacity`` burst, refilled at ``rate`` tokens per second."""

    def __init__(self, rate: float, capacity: float, *, clock: Clock = time.monotonic) -> None:
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.rate = float(rate)
        self.capacity = float(capacity)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        elapsed = max(0.0, now - self._last)
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last = now

    @property
    def tokens(self) -> float:
        self._refill()
        return self._tokens

    def try_acquire(self, n: float = 1.0) -> tuple[bool, float]:
        """Take ``n`` tokens if available. Returns ``(ok, retry_after_seconds)``."""
        if n <= 0:
            return True, 0.0
        if n > self.capacity:
            raise ValueError(f"cannot acquire {n} tokens from a bucket of capacity {self.capacity}")
        self._refill()
        if self._tokens >= n:
            self._tokens -= n
            return True, 0.0
        return False, (n - self._tokens) / self.rate

    async def acquire(
        self,
        n: float = 1.0,
        *,
        wait: bool = False,
        max_wait: float = 5.0,
        sleep: Callable[[float], object] = asyncio.sleep,
    ) -> None:
        """Acquire or raise :class:`RateLimited`; with ``wait=True`` sleep up to ``max_wait``."""
        async with self._lock:
            ok, retry = self.try_acquire(n)
            while not ok and wait and retry <= max_wait:
                res = sleep(retry)
                if asyncio.iscoroutine(res):
                    await res
                max_wait -= retry
                ok, retry = self.try_acquire(n)
            if not ok:
                raise RateLimited(f"rate limit exceeded; retry in {retry:.2f}s", retry)


class RateLimiter:
    """Per-key token buckets (key = user id, API key, tenant ...), LRU-bounded."""

    def __init__(
        self,
        rate: float = 1.0,
        burst: float = 10.0,
        *,
        clock: Clock = time.monotonic,
        max_keys: int = 100_000,
    ) -> None:
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self.max_keys = max_keys
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()

    def bucket(self, key: str) -> TokenBucket:
        b = self._buckets.get(key)
        if b is None:
            b = TokenBucket(self.rate, self.burst, clock=self._clock)
            self._buckets[key] = b
            if len(self._buckets) > self.max_keys:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
        return b

    def try_acquire(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        return self.bucket(key).try_acquire(cost)

    async def check(self, key: str, cost: float = 1.0) -> None:
        """Raise :class:`RateLimited` (with ``retry_after``) if ``key`` is over its rate."""
        ok, retry = self.try_acquire(key, cost)
        if not ok:
            logger.info("rate limited key=%s retry_after=%.2fs", key, retry)
            raise RateLimited(f"rate limit exceeded for {key}; retry in {retry:.2f}s", retry, key)

    async def __call__(self, principal: Principal) -> None:
        """Hook signature used by :class:`chatbot.guard.input.InputGuard`."""
        await self.check(f"{principal.tenant_id}:{principal.user_id}")


@dataclass(slots=True)
class _Usage:
    day: str
    tokens: int = 0
    usd: float = 0.0


class DailyBudget:
    """Per-user daily token and dollar budgets (reset at UTC midnight)."""

    def __init__(
        self,
        max_tokens_per_day: int | None = 2_000_000,
        max_usd_per_day: float | None = 5.0,
        *,
        now: WallClock = lambda: datetime.now(UTC),
    ) -> None:
        if max_tokens_per_day is not None and max_tokens_per_day <= 0:
            raise ValueError("max_tokens_per_day must be positive or None")
        if max_usd_per_day is not None and max_usd_per_day <= 0:
            raise ValueError("max_usd_per_day must be positive or None")
        self.max_tokens = max_tokens_per_day
        self.max_usd = max_usd_per_day
        self._now = now
        self._usage: dict[str, _Usage] = {}

    def _today(self) -> tuple[str, float]:
        now = self._now().astimezone(UTC)
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return now.date().isoformat(), (midnight - now).total_seconds()

    def _get(self, key: str) -> _Usage:
        day, _ = self._today()
        u = self._usage.get(key)
        if u is None or u.day != day:
            u = _Usage(day=day)
            self._usage[key] = u
        return u

    def usage(self, key: str) -> tuple[int, float]:
        u = self._get(key)
        return u.tokens, u.usd

    def remaining(self, key: str) -> tuple[int | None, float | None]:
        u = self._get(key)
        tok = None if self.max_tokens is None else max(0, self.max_tokens - u.tokens)
        usd = None if self.max_usd is None else max(0.0, self.max_usd - u.usd)
        return tok, usd

    def check(self, key: str, *, est_tokens: int = 0, est_usd: float = 0.0) -> None:
        """Raise :class:`BudgetExceeded` if the (estimated) request would exceed a budget."""
        u = self._get(key)
        _, until_reset = self._today()
        if self.max_tokens is not None and u.tokens + est_tokens > self.max_tokens:
            raise BudgetExceeded(
                f"daily token budget exhausted for {key}", until_reset, key, kind="tokens"
            )
        if self.max_usd is not None and u.usd + est_usd > self.max_usd:
            raise BudgetExceeded(
                f"daily cost budget exhausted for {key}", until_reset, key, kind="usd"
            )

    def record(self, key: str, *, tokens: int = 0, usd: float = 0.0) -> None:
        if tokens < 0 or usd < 0:
            raise ValueError("usage must be non-negative")
        u = self._get(key)
        u.tokens += tokens
        u.usd += usd

    async def __call__(self, principal: Principal) -> None:
        """Hook signature used by :class:`chatbot.guard.input.InputGuard`."""
        self.check(f"{principal.tenant_id}:{principal.user_id}")


class UserLimits:
    """Composite hook: request rate limit + daily budget for a principal."""

    def __init__(self, rate: RateLimiter | None = None, budget: DailyBudget | None = None) -> None:
        self.rate = rate
        self.budget = budget

    @staticmethod
    def key(principal: Principal) -> str:
        return f"{principal.tenant_id}:{principal.user_id}"

    async def __call__(self, principal: Principal) -> None:
        if self.budget is not None:
            self.budget.check(self.key(principal))
        if self.rate is not None:
            await self.rate.check(self.key(principal))

    def record(self, principal: Principal, *, tokens: int = 0, usd: float = 0.0) -> None:
        if self.budget is not None:
            self.budget.record(self.key(principal), tokens=tokens, usd=usd)
