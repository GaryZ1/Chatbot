import asyncio
from datetime import UTC, datetime

import pytest

from chatbot.core.types import Principal
from chatbot.llm.base import Usage
from chatbot.ops.budgets import (
    BudgetLimits,
    DeadlineExceeded,
    RequestBudget,
    with_fallback,
    with_timeout,
)
from chatbot.ops.ratelimit import (
    BudgetExceeded,
    DailyBudget,
    RateLimited,
    RateLimiter,
    TokenBucket,
    UserLimits,
)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_token_bucket_timing():
    c = Clock()
    b = TokenBucket(rate=2.0, capacity=4, clock=c)
    for _ in range(4):
        assert b.try_acquire()[0]
    ok, retry = b.try_acquire()
    assert not ok and retry == pytest.approx(0.5)
    c.t += 0.25
    ok, retry = b.try_acquire()
    assert not ok and retry == pytest.approx(0.25)
    c.t += 0.25
    assert b.try_acquire()[0]
    c.t += 100
    assert b.tokens == pytest.approx(4.0)  # capped at capacity


async def test_token_bucket_wait_with_fake_sleep():
    c = Clock()
    b = TokenBucket(rate=1.0, capacity=1, clock=c)
    await b.acquire()
    slept = []

    async def fake_sleep(s):
        slept.append(s)
        c.t += s

    await b.acquire(wait=True, sleep=fake_sleep)
    assert slept == [pytest.approx(1.0)]
    with pytest.raises(RateLimited):
        await b.acquire()


async def test_rate_limiter_per_key():
    c = Clock()
    rl = RateLimiter(rate=1.0, burst=2, clock=c)
    await rl.check("a")
    await rl.check("a")
    with pytest.raises(RateLimited) as e:
        await rl.check("a")
    assert e.value.retry_after == pytest.approx(1.0)
    await rl.check("b")  # independent key


async def test_daily_budget():
    now = [datetime(2026, 11, 5, 23, 0, tzinfo=UTC)]
    budget = DailyBudget(max_tokens_per_day=1000, max_usd_per_day=1.0, now=lambda: now[0])
    budget.record("u", tokens=900, usd=0.1)
    budget.check("u", est_tokens=100)
    with pytest.raises(BudgetExceeded) as e:
        budget.check("u", est_tokens=101)
    assert e.value.kind == "tokens" and e.value.retry_after == pytest.approx(3600)
    budget.record("u", usd=1.0)
    with pytest.raises(BudgetExceeded):
        budget.check("u")
    now[0] = datetime(2026, 11, 6, 0, 1, tzinfo=UTC)  # next day resets
    budget.check("u", est_tokens=1000)


async def test_user_limits_hook():
    p = Principal(user_id="u")
    limits = UserLimits(budget=DailyBudget(max_tokens_per_day=10, max_usd_per_day=None))
    await limits(p)
    limits.record(p, tokens=11)
    with pytest.raises(BudgetExceeded):
        await limits(p)


def test_request_budget_caps():
    c = Clock()
    b = RequestBudget(
        BudgetLimits(max_llm_calls=2, max_tokens=100, max_agent_steps=2, deadline_s=5), clock=c
    )
    b.check_llm_call()
    b.charge_llm(Usage(input_tokens=30, output_tokens=10))
    b.check_llm_call()
    with pytest.raises(BudgetExceeded) as e:
        b.charge_llm(Usage(input_tokens=60, output_tokens=10))
    assert e.value.kind == "tokens"
    with pytest.raises(BudgetExceeded) as e:
        b.check_llm_call()
    assert e.value.kind == "llm_calls"
    assert b.step() == 1 and b.step() == 2
    with pytest.raises(BudgetExceeded) as e:
        b.step()
    assert e.value.kind == "agent_steps"
    c.t += 6
    with pytest.raises(DeadlineExceeded):
        b.check_deadline()


async def test_deadline_helpers():
    b = RequestBudget(BudgetLimits(deadline_s=10))
    with pytest.raises(DeadlineExceeded) as e:
        async with b.deadline("rerank", stage_timeout_s=0.01):
            await asyncio.sleep(1)
    assert e.value.stage == "rerank" and isinstance(e.value, TimeoutError)
    assert await with_timeout(asyncio.sleep(0, result=5), 1) == 5
    assert await with_fallback(asyncio.sleep(1, result=1), 0.01, fallback=0) == 0
