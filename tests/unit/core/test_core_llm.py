import pytest
from pydantic import BaseModel

from chatbot.llm import FailoverLLM, FakeLLM, LLMError, LLMRefusal, LLMRequest, Tier
from chatbot.llm.anthropic_llm import AnthropicLLM


class Plan(BaseModel):
    intent: str


async def test_fake_llm_parse_and_script():
    llm = FakeLLM(script=['{"intent": "kb_qa"}', "hello world, streamed"])
    plan = await llm.parse(LLMRequest.simple("q", tier=Tier.SMALL), Plan)
    assert plan.intent == "kb_qa"
    chunks = [c async for c in llm.stream(LLMRequest.simple("q"))]
    assert "".join(chunks) == "hello world, streamed"
    with pytest.raises(LLMError):
        await FakeLLM(default="not json").parse(LLMRequest.simple("q"), Plan)


class _Boom(FakeLLM):
    async def generate(self, request):
        raise LLMError("down")


class _Refuse(FakeLLM):
    async def generate(self, request):
        raise LLMRefusal("no")


async def test_failover_skips_errors_not_refusals():
    ok = FakeLLM(default="fine")
    assert (await FailoverLLM([_Boom(), ok]).generate(LLMRequest.simple("x"))).text == "fine"
    with pytest.raises(LLMRefusal):
        await FailoverLLM([_Refuse(), ok]).generate(LLMRequest.simple("x"))


def test_anthropic_params_tiering_caching_fallbacks():
    llm = AnthropicLLM(client=object())
    p = llm._params(LLMRequest.simple("hi", system="stable", tier=Tier.MID))
    assert p["model"] == "claude-opus-5-5"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert p["fallbacks"] == "default" and p["betas"]
    assert p["output_config"] == {"effort": "medium"}
    assert "temperature" not in p
    small = llm._params(LLMRequest.simple("hi", tier=Tier.SMALL))
    assert small["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in small and "system" not in small
