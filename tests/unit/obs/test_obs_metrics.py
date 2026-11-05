import math

import pytest

from chatbot.llm.base import Usage
from chatbot.obs.metrics import (
    MetricsRegistry,
    ModelPrice,
    PriceTable,
    cost_per_resolved,
    estimate_cost,
    percentile,
)


def test_percentiles():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == pytest.approx(50.5)
    assert percentile(vals, 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
    assert math.isnan(percentile([], 50))


def test_histogram_and_prometheus():
    reg = MetricsRegistry()
    h = reg.histogram("stage_latency_ms", "latency")
    for v in range(1, 101):
        h.observe(v, stage="rerank")
    assert h.p50(stage="rerank") == pytest.approx(50.5)
    assert h.p95(stage="rerank") == pytest.approx(95.05)
    assert h.count(stage="rerank") == 100
    reg.counter("requests_total").inc(route="rag", status="answered")
    reg.counter("requests_total").inc(2, route="rag", status="answered")
    text = reg.render_prometheus()
    assert "# TYPE chatbot_stage_latency_ms summary" in text
    assert 'chatbot_stage_latency_ms{stage="rerank",quantile="0.95"} 95.05' in text
    assert 'chatbot_stage_latency_ms_count{stage="rerank"} 100' in text
    assert 'chatbot_requests_total{route="rag",status="answered"} 3' in text
    with pytest.raises(TypeError):
        reg.counter("stage_latency_ms")
    with pytest.raises(ValueError):
        reg.counter("requests_total").inc(-1)


def test_cost_defaults():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_input_tokens=1_000_000)
    assert estimate_cost("claude-opus-5-5", u) == pytest.approx(4 + 20 + 0.20)
    # haiku: cache reads default to 10% of input
    assert estimate_cost("claude-haiku-5-5", u) == pytest.approx(0.10 + 0.50 + 0.01)
    # prefix match on suffixed ids
    assert estimate_cost("claude-opus-5-5-20261101", Usage(output_tokens=1000)) == pytest.approx(
        0.02
    )
    assert estimate_cost("unknown-model", u) == 0.0
    with pytest.raises(KeyError):
        PriceTable(strict=True).cost("unknown-model", u)


def test_custom_price_table_and_cache_write():
    t = PriceTable.from_config({"my-model": {"input": 2.0, "output": 8.0}})
    assert t.cost("my-model", Usage(cache_creation_input_tokens=1_000_000)) == pytest.approx(2.5)
    assert ModelPrice(1.0, 2.0).cache_read_price == pytest.approx(0.1)


def test_cost_per_resolved():
    assert cost_per_resolved([0.1, 0.2, 0.3], [True, False, True]) == pytest.approx(0.3)
    assert cost_per_resolved([], []) == 0.0


def test_observe_request():
    reg = MetricsRegistry()
    reg.observe_request(
        route="rag",
        status="answered",
        latency_ms={"retrieve": 40.0, "total": 900.0},
        model="claude-haiku-5-5",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=0.001,
    )
    assert reg.counter("llm_tokens_total").value(model="claude-haiku-5-5", kind="input") == 10
    assert reg.histogram("stage_latency_ms").count(stage="total", route="rag") == 1
