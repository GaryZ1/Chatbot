"""In-process metrics and cost accounting (docs/14-production-llmops.md §1-3).

* :class:`MetricsRegistry` holds labelled counters, gauges and histograms with
  p50/p95/p99 (the latency budget in docs/14 §1 is stated as a p95), rendered in the
  Prometheus text exposition format so any scraper can read them.
* :class:`PriceTable` / :func:`estimate_cost` turn token usage into dollars so cost
  can be tracked per route and per *resolved conversation* (docs/14 §2), not only
  per call.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from chatbot.llm.base import Usage

logger = logging.getLogger(__name__)

LabelKey = tuple[tuple[str, str], ...]

_NAME_RE = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


def _labels(labels: Mapping[str, object] | None) -> LabelKey:
    return tuple(sorted((str(k), str(v)) for k, v in (labels or {}).items()))


def _fmt_labels(key: LabelKey, extra: Sequence[tuple[str, str]] = ()) -> str:
    items = [*key, *extra]
    if not items:
        return ""

    def esc(v: str) -> str:
        return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    return "{" + ",".join(f'{k}="{esc(v)}"' for k, v in items) + "}"


def _fmt_value(v: float) -> str:
    if math.isinf(v):
        return "+Inf" if v > 0 else "-Inf"
    if math.isnan(v):
        return "NaN"
    return repr(float(v)) if not float(v).is_integer() else str(int(v))


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]); NaN for no data."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    if not values:
        return math.nan
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help: str = "") -> None:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid metric name {name!r}")
        self.name = name
        self.help = help
        self._lock = threading.Lock()


class Counter(_Metric):
    kind = "counter"

    def __init__(self, name: str, help: str = "") -> None:
        super().__init__(name, help)
        self._values: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: object) -> None:
        if amount < 0:
            raise ValueError("counters can only increase")
        key = _labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: object) -> float:
        return self._values.get(_labels(labels), 0.0)

    def total(self) -> float:
        return sum(self._values.values())

    def render(self) -> list[str]:
        return [f"{self.name}{_fmt_labels(k)} {_fmt_value(v)}" for k, v in sorted(self._values.items())]


class Gauge(_Metric):
    kind = "gauge"

    def __init__(self, name: str, help: str = "") -> None:
        super().__init__(name, help)
        self._values: dict[LabelKey, float] = {}

    def set(self, value: float, **labels: object) -> None:
        with self._lock:
            self._values[_labels(labels)] = float(value)

    def value(self, **labels: object) -> float:
        return self._values.get(_labels(labels), math.nan)

    def render(self) -> list[str]:
        return [f"{self.name}{_fmt_labels(k)} {_fmt_value(v)}" for k, v in sorted(self._values.items())]


@dataclass(slots=True)
class _Series:
    samples: deque[float]
    count: int = 0
    total: float = 0.0


class Histogram(_Metric):
    """Sliding-window histogram: exact count/sum, percentiles over the last ``window`` samples.

    Rendered as a Prometheus *summary* (quantiles + ``_sum`` + ``_count``).
    """

    kind = "summary"
    QUANTILES = (0.5, 0.95, 0.99)

    def __init__(self, name: str, help: str = "", window: int = 10_000) -> None:
        super().__init__(name, help)
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = window
        self._series: dict[LabelKey, _Series] = {}

    def observe(self, value: float, **labels: object) -> None:
        key = _labels(labels)
        with self._lock:
            s = self._series.get(key)
            if s is None:
                s = _Series(deque(maxlen=self.window))
                self._series[key] = s
            s.samples.append(float(value))
            s.count += 1
            s.total += float(value)

    def percentile(self, q: float, **labels: object) -> float:
        s = self._series.get(_labels(labels))
        return percentile(list(s.samples), q) if s else math.nan

    def p50(self, **labels: object) -> float:
        return self.percentile(50, **labels)

    def p95(self, **labels: object) -> float:
        return self.percentile(95, **labels)

    def count(self, **labels: object) -> int:
        s = self._series.get(_labels(labels))
        return s.count if s else 0

    def sum(self, **labels: object) -> float:
        s = self._series.get(_labels(labels))
        return s.total if s else 0.0

    def render(self) -> list[str]:
        lines: list[str] = []
        for key, s in sorted(self._series.items()):
            samples = list(s.samples)
            for q in self.QUANTILES:
                lines.append(
                    f"{self.name}{_fmt_labels(key, [('quantile', str(q))])} "
                    f"{_fmt_value(percentile(samples, q * 100))}"
                )
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {_fmt_value(s.total)}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {s.count}")
        return lines


M = TypeVar("M", bound=_Metric)


class MetricsRegistry:
    """Named metrics; ``counter``/``gauge``/``histogram`` are get-or-create."""

    def __init__(self, namespace: str = "chatbot") -> None:
        self.namespace = namespace
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _full(self, name: str) -> str:
        return f"{self.namespace}_{name}" if self.namespace else name

    def _get(self, cls: type[M], name: str, help: str, **kw: int) -> M:
        full = self._full(name)
        with self._lock:
            m = self._metrics.get(full)
            if m is None:
                m = cls(full, help, **kw)
                self._metrics[full] = m
            if not isinstance(m, cls):
                raise TypeError(f"metric {full} already registered as {type(m).__name__}")
            return m

    def counter(self, name: str, help: str = "") -> Counter:
        return self._get(Counter, name, help)

    def gauge(self, name: str, help: str = "") -> Gauge:
        return self._get(Gauge, name, help)

    def histogram(self, name: str, help: str = "", window: int = 10_000) -> Histogram:
        return self._get(Histogram, name, help, window=window)

    def render_prometheus(self) -> str:
        """Prometheus text exposition format (version 0.0.4)."""
        out: list[str] = []
        for name in sorted(self._metrics):
            m = self._metrics[name]
            if m.help:
                out.append(f"# HELP {name} {m.help}")
            out.append(f"# TYPE {name} {m.kind}")
            out.extend(m.render())  # type: ignore[attr-defined]
        return "\n".join(out) + "\n"

    # ------------------------------------------------------------------ standard RAG metrics

    def observe_request(
        self,
        *,
        route: str,
        status: str,
        latency_ms: Mapping[str, float],
        model: str = "",
        usage: Usage | None = None,
        cost_usd: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        """Record the standard per-request metrics from a finished trace."""
        self.counter("requests_total", "Chat requests by route and outcome").inc(
            route=route, status=status
        )
        lat = self.histogram("stage_latency_ms", "Latency per pipeline stage (ms)")
        for stage, ms in latency_ms.items():
            lat.observe(ms, stage=stage, route=route)
        if cache_hit:
            self.counter("semantic_cache_hits_total", "Semantic cache hits").inc(route=route)
        if usage is not None:
            tok = self.counter("llm_tokens_total", "LLM tokens by model and kind")
            tok.inc(usage.input_tokens, model=model, kind="input")
            tok.inc(usage.output_tokens, model=model, kind="output")
            tok.inc(usage.cache_read_input_tokens, model=model, kind="cache_read")
            tok.inc(usage.cache_creation_input_tokens, model=model, kind="cache_write")
        if cost_usd:
            self.counter("llm_cost_usd_total", "Estimated LLM spend (USD)").inc(
                cost_usd, route=route
            )


# --------------------------------------------------------------------------- cost


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float
    #: Cache reads; defaults to 10% of the input price.
    cache_read: float | None = None
    #: Cache writes (5-minute TTL); defaults to 125% of the input price.
    cache_write: float | None = None

    @property
    def cache_read_price(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * 0.10

    @property
    def cache_write_price(self) -> float:
        return self.cache_write if self.cache_write is not None else self.input * 1.25

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_input_tokens * self.cache_read_price
            + usage.cache_creation_input_tokens * self.cache_write_price
        ) / 1_000_000


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.0, output=20.0, cache_read=0.20),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50),
    "fake": ModelPrice(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
}


class PriceTable:
    """Model id -> price. Lookup is exact, then longest prefix (dated / suffixed ids)."""

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None, *, strict: bool = False) -> None:
        self.prices: dict[str, ModelPrice] = dict(DEFAULT_PRICES if prices is None else prices)
        self.strict = strict
        self._warned: set[str] = set()

    @classmethod
    def from_config(cls, data: Mapping[str, Mapping[str, float]], *, strict: bool = False) -> PriceTable:
        """``{"model": {"input": 1.0, "output": 5.0, "cache_read": 0.1}}`` (USD / MTok)."""
        table = dict(DEFAULT_PRICES)
        for model, p in data.items():
            table[model] = ModelPrice(
                input=float(p["input"]),
                output=float(p["output"]),
                cache_read=float(p["cache_read"]) if "cache_read" in p else None,
                cache_write=float(p["cache_write"]) if "cache_write" in p else None,
            )
        return cls(table, strict=strict)

    def get(self, model: str) -> ModelPrice | None:
        if model in self.prices:
            return self.prices[model]
        base = model.split("[", 1)[0]
        candidates = [k for k in self.prices if base.startswith(k)]
        return self.prices[max(candidates, key=len)] if candidates else None

    def cost(self, model: str, usage: Usage) -> float:
        price = self.get(model)
        if price is None:
            if self.strict:
                raise KeyError(f"no price configured for model {model!r}")
            if model not in self._warned:
                logger.warning("no price for model %r; cost counted as 0", model)
                self._warned.add(model)
            return 0.0
        return price.cost(usage)


_DEFAULT_TABLE = PriceTable()


def estimate_cost(model: str, usage: Usage, table: PriceTable | None = None) -> float:
    """USD cost of one call."""
    return (table or _DEFAULT_TABLE).cost(model, usage)


def cost_per_resolved(costs: Iterable[float], resolved: Iterable[bool]) -> float:
    """$/resolved conversation (docs/14 §2): total spend over resolved conversations."""
    total = 0.0
    n_resolved = 0
    for c, r in zip(costs, resolved, strict=True):
        total += c
        n_resolved += bool(r)
    return total / n_resolved if n_resolved else math.inf if total else 0.0
