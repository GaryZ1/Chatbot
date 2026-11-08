"""Shared building blocks for the advanced-RAG catalog (docs/20-advanced-rag-catalog.md).

Every method file in this folder imports from here, so each listing in the doc only
shows the code that is specific to that method. The kit contains:

* a small corpus with multi-hop facts, a structured table and "themes";
* ``tokenize``, ``BM25``, a deterministic ``HashingEmbedder`` and a ``DenseIndex``;
* ``rrf`` (reciprocal rank fusion) and a hybrid ``Retriever``;
* ``format_context`` / ``generate_answer`` (grounded, cited generation);
* two interchangeable LLMs behind one call signature ``llm(prompt, task=...)``:
  - ``ClaudeLLM`` calls the Anthropic API (set ``RAG_LLM=claude``);
  - ``OfflineLLM`` (offline_llm.py) is a deterministic heuristic stand-in so every
    example runs with no network and no API key (a test double, not a model).

``task`` names the job of the call ("grade", "rewrite", "answer", ...). ``ClaudeLLM``
uses it to route cheap jobs to a small model; ``OfflineLLM`` uses it to pick a heuristic.
Prompts put inputs in XML tags (<question>, <context>, <document>, ...), which is good
prompting practice and lets the offline double find its inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

# --------------------------------------------------------------------------- corpus


@dataclass
class Doc:
    id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


CORPUS: list[Doc] = [
    Doc("d01", "Orbital Systems overview",
        "Orbital Systems is a satellite software company founded in 2009 by Mara Quinn. "
        "Orbital Systems is headquartered in Lisbon. Its main product is Skyline, "
        "a ground-station scheduling platform.", {"year": 2024, "type": "company"}),
    Doc("d02", "Acquisition of Lumen Labs",
        "In 2021, Orbital Systems acquired Lumen Labs for 40 million euros. "
        "Lumen Labs builds low-power star trackers. After the acquisition, Lumen Labs "
        "founder Theo Brandt became the CTO of Orbital Systems.", {"year": 2021, "type": "news"}),
    Doc("d03", "Theo Brandt profile",
        "Theo Brandt studied physics at Uppsala University. Before founding Lumen Labs, "
        "Theo Brandt worked at the European Space Agency for six years.",
        {"year": 2022, "type": "people"}),
    Doc("d04", "Skyline product",
        "Skyline schedules contacts between satellites and ground stations. Skyline 3.0, "
        "released in 2023, added collision-avoidance alerts. Skyline is used by 120 operators.",
        {"year": 2023, "type": "product"}),
    Doc("d05", "Refund policy",
        "Customers on annual Skyline plans can request a refund within 30 days of purchase. "
        "Refunds are paid to the original payment method within 10 business days.",
        {"year": 2024, "type": "policy"}),
    Doc("d06", "Security incident 2022",
        "In March 2022 a misconfigured storage bucket exposed Skyline telemetry logs. "
        "Orbital Systems rotated all keys within 24 hours and introduced mandatory access "
        "reviews.", {"year": 2022, "type": "security"}),
    Doc("d07", "Remote work policy",
        "Since 2024, employees may work remotely up to three days per week. "
        "Teams in Lisbon meet in the office on Tuesdays.", {"year": 2024, "type": "policy"}),
    Doc("d08", "Lumen ST-2 star tracker",
        "The Lumen ST-2 star tracker weighs 180 grams and consumes 0.9 watts. "
        "The Lumen ST-2 is built in the Lumen Labs factory in Gothenburg.",
        {"year": 2023, "type": "product"}),
    Doc("d09", "Mara Quinn profile",
        "Mara Quinn is the CEO of Orbital Systems. Mara Quinn previously led mission "
        "planning at Helios Aerospace.", {"year": 2023, "type": "people"}),
    Doc("d10", "Helios Aerospace",
        "Helios Aerospace is a launch provider based in Toulouse. Helios Aerospace is the "
        "largest customer of Orbital Systems.", {"year": 2023, "type": "company"}),
    Doc("d11", "Sustainability report 2023",
        "Orbital Systems cut data-center energy use by 35 percent in 2023 by moving Skyline "
        "to renewable-powered regions.", {"year": 2023, "type": "report"}),
    Doc("d12", "Customer feedback themes",
        "Operators praise Skyline reliability but complain about the steep learning curve "
        "and limited API documentation.", {"year": 2024, "type": "report"}),
    Doc("d13", "Q3 2024 results",
        "Orbital Systems revenue grew 22 percent year over year in Q3 2024 to 18 million "
        "euros, driven by Skyline subscriptions.", {"year": 2024, "type": "finance"}),
]

#: A small structured table for text-to-SQL (quarter, revenue in million EUR, operators).
REVENUE_ROWS: list[tuple[str, float, int]] = [
    ("2023-Q3", 14.8, 96), ("2023-Q4", 15.6, 101), ("2024-Q1", 16.1, 108),
    ("2024-Q2", 17.0, 114), ("2024-Q3", 18.0, 120),
]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def chunk_docs(docs: Sequence[Doc] = CORPUS, sentences_per_chunk: int = 2) -> list[Chunk]:
    """Sentence-group chunking: small, clean chunks keep the examples readable."""
    chunks: list[Chunk] = []
    for d in docs:
        sents = split_sentences(d.text)
        for i in range(0, len(sents), sentences_per_chunk):
            text = " ".join(sents[i : i + sentences_per_chunk])
            chunks.append(Chunk(f"{d.id}#{i // sentences_per_chunk}", d.id, d.title, text,
                                dict(d.meta)))
    return chunks


# --------------------------------------------------------------------------- text utils

STOPWORDS = set(
    ["a", "an", "and", "are", "as", "at", "be", "been", "before", "but", "by", "can", "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his", "how", "i", "in", "into", "is", "it", "its", "may", "me", "of", "on", "or", "our", "she", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "up", "was", "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with", "would", "you", "your", "after", "about", "also", "all", "any", "most", "much", "many", "over", "per", "since", "such", "via"]
)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.lower())


def stem(t: str) -> str:
    """Tiny suffix stripper (founded/founding -> found). Use a real stemmer in production."""
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def content_terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def overlap(a: str, b: str) -> float:
    """Fraction of ``a``'s content terms that appear in ``b`` (0..1)."""
    ta = set(content_terms(a))
    return len(ta & set(content_terms(b))) / len(ta) if ta else 0.0


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else ""


def extract_all(text: str, tag: str) -> list[str]:
    return [m.strip() for m in re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)]


def parse_json(text: str, default: Any) -> Any:
    """Tolerant JSON extraction: models sometimes wrap JSON in prose or fences."""
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return default


ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9-]*|ST-2)(?: +(?:[A-Z][a-zA-Z0-9-]*|ST-2|\d\.\d))*")
_NOT_ENTITY = {"In", "The", "It", "Its", "Since", "After", "Before", "Customers", "Refunds",
               "Teams", "Operators", "What", "Which", "Who", "Where", "When", "How", "Is",
               "Did", "Does", "Are", "Q3", "This", "A", "An"}


def entities(text: str) -> list[str]:
    """Capitalised-phrase entity extractor (a stand-in for NER / LLM extraction)."""
    out: list[str] = []
    for m in ENTITY_RE.findall(text):
        words = m.split()
        while words and words[0] in _NOT_ENTITY:
            words = words[1:]
        name = " ".join(words)
        if name and name not in out and len(name) > 2:
            out.append(name)
    return out


# --------------------------------------------------------------------------- sparse: BM25


class BM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query: str) -> np.ndarray:
        q = [t for t in tokenize(query) if t not in STOPWORDS]
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                         for t in q if t in tf)
        return out


# --------------------------------------------------------------------------- dense


class HashingEmbedder:
    """Deterministic bag-of-words + char-trigram hashing embedder (offline stand-in).

    Swap for a real model in production, e.g. ``sentence_transformers.SentenceTransformer
    ("BAAI/bge-m3").encode(texts, normalize_embeddings=True)`` or Voyage / Cohere / OpenAI.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, s: str) -> tuple[int, float]:
        h = int(hashlib.md5(s.encode()).hexdigest(), 16)
        return h % self.dim, 1.0 if (h >> 64) & 1 else -1.0

    def embed_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for t in content_terms(text):
            i, s = self._bucket("w:" + t)
            v[i] += 2.0 * s
            for j in range(len(t) - 2):
                i, s = self._bucket("c:" + t[j : j + 3])
                v[i] += 0.5 * s
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        rows = [self.embed_one(t) for t in texts]
        return np.vstack(rows) if rows else np.zeros((0, self.dim))

    def token_vectors(self, text: str) -> np.ndarray:
        """One vector per content token (for late interaction / late chunking demos)."""
        return self.embed(content_terms(text) or [""])


class DenseIndex:
    def __init__(self, texts: Sequence[str], embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.matrix = self.embedder.embed(texts)

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self.embedder.embed_one(query)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# --------------------------------------------------------------------------- fusion + retriever


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class Retriever:
    """Hybrid BM25 + dense retriever over chunks, fused with RRF."""

    def __init__(self, chunks: Sequence[Chunk] | None = None,
                 embedder: HashingEmbedder | None = None) -> None:
        self.chunks = list(chunks if chunks is not None else chunk_docs())
        self.by_id = {c.id: c for c in self.chunks}
        texts = [f"{c.title}. {c.text}" for c in self.chunks]
        self.bm25 = BM25(texts)
        self.dense = DenseIndex(texts, embedder)

    def _rank(self, scores: np.ndarray, k: int) -> list[str]:
        order = np.argsort(-scores)[:k]
        return [self.chunks[i].id for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 4, mode: str = "hybrid",
               where: Callable[[Chunk], bool] | None = None) -> list[Chunk]:
        pool = 4 * k
        sparse = self._rank(self.bm25.scores(query), pool)
        dense = self._rank(self.dense.scores(query), pool)
        ids = {"sparse": sparse, "dense": dense}.get(mode) or [i for i, _ in rrf([sparse, dense])]
        hits = [self.by_id[i] for i in ids]
        if where is not None:
            hits = [c for c in hits if where(c)]
        return hits[:k]


# --------------------------------------------------------------------------- LLMs


class LLM(Protocol):
    def __call__(self, prompt: str, *, task: str = "answer", system: str = "",
                 max_tokens: int = 16000) -> str: ...


#: Jobs that a small, fast model does well (routing, grading, extraction, rewriting).
SMALL_TASKS = {
    "grade", "classify", "rewrite", "queries", "step_back", "decompose", "filters",
    "entities", "extract_graph", "keywords", "judge", "confidence", "propositions",
    "context", "questions", "notes", "summarize", "memory_clues", "map", "sufficient",
    "verify_questions", "verify",
}


class ClaudeLLM:
    """Anthropic Messages API client (``pip install anthropic``; set ``RAG_LLM=claude``).

    * Opus 5.5 for answers, planning and synthesis; Haiku 5.5 for SMALL_TASKS.
    * Thinking is adaptive on these models; depth is set with ``output_config.effort``.
    * The stable system prompt is a prompt-cache breakpoint.
    * Opus requests opt into server-side refusal fallbacks (``fallbacks="default"``).
    """

    def __init__(self, big: str = "claude-opus-5-5", small: str = "claude-haiku-5-5") -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.big, self.small = big, small

    def __call__(self, prompt: str, *, task: str = "answer", system: str = "",
                 max_tokens: int = 16000) -> str:
        small = task in SMALL_TASKS
        params: dict[str, Any] = {
            "model": self.small if small else self.big,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": "low" if small else "medium"},
        }
        if system:  # cache breakpoint: pays off once the stable prefix is >= ~1-4k tokens
            params["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        if small:
            msg = self.client.messages.create(**params)
        else:
            msg = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **params)
        if msg.stop_reason == "refusal":
            return "I can't help with that request."
        return "".join(b.text for b in msg.content if b.type == "text")


def get_llm() -> LLM:
    """``RAG_LLM=claude`` -> real Claude calls; anything else -> offline double."""
    if os.getenv("RAG_LLM", "offline").lower() == "claude":
        return ClaudeLLM()
    from offline_llm import OfflineLLM  # deterministic test double, see offline_llm.py

    return OfflineLLM()


# --------------------------------------------------------------------------- generation

ANSWER_SYSTEM = (
    "You answer questions using only the provided sources. Cite every claim with the "
    "source id in square brackets, like [d01#0]. If the sources do not contain the answer, "
    "say you don't know. Be concise."
)


def format_context(chunks: Sequence[Chunk]) -> str:
    return "\n".join(f'<source id="{c.id}" title="{c.title}">{c.text}</source>' for c in chunks)


def generate_answer(llm: LLM, question: str, chunks: Sequence[Chunk],
                    extra: str = "") -> str:
    prompt = (f"<context>\n{format_context(chunks)}\n</context>\n{extra}"
              f"<question>{question}</question>")
    return llm(prompt, task="answer", system=ANSWER_SYSTEM)


def show(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    if not isinstance(value, str):
        value = json.dumps(value, default=str)
        if len(value) > 100:
            value = json.dumps(json.loads(value), indent=1)
    print(value)
