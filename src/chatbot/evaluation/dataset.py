"""Golden dataset: schema, JSONL persistence, versioning, frozen splits, label QA.

Implements docs/12-evaluation.md §3 ("Building the golden dataset"):

* labels ``question, reference_answer, gold_doc_ids, gold_chunk_ids, intent,
  difficulty, must_cite, should_abstain`` (+ slice tags and optional history for
  multi-turn cases, docs/11-conversation-memory.md);
* versioning "like code": :meth:`GoldenSet.version_hash` is a content hash;
* a frozen *test* split that never moves: membership is a pure function of the
  example id (salted SHA-256), so adding new examples never reshuffles old ones;
* stratified sampling by intent/difficulty/slice for review queues and quick runs;
* :meth:`GoldenSet.validate` catches label-quality problems before they poison a run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chatbot.core.types import Turn

logger = logging.getLogger(__name__)

#: Salt for the frozen dev/test split. Changing it re-splits every dataset: never do
#: that for an existing golden set (the test split must stay untouched by tuning).
DEFAULT_SPLIT_SALT = "golden-split-v1"
#: Fraction of examples that land in the frozen test split.
DEFAULT_TEST_FRACTION = 0.3
#: Recommended unanswerable share (docs/12-evaluation.md §3: ~10-15%).
UNANSWERABLE_RANGE = (0.10, 0.20)
#: Valid graded-relevance labels (0 = not relevant ... 3 = perfectly relevant).
MAX_GRADE = 3


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


class ExampleSource(StrEnum):
    SME = "sme"
    SYNTHETIC = "synthetic"
    PRODUCTION = "production"
    ADVERSARIAL = "adversarial"


class GoldenExample(BaseModel):
    """One labelled evaluation case (docs/12-evaluation.md §3.3)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str = ""
    gold_doc_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    #: Optional graded relevance (id -> 0..3) for nDCG. Keys may be doc or chunk ids.
    #: Ids in ``gold_*_ids`` without a grade default to grade 1.
    graded_relevance: dict[str, int] = Field(default_factory=dict)
    intent: str = "lookup"
    difficulty: Difficulty = Difficulty.MEDIUM
    must_cite: bool = True
    should_abstain: bool = False
    #: Free-form slice tags, e.g. ``["multi_hop", "lang:de", "policy"]``.
    slices: list[str] = Field(default_factory=list)
    #: Prior conversation turns for multi-turn cases (the question is the last turn).
    history: list[Turn] = Field(default_factory=list)
    language: str = "en"
    source: ExampleSource = ExampleSource.SME
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def _strip_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("question must not be blank")
        return v

    @field_validator("graded_relevance")
    @classmethod
    def _check_grades(cls, v: dict[str, int]) -> dict[str, int]:
        for key, grade in v.items():
            if not 0 <= grade <= MAX_GRADE:
                raise ValueError(f"grade for {key!r} must be in 0..{MAX_GRADE}, got {grade}")
        return v

    # ------------------------------------------------------------------ helpers

    def relevance(self, level: str = "doc") -> dict[str, int]:
        """Graded relevance map at ``doc`` or ``chunk`` level (gold ids default to 1)."""
        if level not in {"doc", "chunk"}:
            raise ValueError("level must be 'doc' or 'chunk'")
        ids, other = (
            (self.gold_doc_ids, set(self.gold_chunk_ids))
            if level == "doc"
            else (self.gold_chunk_ids, set(self.gold_doc_ids))
        )
        out = {i: 1 for i in ids}
        for key, grade in self.graded_relevance.items():
            # A grade belongs to this level unless its key is a gold id of the other level.
            if key not in other:
                out[key] = grade
        return {k: g for k, g in out.items() if g > 0}

    def slice_keys(self) -> list[str]:
        """All slice keys this example contributes to (always includes ``all``)."""
        keys = ["all", f"intent:{self.intent}", f"difficulty:{self.difficulty.value}"]
        keys.append("unanswerable" if self.should_abstain else "answerable")
        if self.history:
            keys.append("multi_turn")
        if self.language != "en":
            keys.append(f"lang:{self.language}")
        keys.append(f"source:{self.source.value}")
        keys.extend(f"tag:{s}" for s in self.slices)
        return list(dict.fromkeys(keys))


class LabelIssue(BaseModel):
    severity: str  # "error" | "warning"
    example_id: str | None
    message: str


def _norm_question(q: str) -> str:
    return re.sub(r"[^\w]+", " ", q.lower()).strip()


def split_of(example_id: str, test_fraction: float = DEFAULT_TEST_FRACTION, salt: str = DEFAULT_SPLIT_SALT) -> Split:
    """Deterministic frozen split assignment from the id hash alone."""
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError("test_fraction must be in [0, 1]")
    digest = hashlib.sha256(f"{salt}|{example_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return Split.TEST if bucket < test_fraction else Split.DEV


class GoldenSet:
    """An ordered, id-unique collection of :class:`GoldenExample`."""

    def __init__(self, examples: Iterable[GoldenExample] = (), name: str = "golden") -> None:
        self.name = name
        self._examples: list[GoldenExample] = []
        self._ids: set[str] = set()
        for ex in examples:
            self.add(ex)

    # ------------------------------------------------------------------ container API

    def __len__(self) -> int:
        return len(self._examples)

    def __iter__(self) -> Iterator[GoldenExample]:
        return iter(self._examples)

    def __getitem__(self, idx: int) -> GoldenExample:
        return self._examples[idx]

    @property
    def examples(self) -> list[GoldenExample]:
        return list(self._examples)

    def get(self, example_id: str) -> GoldenExample | None:
        return next((e for e in self._examples if e.id == example_id), None)

    def add(self, example: GoldenExample, *, replace: bool = False) -> None:
        """Add one example (e.g. a triaged production failure, §3.6)."""
        if example.id in self._ids:
            if not replace:
                raise ValueError(f"duplicate example id {example.id!r}")
            self._examples = [example if e.id == example.id else e for e in self._examples]
            return
        self._ids.add(example.id)
        self._examples.append(example)

    def filter(self, predicate: Callable[[GoldenExample], bool], name: str | None = None) -> GoldenSet:
        return GoldenSet((e for e in self._examples if predicate(e)), name=name or self.name)

    def merge(self, other: GoldenSet, name: str | None = None) -> GoldenSet:
        return GoldenSet([*self._examples, *other], name=name or self.name)

    # ------------------------------------------------------------------ persistence

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> GoldenSet:
        """Load JSONL (one example per line; blank lines ignored)."""
        p = Path(path)
        examples: list[GoldenExample] = []
        with p.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    examples.append(GoldenExample.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as exc:
                    raise ValueError(f"{p}:{lineno}: invalid golden example: {exc}") from exc
        return cls(examples, name=name or p.stem)

    @classmethod
    def load_many(cls, paths: Sequence[str | Path], name: str = "golden") -> GoldenSet:
        out = cls(name=name)
        for path in paths:
            for ex in cls.load(path):
                out.add(ex)
        return out

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for ex in self._examples:
                fh.write(ex.model_dump_json(exclude_defaults=True) + "\n")

    # ------------------------------------------------------------------ versioning

    def version_hash(self) -> str:
        """Order-independent content hash. Any label edit produces a new version."""
        h = hashlib.sha256()
        for ex in sorted(self._examples, key=lambda e: e.id):
            payload = ex.model_dump(mode="json", exclude={"history"})
            payload["history"] = [{"role": t.role.value, "content": t.content} for t in ex.history]
            h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
            h.update(b"\n")
        return "gs-" + h.hexdigest()[:12]

    # ------------------------------------------------------------------ splits

    def split(
        self, which: Split | str, test_fraction: float = DEFAULT_TEST_FRACTION, salt: str = DEFAULT_SPLIT_SALT
    ) -> GoldenSet:
        target = Split(which)
        return self.filter(
            lambda e: split_of(e.id, test_fraction, salt) is target, name=f"{self.name}:{target.value}"
        )

    def dev(self, **kw: Any) -> GoldenSet:
        return self.split(Split.DEV, **kw)

    def test(self, **kw: Any) -> GoldenSet:
        return self.split(Split.TEST, **kw)

    # ------------------------------------------------------------------ sampling / slices

    def stratified_sample(
        self,
        n: int,
        by: Callable[[GoldenExample], str] | str = "intent",
        seed: int = 0,
    ) -> GoldenSet:
        """Proportional stratified sample (largest-remainder allocation, ≥1 per stratum
        when ``n`` allows), deterministic for a given seed."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n >= len(self):
            return GoldenSet(self._examples, name=self.name)
        key_fn = _stratum_key(by)
        strata: dict[str, list[GoldenExample]] = defaultdict(list)
        for ex in self._examples:
            strata[key_fn(ex)].append(ex)
        total = len(self)
        quotas = {k: n * len(v) / total for k, v in strata.items()}
        alloc = {k: math.floor(q) for k, q in quotas.items()}
        if n >= len(strata):
            for k in alloc:
                if alloc[k] == 0:
                    alloc[k] = 1
        remaining = n - sum(alloc.values())
        order = sorted(quotas, key=lambda k: (quotas[k] - math.floor(quotas[k]), k), reverse=True)
        i = 0
        while remaining > 0 and order:
            k = order[i % len(order)]
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
                remaining -= 1
            i += 1
            if i > 10 * len(order) + n:
                break
        while remaining < 0:  # the ≥1 floor over-allocated: trim the largest strata
            k = max(alloc, key=lambda s: (alloc[s], s))
            alloc[k] -= 1
            remaining += 1
        rng = random.Random(seed)
        chosen: set[str] = set()
        for k in sorted(strata):
            members = sorted(strata[k], key=lambda e: e.id)
            chosen.update(e.id for e in rng.sample(members, min(alloc[k], len(members))))
        return self.filter(lambda e: e.id in chosen, name=f"{self.name}:sample{n}")

    def slice_counts(self) -> dict[str, int]:
        c: Counter[str] = Counter()
        for ex in self._examples:
            c.update(ex.slice_keys())
        return dict(sorted(c.items()))

    # ------------------------------------------------------------------ label QA

    def validate(self, corpus_doc_ids: set[str] | None = None) -> list[LabelIssue]:
        """Label-quality checks. ``error`` issues should block a run; warnings are triage."""
        issues: list[LabelIssue] = []
        seen_q: dict[str, str] = {}
        for ex in self._examples:
            nq = _norm_question(ex.question)
            if nq in seen_q:
                issues.append(
                    LabelIssue(
                        severity="warning",
                        example_id=ex.id,
                        message=f"duplicate question (same as {seen_q[nq]!r})",
                    )
                )
            else:
                seen_q[nq] = ex.id
            if ex.should_abstain:
                if ex.gold_doc_ids or ex.gold_chunk_ids:
                    issues.append(
                        LabelIssue(
                            severity="error",
                            example_id=ex.id,
                            message="should_abstain example must not have gold evidence",
                        )
                    )
                if ex.must_cite:
                    issues.append(
                        LabelIssue(
                            severity="warning",
                            example_id=ex.id,
                            message="must_cite is meaningless for an unanswerable example",
                        )
                    )
            else:
                if not ex.gold_doc_ids and not ex.gold_chunk_ids:
                    issues.append(
                        LabelIssue(
                            severity="warning",
                            example_id=ex.id,
                            message="answerable example has no gold evidence (retrieval metrics skipped)",
                        )
                    )
                if not ex.reference_answer.strip():
                    issues.append(
                        LabelIssue(
                            severity="warning",
                            example_id=ex.id,
                            message="answerable example has no reference answer (correctness skipped)",
                        )
                    )
            if ex.gold_chunk_ids and not ex.gold_doc_ids:
                issues.append(
                    LabelIssue(
                        severity="warning",
                        example_id=ex.id,
                        message="gold_chunk_ids without gold_doc_ids: doc-level metrics unavailable",
                    )
                )
            if len(set(ex.gold_doc_ids)) != len(ex.gold_doc_ids):
                issues.append(
                    LabelIssue(severity="warning", example_id=ex.id, message="duplicate gold_doc_ids")
                )
            if corpus_doc_ids is not None:
                missing = [d for d in ex.gold_doc_ids if d not in corpus_doc_ids]
                if missing:
                    issues.append(
                        LabelIssue(
                            severity="error",
                            example_id=ex.id,
                            message=f"gold docs not in corpus: {missing}",
                        )
                    )
            if ex.history and ex.history[-1].role.value == "user" and ex.history[-1].content == ex.question:
                issues.append(
                    LabelIssue(
                        severity="warning",
                        example_id=ex.id,
                        message="history must not repeat the current question as its last turn",
                    )
                )
        if self._examples:
            share = sum(e.should_abstain for e in self._examples) / len(self._examples)
            lo, hi = UNANSWERABLE_RANGE
            if not lo <= share <= hi:
                issues.append(
                    LabelIssue(
                        severity="warning",
                        example_id=None,
                        message=f"unanswerable share {share:.0%} outside recommended {lo:.0%}-{hi:.0%}",
                    )
                )
        return issues


def _stratum_key(by: Callable[[GoldenExample], str] | str) -> Callable[[GoldenExample], str]:
    if callable(by):
        return by
    if by == "slice":
        return lambda e: e.slices[0] if e.slices else "untagged"
    if by not in GoldenExample.model_fields:
        raise ValueError(f"unknown stratification field {by!r}")
    return lambda e: str(getattr(e, by).value if isinstance(getattr(e, by), StrEnum) else getattr(e, by))
