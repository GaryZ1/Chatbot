"""Answer-level metrics (docs/12-evaluation.md §2, "Label-free / LLM-judged").

* Lexical correctness: SQuAD-style normalised exact match and token F1 against
  the reference answer (cheap, deterministic; use judges for semantic match).
* Abstention accuracy on the unanswerable set (should refuse ↔ did refuse).
* Citation precision / recall following ALCE (Gao et al., 2023): a statement is
  *recalled* when its citations jointly support it; a citation is *precise* unless it
  is irrelevant — it does not support the statement alone **and** the remaining
  citations still support it without it.
* Faithfulness = share of answer claims supported by the context (claim-level,
  computed from judge verdicts, see :mod:`chatbot.evaluation.judges`).
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = str.maketrans("", "", string.punctuation)
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

#: Phrases that indicate the system declined to answer. Pipelines should prefer an
#: explicit ``AnswerStatus.ABSTAINED``; this is the fallback for free-text systems.
ABSTENTION_PATTERNS: tuple[str, ...] = (
    r"\bi (?:do not|don't|cannot|can't) (?:know|find|answer|say)\b",
    r"\b(?:not|no) (?:enough|sufficient) (?:information|context|evidence)\b",
    r"\b(?:couldn't|could not|unable to) find\b",
    r"\bnot (?:mentioned|covered|stated|specified|available) in the (?:provided |available )?(?:documents?|context|sources?)\b",
    r"\bno (?:relevant )?(?:information|documents?) (?:about|on|regarding)\b",
    r"\bi'm not able to (?:answer|find)\b",
)
_ABSTAIN_RE = re.compile("|".join(ABSTENTION_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------------- lexical


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation, articles and extra whitespace (SQuAD)."""
    text = _CITE_RE.sub(" ", text.lower())
    text = text.translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, references: str | Sequence[str]) -> float:
    refs = [references] if isinstance(references, str) else list(references)
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(r) for r in refs) else 0.0


def _f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(ref_tokens)
    return 2 * p * r / (p + r)


def token_f1(prediction: str, references: str | Sequence[str]) -> float:
    """Max token-F1 over references."""
    refs = [references] if isinstance(references, str) else list(references)
    pred = normalize_answer(prediction).split()
    return max((_f1(pred, normalize_answer(r).split()) for r in refs), default=0.0)


def token_recall(prediction: str, reference: str) -> float:
    """Share of reference tokens present in the prediction (robust to verbose answers)."""
    ref = normalize_answer(reference).split()
    if not ref:
        return 0.0
    common = Counter(normalize_answer(prediction).split()) & Counter(ref)
    return sum(common.values()) / len(ref)


# --------------------------------------------------------------------------- abstention


def is_abstention(text: str) -> bool:
    """Heuristic refusal detector for free-text answers."""
    return not text.strip() or bool(_ABSTAIN_RE.search(text))


class AbstentionReport(BaseModel):
    """Confusion counts on ``should_abstain`` (positive class = abstain)."""

    tp: int = 0  # abstained, should have
    fp: int = 0  # abstained, should have answered (false refusal)
    fn: int = 0  # answered, should have abstained (hallucination risk)
    tn: int = 0

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def false_refusal_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    def as_metrics(self) -> dict[str, float]:
        return {
            "abstention_accuracy": self.accuracy,
            "abstention_precision": self.precision,
            "abstention_recall": self.recall,
            "false_refusal_rate": self.false_refusal_rate,
        }


def abstention_report(pairs: Iterable[tuple[bool, bool]]) -> AbstentionReport:
    """``pairs`` of ``(should_abstain, abstained)``."""
    rep = AbstentionReport()
    for should, did in pairs:
        if should and did:
            rep.tp += 1
        elif did:
            rep.fp += 1
        elif should:
            rep.fn += 1
        else:
            rep.tn += 1
    return rep


# --------------------------------------------------------------------------- claims & citations


class CitedStatement(BaseModel):
    text: str
    markers: list[int] = Field(default_factory=list)


def split_statements(answer: str) -> list[CitedStatement]:
    """Split an answer into sentences and the ``[n]`` markers each one carries.

    A trailing citation-only fragment (``... policy. [2]``) is attached to the
    preceding sentence, matching how generators place markers."""
    pieces = [p.strip() for p in _SENT_RE.split(answer.strip()) if p.strip()]
    out: list[CitedStatement] = []
    for piece in pieces:
        markers = [int(n) for grp in _CITE_RE.findall(piece) for n in re.split(r"\s*,\s*", grp)]
        text = _CITE_RE.sub("", piece).strip()
        if not text and out:
            out[-1].markers.extend(m for m in markers if m not in out[-1].markers)
            continue
        if text:
            out.append(CitedStatement(text=text, markers=list(dict.fromkeys(markers))))
    return out


class ClaimSupport(BaseModel):
    """Support information for one cited statement (input to ALCE scoring)."""

    claim: str
    cited_ids: list[str] = Field(default_factory=list)
    #: Sources that *individually* support the claim.
    supporting_ids: set[str] = Field(default_factory=set)
    #: Whether all cited sources *together* support it. ``None`` → approximated as
    #: "at least one cited source supports it".
    jointly_supported: bool | None = None


class CitationScores(BaseModel):
    citation_recall: float
    citation_precision: float
    n_statements: int
    n_citations: int


def citation_scores(claims: Sequence[ClaimSupport]) -> CitationScores:
    """ALCE citation precision/recall from precomputed claim→source support.

    Precision for a citation ``c`` of claim ``s``: precise if ``c`` alone supports
    ``s``; otherwise irrelevant iff the other citations still support ``s`` (then
    ``c`` added nothing). Joint support of the remainder is approximated by "one of
    the remaining citations individually supports ``s``".
    """
    if not claims:
        return CitationScores(
            citation_recall=0.0, citation_precision=0.0, n_statements=0, n_citations=0
        )
    recall_hits = 0
    precise = 0
    n_cit = 0
    for cl in claims:
        cited = list(dict.fromkeys(cl.cited_ids))
        joint = (
            cl.jointly_supported
            if cl.jointly_supported is not None
            else bool(set(cited) & cl.supporting_ids)
        )
        joint = joint and bool(cited)
        recall_hits += int(joint)
        for c in cited:
            n_cit += 1
            if c in cl.supporting_ids:
                precise += 1
                continue
            rest_supports = bool((set(cited) - {c}) & cl.supporting_ids)
            if joint and not rest_supports:
                precise += 1  # needed for joint support
    return CitationScores(
        citation_recall=recall_hits / len(claims),
        citation_precision=precise / n_cit if n_cit else 0.0,
        n_statements=len(claims),
        n_citations=n_cit,
    )


EntailFn = Callable[[str, Sequence[str]], Awaitable[bool]]


async def alce_citation_scores(
    statements: Sequence[CitedStatement],
    sources: Mapping[int, str],
    entails: EntailFn,
) -> CitationScores:
    """Exact ALCE scoring with an entailment oracle ``entails(claim, passages)``
    (an NLI model or an LLM judge). Markers not in ``sources`` count as unsupported."""
    if not statements:
        return CitationScores(
            citation_recall=0.0, citation_precision=0.0, n_statements=0, n_citations=0
        )
    recall_hits = 0
    precise = 0
    n_cit = 0
    for st in statements:
        cited = [m for m in dict.fromkeys(st.markers)]
        valid = [m for m in cited if m in sources]
        joint = bool(valid) and await entails(st.text, [sources[m] for m in valid])
        recall_hits += int(joint)
        for m in cited:
            n_cit += 1
            if m not in sources:
                continue
            if await entails(st.text, [sources[m]]):
                precise += 1
                continue
            rest = [sources[o] for o in valid if o != m]
            if joint and not (rest and await entails(st.text, rest)):
                precise += 1
    return CitationScores(
        citation_recall=recall_hits / len(statements),
        citation_precision=precise / n_cit if n_cit else 0.0,
        n_statements=len(statements),
        n_citations=n_cit,
    )


def faithfulness_score(verdicts: Sequence[bool]) -> float | None:
    """Share of supported claims; ``None`` when the answer makes no claims (e.g. an
    abstention), so it is excluded from averages instead of counting as perfect."""
    if not verdicts:
        return None
    return sum(bool(v) for v in verdicts) / len(verdicts)


def cited_gold_precision(cited_doc_ids: Sequence[str], gold_doc_ids: Iterable[str]) -> float | None:
    """Label-based proxy: share of cited documents that are gold documents."""
    cited = list(dict.fromkeys(cited_doc_ids))
    if not cited:
        return None
    gold = set(gold_doc_ids)
    return sum(d in gold for d in cited) / len(cited)
