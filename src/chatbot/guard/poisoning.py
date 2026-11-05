"""Ingest-time content scanning against knowledge-base poisoning (docs/13-guardrails-security.md §1).

PoisonedRAG (Zou et al. 2024) shows that a handful of crafted passages per target
question can dominate retrieval. Defences applied here, before a document is
indexed:

* injection patterns in the document text (same rules as the retrieval-time scan,
  so attacks are caught once at ingest rather than on every query);
* secret scanning (credentials must never become retrievable);
* provenance / authority scoring from the source location and author, used both
  for quarantine decisions and as a ranking weight ("prefer authoritative sources");
* anomaly flags typical of crafted passages: hidden text, keyword stuffing, a block
  of leading questions (query stuffing), new documents, low-authority near-copies of
  existing content.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from chatbot.core.types import Document
from chatbot.guard.injection import InjectionDetector, strip_invisible
from chatbot.guard.output import content_terms
from chatbot.guard.pii import SECRET_TYPES, PIIDetector
from chatbot.guard.policies import InjectionPolicy, PoisoningPolicy

logger = logging.getLogger(__name__)

_HIDDEN_HTML_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0|opacity\s*:\s*0(?:\.0+)?\b"
    r"|color\s*:\s*(?:white|#fff(?:fff)?|transparent)\b|<!--",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class PoisoningReport(BaseModel):
    doc_id: str
    authority: float
    injection_score: float = 0.0
    injection_rules: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    quarantine: bool = False
    reasons: list[str] = Field(default_factory=list)


class PoisoningScanner:
    """Scores and flags documents at ingest; quarantined docs must not be indexed."""

    def __init__(
        self,
        policy: PoisoningPolicy | None = None,
        *,
        injection: InjectionDetector | None = None,
        pii: PIIDetector | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.policy = policy or PoisoningPolicy()
        self.injection = injection or InjectionDetector(
            InjectionPolicy(chunk_threshold=self.policy.injection_threshold)
        )
        self.pii = pii or PIIDetector(SECRET_TYPES)
        self._now = now

    # ------------------------------------------------------------------ provenance

    def authority(self, doc: Document) -> float:
        """Authority score in [0, 1] from source prefix / domain and author.

        The longest matching ``trusted_sources`` prefix (or domain) wins; trusted
        authors get +0.2; an explicit ``metadata["authority"]`` set by a connector
        is respected when no rule matches.
        """
        p = self.policy
        uri = doc.source_uri or ""
        host = (urlsplit(uri).hostname or "").lower()
        best: tuple[int, float] | None = None
        for key, score in p.trusted_sources.items():
            k = key.lower()
            match = uri.lower().startswith(k) or (host and (host == k or host.endswith("." + k)))
            if match and (best is None or len(k) > best[0]):
                best = (len(k), score)
        if best is not None:
            score = best[1]
        elif isinstance(doc.metadata.get("authority"), int | float):
            score = float(doc.metadata["authority"])
        else:
            score = p.default_authority
        if doc.author and doc.author in p.trusted_authors:
            score += 0.2
        return max(0.0, min(1.0, score))

    @staticmethod
    def ranking_weight(authority: float, strength: float = 0.2) -> float:
        """Multiplicative ranking prior: 1 +/- ``strength`` around neutral authority 0.5."""
        return 1.0 + strength * (2.0 * authority - 1.0)

    # ------------------------------------------------------------------ anomalies

    def _anomalies(self, doc: Document, text: str) -> list[str]:
        p = self.policy
        flags: list[str] = []
        if strip_invisible(text) != text:
            flags.append("hidden_characters")
        if any(b.html and _HIDDEN_HTML_RE.search(b.html) for b in doc.blocks):
            flags.append("hidden_html")
        terms = content_terms(text)
        if len(terms) >= 30:
            _, top = Counter(terms).most_common(1)[0]
            if top / len(terms) > p.max_term_share:
                flags.append("keyword_stuffing")
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
        lead = sentences[:5]
        if len(lead) >= 3:
            q_share = sum(s.endswith("?") for s in lead) / len(lead)
            if q_share >= p.max_leading_question_share:
                flags.append("query_stuffing")
        ref = doc.created_at or doc.updated_at
        if ref is not None:
            ref_utc = ref if ref.tzinfo else ref.replace(tzinfo=UTC)
            if self._now() - ref_utc <= timedelta(days=p.new_doc_days):
                flags.append("new_document")
        return flags

    # ------------------------------------------------------------------ scan

    def scan(
        self,
        doc: Document,
        *,
        known_hashes: Mapping[str, tuple[str, float]] | None = None,
    ) -> PoisoningReport:
        """Scan one document.

        ``known_hashes`` maps ``content_hash -> (doc_id, authority)`` of indexed
        documents; a lower-authority exact copy is flagged.
        """
        p = self.policy
        text = doc.text
        authority = self.authority(doc)
        report = PoisoningReport(doc_id=doc.doc_id, authority=round(authority, 4))

        inj = self.injection.scan_content(text) if text else None
        if inj is not None:
            report.injection_score = inj.score
            report.injection_rules = inj.matched
            if inj.score >= p.injection_threshold:
                report.flags.append("injection_pattern")
                report.quarantine = True
                report.reasons.append(f"injection score {inj.score:.2f} ({', '.join(inj.matched)})")

        secrets = [m for m in self.pii.detect(text) if m.is_secret]
        if secrets:
            report.flags.append("contains_secret")
            if p.quarantine_on_secrets:
                report.quarantine = True
                report.reasons.append(f"{len(secrets)} credential(s) found")

        anomalies = self._anomalies(doc, text)
        report.flags.extend(anomalies)

        if known_hashes:
            other = known_hashes.get(doc.content_hash)
            if other is not None and other[0] != doc.doc_id:
                report.flags.append(f"duplicate_of:{other[0]}")
                if other[1] > authority:
                    report.flags.append("low_authority_copy")

        suspicious = [
            f
            for f in report.flags
            if f not in ("new_document",) and not f.startswith("duplicate_of:")
        ]
        if authority < p.min_authority and suspicious and not report.quarantine:
            report.quarantine = True
            report.reasons.append(f"low authority {authority:.2f} with anomalies {suspicious}")
        if report.quarantine:
            logger.warning("quarantining document %s: %s", doc.doc_id, "; ".join(report.reasons))
        return report

    def scan_many(
        self,
        docs: Iterable[Document],
        *,
        known_hashes: Mapping[str, tuple[str, float]] | None = None,
    ) -> list[PoisoningReport]:
        """Scan a batch; also detects duplicates *within* the batch."""
        seen: dict[str, tuple[str, float]] = dict(known_hashes or {})
        out: list[PoisoningReport] = []
        for d in docs:
            r = self.scan(d, known_hashes=seen)
            out.append(r)
            if not r.quarantine:
                prev = seen.get(d.content_hash)
                if prev is None or prev[1] < r.authority:
                    seen[d.content_hash] = (d.doc_id, r.authority)
        return out


def quarantined_ids(reports: Sequence[PoisoningReport]) -> set[str]:
    return {r.doc_id for r in reports if r.quarantine}
