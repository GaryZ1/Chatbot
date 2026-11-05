"""Freshness SLO and stale-content tracking (docs/14-production-llmops.md §4).

* **Freshness lag** = ``indexed_at - source_updated_at``: how long a change in the
  source takes to become searchable. The SLO is "p95 lag <= ``slo_s``".
* **Pending** documents changed in the source but not (re)indexed yet; their lag is
  measured against *now* and counts against the SLO.
* **Stale** documents have not been updated in the source for ``stale_after_days``
  (a content-owner review signal; feeds failure code F12).
* **Duplicates**: identical content under several doc ids (conflicting / low-authority copies).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from chatbot.core.types import Document

logger = logging.getLogger(__name__)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (kept local so ops does not depend on obs)."""
    s = sorted(values)
    if not s:
        return 0.0
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class FreshnessRecord(BaseModel):
    doc_id: str
    source_uri: str = ""
    title: str = ""
    source_updated_at: datetime
    indexed_at: datetime | None = None
    content_hash: str = ""

    def lag_s(self, now: datetime) -> float:
        end = _utc(self.indexed_at) if self.indexed_at else now
        return max(0.0, (end - _utc(self.source_updated_at)).total_seconds())

    @property
    def pending(self) -> bool:
        return self.indexed_at is None or _utc(self.indexed_at) < _utc(self.source_updated_at)


class FreshnessReport(BaseModel):
    generated_at: datetime
    documents: int
    slo_s: float
    lag_p50_s: float
    lag_p95_s: float
    lag_max_s: float
    slo_attainment: float  # share of documents within the SLO
    slo_met: bool  # p95 within the SLO
    violations: list[str] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    duplicates: dict[str, list[str]] = Field(default_factory=dict)


class FreshnessTracker:
    """In-process freshness ledger fed by the ingestion pipeline."""

    def __init__(
        self,
        *,
        slo_s: float = 3600.0,
        stale_after_days: int = 180,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if slo_s <= 0 or stale_after_days <= 0:
            raise ValueError("slo_s and stale_after_days must be positive")
        self.slo_s = slo_s
        self.stale_after = timedelta(days=stale_after_days)
        self._now = now
        self._records: dict[str, FreshnessRecord] = {}

    def record_source_update(
        self, doc_id: str, updated_at: datetime, *, source_uri: str = "", title: str = ""
    ) -> None:
        """A connector saw a change in the source (indexing may follow later)."""
        rec = self._records.get(doc_id)
        if rec is None:
            self._records[doc_id] = FreshnessRecord(
                doc_id=doc_id, source_uri=source_uri, title=title, source_updated_at=updated_at
            )
            return
        if _utc(updated_at) > _utc(rec.source_updated_at):
            rec.source_updated_at = updated_at
        rec.source_uri = source_uri or rec.source_uri
        rec.title = title or rec.title

    def record_indexed(
        self,
        doc_id: str,
        *,
        source_updated_at: datetime | None = None,
        indexed_at: datetime | None = None,
        content_hash: str = "",
        source_uri: str = "",
        title: str = "",
    ) -> FreshnessRecord:
        """The document became searchable at ``indexed_at`` (default: now)."""
        rec = self._records.get(doc_id)
        if rec is None:
            if source_updated_at is None:
                raise ValueError(f"unknown doc {doc_id}: source_updated_at is required")
            rec = FreshnessRecord(doc_id=doc_id, source_updated_at=source_updated_at)
            self._records[doc_id] = rec
        elif source_updated_at is not None:
            rec.source_updated_at = source_updated_at
        rec.indexed_at = indexed_at or self._now()
        rec.content_hash = content_hash or rec.content_hash
        rec.source_uri = source_uri or rec.source_uri
        rec.title = title or rec.title
        lag = rec.lag_s(self._now())
        if lag > self.slo_s:
            logger.warning("freshness SLO miss: %s lag %.0fs > %.0fs", doc_id, lag, self.slo_s)
        return rec

    def record_document(
        self, doc: Document, *, indexed_at: datetime | None = None
    ) -> FreshnessRecord:
        updated = doc.updated_at or doc.created_at or self._now()
        return self.record_indexed(
            doc.doc_id,
            source_updated_at=updated,
            indexed_at=indexed_at,
            content_hash=doc.content_hash,
            source_uri=doc.source_uri,
            title=doc.title,
        )

    def forget(self, doc_id: str) -> None:
        """A delete propagated: stop tracking the document."""
        self._records.pop(doc_id, None)

    def get(self, doc_id: str) -> FreshnessRecord | None:
        return self._records.get(doc_id)

    def stale_documents(self) -> list[str]:
        cutoff = self._now() - self.stale_after
        return sorted(d for d, r in self._records.items() if _utc(r.source_updated_at) < cutoff)

    def duplicates(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for d, r in self._records.items():
            if r.content_hash:
                groups.setdefault(r.content_hash, []).append(d)
        return {h: sorted(ids) for h, ids in groups.items() if len(ids) > 1}

    def report(self) -> FreshnessReport:
        now = self._now()
        recs = list(self._records.values())
        lags = [r.lag_s(now) for r in recs]
        within = sum(lag <= self.slo_s for lag in lags)
        p95 = percentile(lags, 95) if lags else 0.0
        return FreshnessReport(
            generated_at=now,
            documents=len(recs),
            slo_s=self.slo_s,
            lag_p50_s=round(percentile(lags, 50), 3) if lags else 0.0,
            lag_p95_s=round(p95, 3),
            lag_max_s=round(max(lags), 3) if lags else 0.0,
            slo_attainment=within / len(recs) if recs else 1.0,
            slo_met=p95 <= self.slo_s,
            violations=sorted(
                r.doc_id for r, lag in zip(recs, lags, strict=True) if lag > self.slo_s
            ),
            pending=sorted(r.doc_id for r in recs if r.pending),
            stale=self.stale_documents(),
            duplicates=self.duplicates(),
        )


def stale_documents(
    docs: Iterable[Document], *, months: int = 6, now: datetime | None = None
) -> list[str]:
    """Doc ids whose ``updated_at`` (or ``created_at``) is older than ``months`` (30-day months)."""
    ref = now or datetime.now(UTC)
    cutoff = ref - timedelta(days=30 * months)
    out: list[str] = []
    for d in docs:
        ts = d.updated_at or d.created_at
        if ts is not None and _utc(ts) < cutoff:
            out.append(d.doc_id)
    return sorted(out)
