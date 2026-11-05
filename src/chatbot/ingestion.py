"""Offline / indexing plane (docs/00 §2 steps 1-7):

connector -> parse -> clean -> poisoning scan -> dedupe -> chunk -> enrich -> embed + index.

Children (``CHUNK``/``QUESTION``/``SUMMARY``/``PROPOSITION``) are embedded and
BM25-indexed; ``PARENT`` sections go to the parent doc store only (small-to-big).
Tombstones delete a document everywhere, including cached answers that cited it.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from chatbot.build import Components
from chatbot.chunking.base import achunk_any
from chatbot.core.types import ACL, ChunkKind, Document
from chatbot.index.lifecycle import IndexAction
from chatbot.ingest import DedupStatus, FilesystemConnector, SourceRecord, SyncState
from chatbot.ingest.connectors import Connector

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    quarantined: list[str] = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    chunks: int = 0

    def merge(self, other: IngestReport) -> None:
        self.added += other.added
        self.updated += other.updated
        self.deleted += other.deleted
        self.skipped += other.skipped
        self.quarantined += other.quarantined
        self.duplicates.update(other.duplicates)
        self.failed.update(other.failed)
        self.chunks += other.chunks


class IngestionPipeline:
    def __init__(self, components: Components) -> None:
        self.c = components
        self._known_hashes: dict[str, tuple[str, float]] = {}

    def filesystem_connector(self, root: str | Path) -> FilesystemConnector:
        cfg = self.c.settings.ingest
        default_acl = ACL(public=cfg.default_public, allow_groups=list(cfg.default_acl_groups))
        name = "fs-" + hashlib.sha256(str(Path(root).resolve()).encode()).hexdigest()[:12]
        return FilesystemConnector(
            root,
            include=cfg.include,
            exclude=cfg.exclude,
            default_acl=default_acl,
            tenant_id=cfg.tenant_id,
            id_prefix="file",
            name=name,
        )

    def _state_path(self, connector: Connector) -> Path:
        return Path(self.c.settings.index.path) / f"sync_{connector.name}.json"

    async def sync(self, connector: Connector, *, full: bool = False) -> IngestReport:
        """Run one sync of ``connector``; persists sync state and local indexes."""
        state_path = self._state_path(connector)
        state = SyncState.load(state_path, connector.name)
        report = IngestReport()
        sem = asyncio.Semaphore(max(1, self.c.settings.ingest.concurrency))
        tasks: list[asyncio.Task[IngestReport]] = []

        async def one(rec: SourceRecord) -> IngestReport:
            async with sem:
                return await self.process(rec)

        async for record in connector.sync(state, full=full):
            tasks.append(asyncio.create_task(one(record)))
        for r in await asyncio.gather(*tasks):
            report.merge(r)
        state.save(state_path)
        self.c.save()
        log.info(
            "ingest %s: +%d ~%d -%d skipped=%d quarantined=%d failed=%d",
            connector.name,
            report.added,
            report.updated,
            report.deleted,
            report.skipped,
            len(report.quarantined),
            len(report.failed),
        )
        return report

    async def ingest_paths(
        self, paths: Sequence[str | Path], *, full: bool = False
    ) -> IngestReport:
        total = IngestReport()
        for p in paths:
            total.merge(await self.sync(self.filesystem_connector(p), full=full))
        return total

    async def _delete(self, doc_id: str) -> None:
        await self.c.writer.delete_document(doc_id)
        await self.c.parents.delete_doc(doc_id)
        self.c.dedup.remove(doc_id)
        if self.c.cache is not None:
            await self.c.cache.invalidate_doc(doc_id)

    async def process(self, record: SourceRecord) -> IngestReport:
        report = IngestReport()
        try:
            doc = await self.c.parser.aparse(record)
            if doc.deleted:
                await self._delete(doc.doc_id)
                report.deleted += 1
                return report
            return await self.index_document(doc)
        except Exception as exc:  # one bad document must not abort the sync
            log.exception("failed to ingest %s", record.uri)
            report.failed[record.doc_id] = f"{type(exc).__name__}: {exc}"
            return report

    async def index_document(self, doc: Document) -> IngestReport:
        report = IngestReport()
        poison = self.c.poisoning.scan(doc, known_hashes=self._known_hashes)
        if poison.quarantine:
            log.warning("quarantined %s: %s", doc.doc_id, "; ".join(poison.reasons))
            report.quarantined.append(doc.doc_id)
            await self._delete(doc.doc_id)
            return report
        doc.metadata["authority"] = poison.authority
        self._known_hashes[doc.content_hash] = (doc.doc_id, poison.authority)

        dup = self.c.dedup.check(doc)
        if dup.status is not DedupStatus.UNIQUE and dup.duplicate_of:
            log.info("skip %s: %s duplicate of %s", doc.doc_id, dup.status, dup.duplicate_of)
            report.duplicates[doc.doc_id] = dup.duplicate_of
            report.skipped += 1
            return report

        chunks = await achunk_any(self.c.chunker, doc)
        parents = [c for c in chunks if c.kind is ChunkKind.PARENT]
        searchable = [c for c in chunks if c.kind is not ChunkKind.PARENT]
        searchable = await self.c.enricher.run(doc, searchable)
        existed = bool(await self.c.vector_store.by_doc(doc.doc_id))
        result = await self.c.writer.index_document(doc, searchable)
        if result.action is IndexAction.SKIPPED:
            report.skipped += 1
            return report
        await self.c.parents.replace_doc(doc.doc_id, parents)
        if self.c.cache is not None:
            await self.c.cache.invalidate_doc(doc.doc_id)
        if existed:
            report.updated += 1
        else:
            report.added += 1
        report.chunks += len(searchable)
        return report
