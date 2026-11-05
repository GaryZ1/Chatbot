"""Source connectors and sync state (docs/01-ingestion-parsing.md §1).

A connector turns a source system into a stream of :class:`SourceRecord` objects.
Sync follows the doc's three modes:

1. **Full crawl** (initial load) – every document is emitted.
2. **Incremental** – only documents changed since the last sync are emitted. The
   filesystem connector uses ``modified_since`` (mtime) as a cheap pre-check and a
   content hash to ignore touches that did not change the content.
3. **Reconciliation** – documents known to the :class:`SyncState` but no longer
   present at the source are emitted as **tombstones** (``deleted=True``) so the
   index can drop them within the deletion SLA.

ACLs are pulled *with* every document (here: from an ``<file>.acl.json`` sidecar,
a resolver callback or a default), and provenance (``source_uri``, ``modified``,
``content_hash``) is carried on each record.

State updates are *acknowledged lazily*: a record's entry in the state is only
updated when the consumer asks the async generator for the next record, i.e.
after the record has been processed. If processing crashes, the record is
re-emitted on the next sync.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import mimetypes
import os
import tempfile
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from chatbot.core.types import ACL, utcnow

logger = logging.getLogger(__name__)

#: Sidecar suffix holding a JSON-serialised :class:`ACL` for a file.
ACL_SIDECAR_SUFFIX = ".acl.json"

_EXTRA_MIME: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".eml": "message/rfc822",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/x-typescript",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".java": "text/x-java",
    ".json": "application/json",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".csv": "text/csv",
}


def guess_mime(path: str | Path) -> str:
    """MIME type from the file extension, with sane defaults for docs and code."""
    suffix = Path(path).suffix.lower()
    if suffix in _EXTRA_MIME:
        return _EXTRA_MIME[suffix]
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def content_hash_bytes(data: bytes) -> str:
    """SHA-256 of the raw bytes (change detection; text-level dedup uses ``sha256_text``)."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


class SourceRecord(BaseModel):
    """One document as fetched from a source system, before parsing.

    Exactly one of ``content`` (bytes) or ``text`` is normally set; tombstones have
    neither. ``doc_id`` is stable across versions of the same source document.
    """

    doc_id: str
    uri: str
    content: bytes | None = None
    text: str | None = None
    mime: str = "application/octet-stream"
    modified: datetime | None = None
    acl: ACL = Field(default_factory=ACL)
    tenant_id: str = "default"
    title: str = ""
    author: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    deleted: bool = False

    @property
    def extension(self) -> str:
        return Path(self.uri.split("?", 1)[0]).suffix.lower()

    def as_bytes(self) -> bytes:
        if self.content is not None:
            return self.content
        return (self.text or "").encode("utf-8")

    def as_text(self, encoding: str = "utf-8") -> str:
        """Decoded text; undecodable bytes become U+FFFD (counted by parse-quality scoring)."""
        if self.text is not None:
            return self.text
        if self.content is None:
            return ""
        return self.content.decode(encoding, errors="replace")


#: Alias used in the docs ("raw document").
RawDocument = SourceRecord


class SyncEntry(BaseModel):
    doc_id: str
    content_hash: str
    modified: datetime | None = None
    synced_at: datetime = Field(default_factory=utcnow)


class SyncStats(BaseModel):
    seen: int = 0
    emitted: int = 0
    unchanged: int = 0
    deleted: int = 0
    errors: int = 0


class SyncState(BaseModel):
    """Per-connector sync bookkeeping, persisted as JSON between runs."""

    connector: str
    last_sync: datetime | None = None
    entries: dict[str, SyncEntry] = Field(default_factory=dict)
    last_stats: SyncStats = Field(default_factory=SyncStats)

    @classmethod
    def load(cls, path: str | Path, connector: str) -> SyncState:
        """Load state from ``path``; a missing file yields a fresh state (forces a full crawl)."""
        p = Path(path)
        if not p.exists():
            return cls(connector=connector)
        state = cls.model_validate_json(p.read_text(encoding="utf-8"))
        if state.connector != connector:
            raise ValueError(
                f"sync state at {p} belongs to connector {state.connector!r}, not {connector!r}"
            )
        return state

    def save(self, path: str | Path) -> None:
        """Atomically write the state (temp file + rename) so a crash never corrupts it."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(self.model_dump_json(indent=2))
            os.replace(tmp, p)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


@runtime_checkable
class Connector(Protocol):
    """A source system. ``sync`` yields changed records and tombstones, updating ``state``."""

    name: str

    def sync(self, state: SyncState, *, full: bool = False) -> AsyncIterator[SourceRecord]: ...


AclResolver = Callable[[Path], ACL | None]


class FilesystemConnector:
    """Crawl a directory tree (docs/01 §1).

    Args:
        root: directory to crawl.
        include: glob patterns (relative POSIX paths) to include.
        exclude: glob patterns to skip (sidecars and hidden files are always skipped).
        default_acl: ACL used when neither a sidecar nor the resolver supplies one.
            Defaults to a *non-public, empty* ACL (deny by default).
        acl_resolver: optional callback mapping a path to its ACL.
        tenant_id: tenant stamped on every record.
        max_bytes: files larger than this are skipped (logged).
        id_prefix: prefix for ``doc_id`` (``<prefix>:<relative path>``).
    """

    def __init__(
        self,
        root: str | Path,
        *,
        include: Sequence[str] = ("**/*",),
        exclude: Sequence[str] = (),
        default_acl: ACL | None = None,
        acl_resolver: AclResolver | None = None,
        tenant_id: str = "default",
        max_bytes: int = 50 * 1024 * 1024,
        id_prefix: str = "fs",
        name: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"connector root {self.root} is not a directory")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.include = tuple(include)
        self.exclude = tuple(exclude)
        self.default_acl = default_acl or ACL()
        self.acl_resolver = acl_resolver
        self.tenant_id = tenant_id
        self.max_bytes = max_bytes
        self.id_prefix = id_prefix
        self.name = name or f"filesystem:{self.root}"

    # ------------------------------------------------------------------ helpers

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def doc_id_for(self, rel_path: str) -> str:
        return f"{self.id_prefix}:{rel_path}"

    def _wanted(self, rel: str) -> bool:
        if rel.endswith(ACL_SIDECAR_SUFFIX):
            return False
        if any(part.startswith(".") for part in rel.split("/")):
            return False
        if not any(fnmatch.fnmatch(rel, pat) or pat in ("**/*", "*") for pat in self.include):
            return False
        return not any(fnmatch.fnmatch(rel, pat) for pat in self.exclude)

    def _iter_files(self) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                if p.is_file() and self._wanted(self._rel(p)):
                    out.append(p)
        return out

    def _acl_for(self, path: Path) -> ACL:
        sidecar = path.with_name(path.name + ACL_SIDECAR_SUFFIX)
        if sidecar.exists():
            try:
                return ACL.model_validate_json(sidecar.read_text(encoding="utf-8"))
            except ValueError:
                logger.warning("invalid ACL sidecar %s; falling back to default ACL", sidecar)
        if self.acl_resolver is not None:
            acl = self.acl_resolver(path)
            if acl is not None:
                return acl
        return self.default_acl.model_copy(deep=True)

    def _record(self, path: Path, data: bytes, modified: datetime, digest: str) -> SourceRecord:
        rel = self._rel(path)
        return SourceRecord(
            doc_id=self.doc_id_for(rel),
            uri=path.as_uri(),
            content=data,
            mime=guess_mime(path),
            modified=modified,
            acl=self._acl_for(path),
            tenant_id=self.tenant_id,
            title=path.stem,
            metadata={"path": rel, "size_bytes": len(data), "connector": self.name},
            content_hash=digest,
        )

    # ------------------------------------------------------------------ sync

    async def sync(self, state: SyncState, *, full: bool = False) -> AsyncIterator[SourceRecord]:
        """Yield new/changed records, then tombstones for files that disappeared.

        ``full=True`` re-emits every document (initial crawl or re-index) but still
        reconciles deletes. The state is mutated in place; persist it with
        :meth:`SyncState.save` after the generator is exhausted.
        """
        if state.connector != self.name:
            raise ValueError(f"state is for {state.connector!r}, connector is {self.name!r}")
        stats = SyncStats()
        started = utcnow()
        seen: set[str] = set()
        for path in self._iter_files():
            rel = self._rel(path)
            seen.add(rel)
            stats.seen += 1
            try:
                st = path.stat()
                if st.st_size > self.max_bytes:
                    logger.warning("skipping %s: %d bytes exceeds max_bytes", rel, st.st_size)
                    continue
                modified = datetime.fromtimestamp(st.st_mtime, tz=UTC)
                prev = state.entries.get(rel)
                if not full and prev is not None and prev.modified == modified:
                    stats.unchanged += 1  # modified_since pre-check: no read needed
                    continue
                data = path.read_bytes()
            except OSError as exc:
                stats.errors += 1
                logger.warning("cannot read %s: %s", rel, exc)
                continue
            digest = content_hash_bytes(data)
            if not full and prev is not None and prev.content_hash == digest:
                prev.modified = modified  # touched but unchanged content
                stats.unchanged += 1
                continue
            record = self._record(path, data, modified, digest)
            stats.emitted += 1
            yield record
            # acknowledged: the consumer asked for the next record
            state.entries[rel] = SyncEntry(
                doc_id=record.doc_id, content_hash=digest, modified=modified
            )

        # reconciliation: anything known but not seen is a delete
        for rel in sorted(set(state.entries) - seen):
            entry = state.entries[rel]
            stats.deleted += 1
            yield SourceRecord(
                doc_id=entry.doc_id,
                uri=(self.root / rel).as_uri(),
                tenant_id=self.tenant_id,
                deleted=True,
                content_hash=entry.content_hash,
                metadata={"path": rel, "connector": self.name},
            )
            state.entries.pop(rel, None)
        state.last_sync = started
        state.last_stats = stats
        logger.info("sync %s finished: %s", self.name, stats.model_dump())


def write_acl_sidecar(path: str | Path, acl: ACL) -> Path:
    """Write ``<path>.acl.json`` next to a file (helper for tests and local corpora)."""
    p = Path(path)
    sidecar = p.with_name(p.name + ACL_SIDECAR_SUFFIX)
    sidecar.write_text(json.dumps(acl.model_dump(), indent=2), encoding="utf-8")
    return sidecar
