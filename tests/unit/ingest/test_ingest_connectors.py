"""Connector sync: full crawl, incremental (modify / touch / delete) and state persistence."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from chatbot.core.types import ACL
from chatbot.ingest import FilesystemConnector, SourceRecord, SyncState, write_acl_sidecar


async def _collect(
    conn: FilesystemConnector, state: SyncState, full: bool = False
) -> list[SourceRecord]:
    return [r async for r in conn.sync(state, full=full)]


def _bump_mtime(path: Path, seconds: int = 10) -> None:
    st = path.stat()
    os.utime(path, (st.st_atime + seconds, st.st_mtime + seconds))


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("# A\n\nAlpha text.", encoding="utf-8")
    (tmp_path / "docs" / "b.txt").write_text("Beta text.", encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("nope", encoding="utf-8")
    write_acl_sidecar(tmp_path / "docs" / "a.md", ACL(allow_groups=["eng"]))
    return tmp_path


async def test_full_sync_emits_all_with_acl_and_provenance(corpus: Path) -> None:
    conn = FilesystemConnector(corpus, default_acl=ACL(public=True), tenant_id="t1")
    state = SyncState(connector=conn.name)
    recs = await _collect(conn, state)
    by_path = {r.metadata["path"]: r for r in recs}
    assert set(by_path) == {"docs/a.md", "docs/b.txt"}  # hidden dirs and sidecars skipped
    a = by_path["docs/a.md"]
    assert a.doc_id == "fs:docs/a.md"
    assert a.mime == "text/markdown"
    assert a.acl.allow_groups == ["eng"] and not a.acl.public
    assert by_path["docs/b.txt"].acl.public
    assert a.tenant_id == "t1"
    assert a.content_hash.startswith("sha256:")
    assert a.modified is not None and a.uri.startswith("file://")
    assert set(state.entries) == {"docs/a.md", "docs/b.txt"}
    assert state.last_sync is not None and state.last_stats.emitted == 2


async def test_incremental_detects_modify_touch_add_delete(corpus: Path) -> None:
    conn = FilesystemConnector(corpus)
    state = SyncState(connector=conn.name)
    await _collect(conn, state)

    # nothing changed → nothing emitted
    assert await _collect(conn, state) == []

    # touch only (mtime changes, content same) → skipped by content hash
    _bump_mtime(corpus / "docs" / "b.txt")
    assert await _collect(conn, state) == []
    assert state.last_stats.unchanged == 2

    # modify + add + delete
    a = corpus / "docs" / "a.md"
    a.write_text("# A\n\nAlpha text, revised.", encoding="utf-8")
    _bump_mtime(a, 20)
    (corpus / "docs" / "c.md").write_text("New doc", encoding="utf-8")
    (corpus / "docs" / "b.txt").unlink()
    recs = await _collect(conn, state)
    live = {r.metadata["path"] for r in recs if not r.deleted}
    tombs = [r for r in recs if r.deleted]
    assert live == {"docs/a.md", "docs/c.md"}
    assert [t.doc_id for t in tombs] == ["fs:docs/b.txt"]
    assert tombs[0].content is None
    assert "docs/b.txt" not in state.entries
    assert state.last_stats.deleted == 1


async def test_full_sync_reemits_unchanged(corpus: Path) -> None:
    conn = FilesystemConnector(corpus)
    state = SyncState(connector=conn.name)
    await _collect(conn, state)
    assert len(await _collect(conn, state, full=True)) == 2


async def test_unacknowledged_record_is_reemitted(corpus: Path) -> None:
    conn = FilesystemConnector(corpus)
    state = SyncState(connector=conn.name)
    gen = conn.sync(state)
    first = await gen.__anext__()
    await gen.aclose()  # consumer crashed while processing `first`
    assert first.metadata["path"] not in state.entries
    recs = await _collect(conn, state)
    assert first.metadata["path"] in {r.metadata["path"] for r in recs}


async def test_state_roundtrip_and_validation(corpus: Path, tmp_path: Path) -> None:
    conn = FilesystemConnector(corpus)
    state = SyncState(connector=conn.name)
    await _collect(conn, state)
    path = tmp_path / ".state" / "fs.json"  # hidden: not crawled
    state.save(path)
    loaded = SyncState.load(path, conn.name)
    assert loaded.entries.keys() == state.entries.keys()
    assert loaded.entries["docs/a.md"].content_hash == state.entries["docs/a.md"].content_hash
    assert await _collect(conn, loaded) == []
    with pytest.raises(ValueError):
        SyncState.load(path, "other")
    assert SyncState.load(tmp_path / "missing.json", "x").entries == {}


async def test_include_exclude_and_resolver(corpus: Path) -> None:
    conn = FilesystemConnector(
        corpus,
        include=["*.md"],
        acl_resolver=lambda p: ACL(allow_users=["u1"]) if p.suffix == ".md" else None,
    )
    recs = await _collect(conn, SyncState(connector=conn.name))
    assert [r.metadata["path"] for r in recs] == ["docs/a.md"]
    # sidecar wins over resolver
    assert recs[0].acl.allow_groups == ["eng"]
    conn2 = FilesystemConnector(corpus, exclude=["docs/a.md"])
    recs2 = await _collect(conn2, SyncState(connector=conn2.name))
    assert [r.metadata["path"] for r in recs2] == ["docs/b.txt"]


def test_invalid_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FilesystemConnector(tmp_path / "missing")


def test_source_record_decoding() -> None:
    r = SourceRecord(doc_id="d", uri="file:///x.txt", content=b"ok \xff")
    assert r.as_text() == "ok �"
    assert r.extension == ".txt"


def test_glob_double_star_matches_root_files():
    from chatbot.ingest.connectors import _glob_match

    assert _glob_match("a.md", "**/*.md")
    assert _glob_match("x/y/a.md", "**/*.md")
    assert not _glob_match("a.txt", "**/*.md")
