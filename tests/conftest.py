import shutil
from pathlib import Path

import pytest

from chatbot.config import Settings
from chatbot.core.types import ACL
from chatbot.ingest import write_acl_sidecar
from chatbot.service import ChatbotService

CORPUS = Path(__file__).resolve().parents[1] / "data" / "sample_corpus"


def make_settings(tmp_path: Path, **overrides) -> Settings:
    data = {
        "llm": {"provider": "fake"},
        "ingest": {"default_acl_groups": ["employees"]},
        "embedding": {"name": "hashing", "params": {"dim": 256}},
        "index": {"path": str(tmp_path / "index")},
        "ops": {"rate_per_s": 1000, "burst": 1000},
        "obs": {"trace_path": None, "feedback_path": None},
    }
    for k, v in overrides.items():
        data.setdefault(k, {}).update(v)
    return Settings.model_validate(data)


@pytest.fixture
def settings_factory(tmp_path: Path):
    return lambda **kw: make_settings(tmp_path, **kw)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Sample corpus; the security policy is restricted to the security team."""
    root = tmp_path / "corpus"
    shutil.copytree(CORPUS, root)
    write_acl_sidecar(root / "security-policy.md", ACL(allow_groups=["security"]))
    return root


@pytest.fixture
async def service(tmp_path: Path, corpus: Path):
    svc = ChatbotService.from_settings(make_settings(tmp_path))
    report = await svc.ingestion.ingest_paths([corpus])
    assert report.added == 4 and not report.failed
    yield svc
    await svc.aclose()
