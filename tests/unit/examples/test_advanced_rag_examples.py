"""Every advanced-RAG example (docs/20-advanced-rag-catalog.md) runs offline, and the
code listings in the doc match the example files."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "advanced_rag"
sys.path.insert(0, str(EXAMPLES))

from run_all import METHODS  # noqa: E402


def test_every_example_file_is_registered() -> None:
    files = {p.stem for p in EXAMPLES.glob("*.py")}
    support = {"rag_kit", "offline_llm", "run_all", "build_doc"}
    assert files - support == set(METHODS)


@pytest.mark.parametrize("name", METHODS)
def test_example_runs_offline(name: str, monkeypatch, capsys) -> None:
    monkeypatch.delenv("RAG_LLM", raising=False)
    importlib.import_module(name).demo()
    assert "===" in capsys.readouterr().out


def test_doc_listings_are_in_sync() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "build_doc.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
