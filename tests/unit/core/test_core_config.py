import pytest

from chatbot.config import Settings


def test_env_overrides_are_typed(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("retrieval: {final_k: 5}\nllm: {provider: fake}\n")
    s = Settings.load(
        cfg, env={"CHATBOT__RETRIEVAL__FINAL_K": "9", "CHATBOT__GUARD__ENABLED": "false"}
    )
    assert s.retrieval.final_k == 9 and s.guard.enabled is False and s.llm.provider == "fake"


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError):
        Settings.model_validate({"retrieval": {"finalk": 3}})


def test_shipped_configs_validate():
    for name in ("pipeline", "ci"):
        Settings.load(f"configs/{name}.yaml", env={})
