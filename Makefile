.PHONY: install lint fmt type test eval serve ingest

install:
	pip install -e ".[dev,anthropic,api]"

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff format src tests
	ruff check --fix src tests

type:
	mypy

test:
	pytest

ingest:
	chatbot ingest --config configs/pipeline.yaml data/sample_corpus

eval:
	chatbot eval --config configs/ci.yaml --dataset eval/datasets/golden_dev.jsonl --gate configs/eval_gate.yaml

serve:
	uvicorn chatbot.api.app:create_app --factory --reload --port 8000
