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
	chatbot --config configs/pipeline.yaml ingest data/sample_corpus

eval:
	chatbot --config configs/ci.yaml eval --dataset eval/datasets/golden_dev.jsonl eval/datasets/unanswerable.jsonl eval/datasets/adversarial.jsonl --gate configs/eval_gate.yaml --out .ci/reports

serve:
	uvicorn chatbot.api.app:create_app --factory --reload --port 8000
