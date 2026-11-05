"""Cross-encoder reranker (docs/08-reranking.md §1–3).

Uses ``sentence_transformers.CrossEncoder`` (lazy import, ``pip install
chatbot[local-models]``), default ``BAAI/bge-reranker-v2-m3``. Inputs are
``(query, smart_truncate(chunk))`` pairs — reranking sees the contextual embed text
but the original chunk is returned. Inference runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from chatbot.core.tokens import Tokenizer
from chatbot.core.types import ScoredChunk
from chatbot.rerank.base import BaseReranker, smart_truncate

log = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        max_tokens: int = 512,
        batch_size: int = 32,
        device: str | None = None,
        tokenizer: Tokenizer | None = None,
        model: Any | None = None,
        name: str = "cross_encoder",
    ) -> None:
        if max_tokens < 16:
            raise ValueError("max_tokens too small")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = tokenizer
        self.name = name
        self._model: Any | None = model
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise ImportError(
                        "CrossEncoderReranker requires `pip install sentence-transformers`"
                    ) from exc
                log.info("loading cross-encoder %s", self.model_name)
                self._model = CrossEncoder(self.model_name, max_length=self.max_tokens, device=self.device)
            return self._model

    def pairs(self, query: str, candidates: Sequence[ScoredChunk]) -> list[tuple[str, str]]:
        # Reserve roughly a third of the window for the query and special tokens.
        doc_budget = max(self.max_tokens - min(self.max_tokens // 3, 64), 16)
        return [
            (query, smart_truncate(query, c.chunk, doc_budget, tokenizer=self.tokenizer)) for c in candidates
        ]

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = self._load().predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in np.asarray(scores, dtype=np.float32).reshape(-1)]

    async def score(self, query: str, candidates: Sequence[ScoredChunk]) -> list[float]:
        return await asyncio.to_thread(self._predict, self.pairs(query, candidates))
