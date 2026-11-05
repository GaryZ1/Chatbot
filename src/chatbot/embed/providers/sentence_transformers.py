"""Self-hosted bi-encoders via sentence-transformers (docs/04-embeddings.md §3).

``sentence-transformers`` is imported lazily on first use (``pip install
chatbot[local-models]``). Inference runs in a worker thread so the event loop
stays responsive. Query/document prefixes default to the model's known
instructions (e.g. E5 ``"query: "``) — see :data:`chatbot.embed.base.KNOWN_PREFIXES`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import numpy as np

from chatbot.embed.base import BaseEmbedder

log = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        dim: int | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        truncate_dim: int | None = None,
        version: str = "1",
        trust_remote_code: bool = False,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.truncate_dim = truncate_dim
        self.trust_remote_code = trust_remote_code
        self._model: Any | None = model
        self._load_lock = threading.Lock()
        if dim is None:
            dim = truncate_dim or int(self._load().get_sentence_embedding_dimension())
        super().__init__(
            name=model_name,
            dim=dim,
            version=version,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            batch_size=batch_size,
        )

    def _load(self) -> Any:
        with self._load_lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - optional dependency
                    raise ImportError(
                        "SentenceTransformerEmbedder requires `pip install sentence-transformers`"
                    ) from exc
                log.info("loading sentence-transformers model %s", self.model_name)
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    truncate_dim=self.truncate_dim,
                    trust_remote_code=self.trust_remote_code,
                )
            return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        out = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return np.asarray(out, dtype=np.float32)

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        return await asyncio.to_thread(self._encode, texts)
