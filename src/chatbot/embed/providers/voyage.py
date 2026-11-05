"""Voyage AI embeddings (docs/04-embeddings.md §3; Matryoshka + int8/binary support).

The ``voyageai`` SDK is imported lazily (``pip install chatbot[voyage]``). Voyage
handles asymmetry server-side via ``input_type="query" | "document"``, so no text
prefixes are added — the input type is the explicit equivalent and is always sent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from chatbot.embed.base import BaseEmbedder

log = logging.getLogger(__name__)

#: Default output dimensions of current Voyage models.
VOYAGE_DIMS: dict[str, int] = {
    "voyage-3-large": 1024,
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-4": 1024,
    "voyage-4-large": 1024,
    "voyage-4-lite": 1024,
    "voyage-code-3": 1024,
    "voyage-3": 1024,
    "voyage-3-lite": 512,
}


class VoyageEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "voyage-3-large",
        *,
        api_key: str | None = None,
        output_dimension: int | None = None,
        batch_size: int = 128,
        version: str = "1",
        max_retries: int = 3,
        client: Any | None = None,
    ) -> None:
        dim = output_dimension or VOYAGE_DIMS.get(model)
        if dim is None:
            raise ValueError(f"unknown Voyage model {model!r}: pass output_dimension explicitly")
        super().__init__(
            name=model,
            dim=dim,
            version=version,
            query_prefix="",
            document_prefix="",
            batch_size=min(batch_size, 1000),
        )
        self.model = model
        self.output_dimension = output_dimension
        self.api_key = api_key
        self.max_retries = max_retries
        self._client: Any | None = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("VoyageEmbedder requires `pip install voyageai`") from exc
            key = self.api_key or os.environ.get("VOYAGE_API_KEY")
            self._client = voyageai.AsyncClient(api_key=key, max_retries=self.max_retries)
        return self._client

    async def _embed_batch(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input_type": "query" if is_query else "document",
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension
        result = await self._get_client().embed(texts, **kwargs)
        return np.asarray(result.embeddings, dtype=np.float32)
