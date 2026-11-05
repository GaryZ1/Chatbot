"""Vector compression (docs/04-embeddings.md §4–5, docs/05-indexing-vector-stores.md §1).

* **int8 scalar quantisation** — per-dimension ranges calibrated on a sample
  (percentile clipping), 4× smaller, ~1% quality loss.
* **binary (sign) quantisation** — 1 bit per dimension packed into bytes, 32×
  smaller, searched by Hamming distance.
* **rescoring** — the production pattern: compressed codes for the candidate scan,
  then rescore the top 100–500 with float vectors (recovers ~95–99% of quality).
* **Matryoshka truncation** — keep the first ``d`` dims of an MRL model and
  re-normalise.

Storage math: ``N × dims × bytes`` (:func:`storage_bytes`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chatbot.embed.base import l2_normalize

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def storage_bytes(n_vectors: int, dims: int, dtype: str = "float32") -> int:
    """Raw vector storage, e.g. 10M × 1024 float32 = 41 GB, int8 = 10 GB, binary = 1.3 GB."""
    per = {"float32": 4.0, "float16": 2.0, "int8": 1.0, "binary": 1 / 8}
    if dtype not in per:
        raise ValueError(f"unknown dtype {dtype!r}")
    return int(n_vectors * dims * per[dtype])


# --------------------------------------------------------------------------- Matryoshka


def matryoshka_truncate(vectors: np.ndarray, dims: int) -> np.ndarray:
    """Keep the leading ``dims`` coordinates (MRL models only) and L2-renormalise."""
    arr = np.asarray(vectors, dtype=np.float32)
    full = arr.shape[-1]
    if not 0 < dims <= full:
        raise ValueError(f"dims must be in (0, {full}]")
    return l2_normalize(arr[..., :dims])


# --------------------------------------------------------------------------- int8


@dataclass
class Int8Quantizer:
    """Per-dimension affine int8 quantiser: ``code = round((x - lo) / scale) - 128``."""

    lo: np.ndarray
    scale: np.ndarray

    @classmethod
    def calibrate(cls, sample: np.ndarray, *, clip_percentile: float = 0.5) -> Int8Quantizer:
        """Fit ranges on a representative sample, clipping ``clip_percentile`` % per tail."""
        arr = np.asarray(sample, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError("calibration sample must be a non-empty 2-D array")
        if not 0 <= clip_percentile < 50:
            raise ValueError("clip_percentile must be in [0, 50)")
        lo = np.percentile(arr, clip_percentile, axis=0).astype(np.float32)
        hi = np.percentile(arr, 100 - clip_percentile, axis=0).astype(np.float32)
        scale = np.maximum(hi - lo, 1e-8) / 255.0
        return cls(lo=lo, scale=scale.astype(np.float32))

    def quantize(self, vectors: np.ndarray) -> np.ndarray:
        arr = np.asarray(vectors, dtype=np.float32)
        q = np.round((arr - self.lo) / self.scale) - 128.0
        return np.clip(q, -128, 127).astype(np.int8)

    def dequantize(self, codes: np.ndarray) -> np.ndarray:
        return ((codes.astype(np.float32) + 128.0) * self.scale + self.lo).astype(np.float32)

    def scores(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Approximate dot products of a float query against int8 codes (asymmetric)."""
        q = np.asarray(query, dtype=np.float32)
        # x ≈ (c + 128) * scale + lo  ⇒  q·x = (c + 128)·(q * scale) + q·lo
        return (codes.astype(np.float32) + 128.0) @ (q * self.scale) + float(q @ self.lo)


# --------------------------------------------------------------------------- binary


def binary_quantize(vectors: np.ndarray) -> np.ndarray:
    """Sign quantisation packed to bits: ``(n, dims) float → (n, ceil(dims/8)) uint8``."""
    arr = np.asarray(vectors)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.packbits(arr > 0, axis=-1)


def hamming_distances(query_code: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Hamming distance between one packed query code and packed codes ``(n, bytes)``."""
    q = np.asarray(query_code, dtype=np.uint8).reshape(-1)
    xor = np.bitwise_xor(codes, q)
    return _POPCOUNT[xor].sum(axis=1, dtype=np.int32)


def hamming_search(query_code: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` codes closest to ``query_code`` (ascending distance)."""
    if k <= 0 or codes.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    d = hamming_distances(query_code, codes)
    k = min(k, d.shape[0])
    idx = np.argpartition(d, k - 1)[:k]
    return idx[np.argsort(d[idx], kind="stable")]


def rescore(
    query: np.ndarray, candidate_idx: np.ndarray, float_vectors: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Re-rank candidates by exact dot product with float vectors → ``(indices, scores)``."""
    if candidate_idx.size == 0:
        return candidate_idx, np.zeros(0, dtype=np.float32)
    scores = float_vectors[candidate_idx] @ np.asarray(query, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:k]
    return candidate_idx[order], scores[order]


def top_k_exact(query: np.ndarray, vectors: np.ndarray, k: int) -> np.ndarray:
    """Flat exact search: ground truth for recall tests (docs/05 §1)."""
    scores = vectors @ np.asarray(query, dtype=np.float32)
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class BinaryRescoreIndex:
    """Binary codes in RAM for the scan, float vectors for rescoring the top ``oversample×k``."""

    def __init__(self, vectors: np.ndarray, *, rescore_top: int = 200) -> None:
        self.vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self.codes = binary_quantize(self.vectors)
        self.rescore_top = rescore_top

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = l2_normalize(np.asarray(query, dtype=np.float32))
        cand = hamming_search(binary_quantize(q)[0], self.codes, max(self.rescore_top, k))
        return rescore(q, cand, self.vectors, k)


class Int8RescoreIndex:
    """int8 codes for the scan, float rescoring of the top ``rescore_top``."""

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        rescore_top: int = 100,
        quantizer: Int8Quantizer | None = None,
    ) -> None:
        self.vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        self.quantizer = quantizer or Int8Quantizer.calibrate(self.vectors)
        self.codes = self.quantizer.quantize(self.vectors)
        self.rescore_top = rescore_top

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = l2_normalize(np.asarray(query, dtype=np.float32))
        approx = self.quantizer.scores(q, self.codes)
        n = min(max(self.rescore_top, k), approx.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        cand = np.argpartition(-approx, n - 1)[:n]
        return rescore(q, cand, self.vectors, k)


def recall_at_k(approx: np.ndarray, exact: np.ndarray) -> float:
    """|approx ∩ exact| / |exact| — the HNSW/quantisation acceptance metric (aim ≥ 0.95)."""
    if exact.size == 0:
        return 1.0
    return len(set(approx.tolist()) & set(exact.tolist())) / exact.size
