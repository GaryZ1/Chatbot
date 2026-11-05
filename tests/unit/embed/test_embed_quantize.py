from __future__ import annotations

import numpy as np
import pytest

from chatbot.embed import (
    BinaryRescoreIndex,
    Int8Quantizer,
    Int8RescoreIndex,
    binary_quantize,
    hamming_distances,
    l2_normalize,
    matryoshka_truncate,
    recall_at_k,
    storage_bytes,
    top_k_exact,
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    # Clustered data, closer to real embeddings than isotropic noise.
    centers = rng.normal(size=(20, 128))
    x = centers[rng.integers(0, 20, 3000)] + 0.6 * rng.normal(size=(3000, 128))
    q = centers[rng.integers(0, 20, 50)] + 0.6 * rng.normal(size=(50, 128))
    return l2_normalize(x.astype(np.float32)), l2_normalize(q.astype(np.float32))


def test_int8_roundtrip_and_recall(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, qs = data
    quant = Int8Quantizer.calibrate(x)
    codes = quant.quantize(x)
    assert codes.dtype == np.int8 and codes.nbytes == x.nbytes // 4
    err = np.abs(quant.dequantize(codes) - x).mean()
    assert err < 0.01
    np.testing.assert_allclose(quant.scores(qs[0], codes), quant.dequantize(codes) @ qs[0], rtol=1e-3, atol=1e-3)
    idx = Int8RescoreIndex(x, rescore_top=50, quantizer=quant)
    recalls = [recall_at_k(idx.search(q, 10)[0], top_k_exact(q, x, 10)) for q in qs]
    assert np.mean(recalls) >= 0.98


def test_binary_hamming_and_rescore_recall(data: tuple[np.ndarray, np.ndarray]) -> None:
    x, qs = data
    codes = binary_quantize(x)
    assert codes.shape == (3000, 16) and codes.dtype == np.uint8
    d = hamming_distances(codes[0], codes)
    assert d[0] == 0 and d.max() <= 128
    manual = int(np.sum((x[0] > 0) != (x[1] > 0)))
    assert d[1] == manual
    idx = BinaryRescoreIndex(x, rescore_top=200)
    recalls = [recall_at_k(idx.search(q, 10)[0], top_k_exact(q, x, 10)) for q in qs]
    assert np.mean(recalls) >= 0.9
    # Without rescoring (pure Hamming) recall is noticeably lower.
    raw = BinaryRescoreIndex(x, rescore_top=10)
    raw_recall = np.mean([recall_at_k(raw.search(q, 10)[0], top_k_exact(q, x, 10)) for q in qs])
    assert raw_recall < np.mean(recalls)


def test_matryoshka_and_storage() -> None:
    v = l2_normalize(np.arange(1, 9, dtype=np.float32).reshape(2, 4))
    t = matryoshka_truncate(v, 2)
    assert t.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-6)
    with pytest.raises(ValueError):
        matryoshka_truncate(v, 5)
    assert storage_bytes(10_000_000, 1024) == 40_960_000_000
    assert storage_bytes(10_000_000, 1024, "binary") == 1_280_000_000
