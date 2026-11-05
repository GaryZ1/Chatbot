"""Late chunking (Gunther et al., Jina AI, 2024): run a long-context embedding model over
the WHOLE document first, then mean-pool the contextualised token embeddings inside each
chunk span. Chunks keep document context ("Its" knows it means "Orbital Systems") without
extra LLM calls."""

import re

import numpy as np
from rag_kit import CORPUS, HashingEmbedder, content_terms, cosine, show, split_sentences


def late_chunk_transformers(text: str, spans: list[tuple[int, int]],
                            model_name: str = "jinaai/jina-embeddings-v3") -> np.ndarray:
    """Real implementation (pip install transformers torch). ``spans`` are char offsets."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, truncation=True,
              max_length=8192)
    offsets = enc.pop("offset_mapping")[0]
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state[0]          # contextual token embeddings
    vecs = []
    for start, end in spans:
        mask = (offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > 0)
        v = hidden[mask].mean(0)
        vecs.append((v / v.norm()).numpy())
    return np.vstack(vecs)


def late_chunk_offline(text: str, spans: list[tuple[int, int]], mix: float = 0.6) -> np.ndarray:
    """Offline approximation: 'contextualise' each token by mixing in the document mean,
    which is what self-attention does far more cleverly."""
    emb = HashingEmbedder()
    doc_mean = emb.embed_one(text)
    vecs = []
    for start, end in spans:
        toks = emb.embed(content_terms(text[start:end]) or [""])
        v = (toks + mix * doc_mean).mean(0)
        vecs.append(v / np.linalg.norm(v))
    return np.vstack(vecs)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for s in split_sentences(text):
        start = text.index(s)
        spans.append((start, start + len(s)))
    return spans


def demo() -> None:
    doc = CORPUS[0].text
    spans = sentence_spans(doc)
    chunk = doc[spans[-1][0] : spans[-1][1]]           # "Its main product is Skyline, ..."
    query = "Orbital Systems main product"
    emb = HashingEmbedder()
    naive = cosine(emb.embed_one(query), emb.embed_one(chunk))
    late = cosine(emb.embed_one(query), late_chunk_offline(doc, spans)[-1])
    show("Chunk", re.sub(r"\s+", " ", chunk))
    show("Similarity", {"naive_chunk": round(naive, 3), "late_chunk": round(late, 3)})


if __name__ == "__main__":
    demo()
