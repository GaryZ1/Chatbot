"""Visual document retrieval: ColPali (Faysse et al., 2024) and successors (ColQwen2.5,
ColNomic, Jina v4 multi-vector). Embed page IMAGES with a vision-language model into
~1,000 patch vectors per page; score with late-interaction MaxSim against the query's
token vectors; send the top page images to a multimodal LLM to answer. No OCR/parsing
loss on charts, tables and layouts. Benchmarked on ViDoRe v1-v3."""

import base64

import numpy as np
from colbert_maxsim import maxsim
from rag_kit import HashingEmbedder, show

# Offline stand-in: a "page" is a grid of regions; each region becomes patch vectors.
PAGES = {
    "deck_p1.png": ["Q3 2024 results", "bar chart revenue by quarter 14.8 15.6 16.1 17.0 18.0",
                    "revenue grew 22 percent", "Skyline subscriptions"],
    "deck_p2.png": ["Lumen ST-2 datasheet", "mass 180 g power 0.9 W", "photo of star tracker",
                    "built in Gothenburg"],
    "deck_p3.png": ["Org chart", "CEO Mara Quinn", "CTO Theo Brandt", "Lisbon headquarters"],
}
emb = HashingEmbedder(dim=256)


def page_patches(regions: list[str]) -> np.ndarray:
    return np.vstack([emb.token_vectors(r) for r in regions])


def binarize(x: np.ndarray) -> np.ndarray:
    """Sign quantisation: 1 bit per dimension on dense VLM embeddings (32x smaller than
    fp32; sparse offline vectors keep zeros). Rescore the top pages in full precision."""
    return np.sign(x) / np.sqrt(x.shape[1])          # pack with np.packbits for storage


def pool_patches(x: np.ndarray, factor: int = 2) -> np.ndarray:
    """Token pooling: merge patches greedily by similarity, cutting storage ~factor x."""
    keep, used = [], set()
    for i in range(len(x)):
        if i in used:
            continue
        sims = x @ x[i]
        group = [j for j in np.argsort(-sims) if j not in used][:factor]
        used.update(group)
        v = x[group].mean(0)
        keep.append(v / (np.linalg.norm(v) or 1))
    return np.vstack(keep)


def visual_search(query: str, k: int = 1, compress: bool = False) -> list[tuple[str, float]]:
    q = emb.token_vectors(query)
    scored = []
    for name, regions in PAGES.items():
        patches = page_patches(regions)
        if compress:
            patches = binarize(pool_patches(patches))
            q_use = binarize(q)
        else:
            q_use = q
        scored.append((name, round(maxsim(q_use, patches), 3)))
    return sorted(scored, key=lambda x: -x[1])[:k]


def colqwen_search(images, queries):
    """Real implementation (pip install colpali-engine torch; GPU recommended)."""
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16,
                                     device_map="auto").eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")
    with torch.no_grad():
        page_emb = model(**processor.process_images(images).to(model.device))
        query_emb = model(**processor.process_queries(queries).to(model.device))
    return processor.score_multi_vector(query_emb, page_emb)    # [n_queries, n_pages]


def answer_from_pages(question: str, png_paths: list[str]) -> str:
    """Send the retrieved page images to Claude (vision) and answer from them."""
    import anthropic

    content = []
    for path in png_paths:
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode()
        content.append({"type": "image", "source": {"type": "base64",
                                                    "media_type": "image/png", "data": data}})
    content.append({"type": "text", "text": f"Answer from these pages only: {question}"})
    resp = anthropic.Anthropic().messages.create(model="claude-opus-5-5", max_tokens=2048,
                                                 messages=[{"role": "user", "content": content}])
    return "".join(b.text for b in resp.content if b.type == "text")


def demo() -> None:
    q = "What is the power draw of the star tracker?"
    show("Visual search (fp)", visual_search(q, k=3))
    show("Visual search (pooled + binary)", visual_search(q, k=3, compress=True))


if __name__ == "__main__":
    demo()
