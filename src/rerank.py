"""Cross-encoder reranker: BAAI/bge-reranker-v2-m3 on Apple Silicon (MPS).

Same multilingual family as the bge-m3 bi-encoder. Reads query + passage
together and scores relevance directly, fixing the ordering noise that a
bi-encoder leaves in the top-K. Falls back to CPU with a warning if MPS is
unavailable; if the model can't load at all, callers should degrade to dense
order rather than crash.
"""

from __future__ import annotations

from functools import lru_cache

from .embed import pick_device

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def load_reranker():
    from sentence_transformers import CrossEncoder

    device = pick_device()
    print(f"Loading {RERANK_MODEL} on device={device} …")
    return CrossEncoder(RERANK_MODEL, device=device)


def rerank(query: str, candidates: list[tuple[str, dict]], top_k: int) -> list[tuple[float, str, dict]]:
    """Score (text, meta) candidates against query; return top_k as (score, text, meta).

    Scores are the cross-encoder's relevance logits (higher = more relevant).
    """
    if not candidates:
        return []
    model = load_reranker()
    pairs = [[query, text] for text, _ in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        ((float(s), text, meta) for s, (text, meta) in zip(scores, candidates)),
        key=lambda r: r[0],
        reverse=True,
    )
    return ranked[:top_k]
