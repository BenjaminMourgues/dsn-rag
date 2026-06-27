"""Shared embedding model loader: BAAI/bge-m3 on Apple Silicon (MPS).

bge-m3 is multilingual with strong French support — required since the cahier
technique is in French. Falls back to CPU with a warning if MPS is unavailable.
"""

from __future__ import annotations

import sys
from functools import lru_cache

EMBED_MODEL = "BAAI/bge-m3"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    print(
        "WARNING: MPS not available, falling back to CPU (embedding will be slow).",
        file=sys.stderr,
    )
    return "cpu"


@lru_cache(maxsize=1)
def load_embedder():
    from sentence_transformers import SentenceTransformer

    device = pick_device()
    print(f"Loading {EMBED_MODEL} on device={device} …")
    return SentenceTransformer(EMBED_MODEL, device=device)
