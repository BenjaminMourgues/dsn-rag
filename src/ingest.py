"""Ingest: extract -> chunk -> embed (bge-m3 on MPS) -> persist to chroma.

Idempotent: drops and recreates the collection each run.

    uv run python -m src.ingest

Embeddings: BAAI/bge-m3 via sentence-transformers on the Apple Silicon GPU
(``device="mps"``), falling back to CPU with a warning if MPS is unavailable.
"""

from __future__ import annotations

import time

import chromadb

from .chunk import chunk_document, est_tokens
from .embed import EMBED_MODEL, load_embedder

COLLECTION = "dsn_cahier_technique"
CHROMA_DIR = "chroma"
BATCH_SIZE = 64


def ingest(pdf_path=None) -> None:
    t0 = time.perf_counter()
    print("Chunking document …")
    chunks = chunk_document(pdf_path)
    pages = len({c.page for c in chunks})
    toks = [est_tokens(c.text) for c in chunks]
    print(
        f"  {len(chunks)} chunks across {pages} pages "
        f"(tokens: med={sorted(toks)[len(toks) // 2]} max={max(toks)})"
    )

    model = load_embedder()

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} …")
    t_embed = time.perf_counter()
    embeddings = model.encode(
        [c.text for c in chunks],
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embed_secs = time.perf_counter() - t_embed
    print(f"  embedded in {embed_secs:.1f}s ({len(chunks) / embed_secs:.1f} chunks/s)")

    print(f"Persisting to {CHROMA_DIR}/ (collection '{COLLECTION}') …")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL}
    )

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=[embeddings[i + j].tolist() for j in range(len(batch))],
            metadatas=[
                {
                    "page": c.page,
                    "bloc": c.bloc or "",
                    "rubrique": c.rubrique or "",
                    "heading": c.heading,
                    "chunk_type": c.chunk_type,
                }
                for c in batch
            ],
        )

    total = time.perf_counter() - t0
    print(
        f"Done. {collection.count()} vectors persisted. "
        f"pages={pages} chunks={len(chunks)} embed={embed_secs:.1f}s total={total:.1f}s"
    )


if __name__ == "__main__":
    ingest()
