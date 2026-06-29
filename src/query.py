"""Hybrid retrieval over the persisted DSN chunks.

``search()`` is the shared retrieval entry point used by the MCP server
(``server.py``) and the eval harness (``evaluate.py``); this module also offers
a CLI for manual inspection:

    uv run python -m src.query "comment déclarer le code postal d'un individu ?" --k 5

Retrieval pipeline (all local, MPS):
  1. Dense recall: bge-m3 nearest ``fetch_k`` chunks.
  2. Lexical recall: BM25 top ``fetch_k`` — closes the dense recall gap on
     keyword-heavy / prose queries (named organisations, exact codes) that the
     bi-encoder never surfaces.
  3. Code-aware exact match: if the query cites a bloc/rubrique code
     (S21.G00.30 / S21.G00.30.009), pull those chunks by metadata too.
  4. Cross-encoder rerank (bge-reranker-v2-m3) reorders the merged candidates
     and returns the top ``k``. Falls back to dense order if the reranker can't
     load.

Prints each hit's score, page, bloc/rubrique, heading and a snippet.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import chromadb

from .embed import load_embedder
from .ingest import CHROMA_DIR, COLLECTION

RUBRIQUE_RE = re.compile(r"S\d{2}\.G\d{2}\.\d{2}\.\d{3}")
BLOC_RE = re.compile(r"S\d{2}\.G\d{2}\.\d{2}")

DEFAULT_FETCH_K = 25


@dataclass
class Hit:
    score: float
    page: int
    bloc: str
    rubrique: str
    heading: str
    chunk_type: str
    text: str


def _codes_in(query: str) -> tuple[set[str], set[str]]:
    """Rubrique and bloc codes mentioned in the query."""
    rubriques = set(RUBRIQUE_RE.findall(query))
    # bloc matches also fire inside rubrique codes; keep blocs not covered by a rubrique
    blocs = {b for b in BLOC_RE.findall(query) if not any(b in r for r in rubriques)}
    return rubriques, blocs


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(COLLECTION)


def search(query: str, k: int = 5, fetch_k: int = DEFAULT_FETCH_K) -> list[Hit]:
    collection = _get_collection()
    model = load_embedder()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()

    # 1. dense recall
    res = collection.query(query_embeddings=[vec], n_results=fetch_k)
    candidates: dict[str, tuple[str, dict]] = {}
    for cid, doc, meta in zip(res["ids"][0], res["documents"][0], res["metadatas"][0]):
        candidates[cid] = (doc, meta)

    # 2. lexical (BM25) recall
    from .lexical import search_bm25

    for cid, doc, meta in search_bm25(query, fetch_k):
        candidates.setdefault(cid, (doc, meta))

    # 3. code-aware exact match
    rubriques, blocs = _codes_in(query)
    clauses = [{"rubrique": r} for r in rubriques] + [{"bloc": b} for b in blocs]
    if clauses:
        where = clauses[0] if len(clauses) == 1 else {"$or": clauses}
        matched = collection.get(where=where)
        for cid, doc, meta in zip(matched["ids"], matched["documents"], matched["metadatas"]):
            candidates.setdefault(cid, (doc, meta))

    # 4. rerank (fall back to dense order if unavailable)
    items = list(candidates.values())
    try:
        from .rerank import rerank

        ranked = rerank(query, items, top_k=k)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: reranker unavailable ({e}); using dense order.")
        # preserve dense order: dense hits first (already insertion-ordered)
        dist = {cid: 1.0 - d for cid, d in zip(res["ids"][0], res["distances"][0])}
        ranked = [
            (dist.get(cid, 0.0), doc, meta)
            for cid, (doc, meta) in list(candidates.items())[:k]
        ]

    return [
        Hit(
            score=score,
            page=int(meta.get("page", 0)),
            bloc=meta.get("bloc", ""),
            rubrique=meta.get("rubrique", ""),
            heading=meta.get("heading", ""),
            chunk_type=meta.get("chunk_type", ""),
            text=doc,
        )
        for score, doc, meta in ranked
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual DSN retrieval eval")
    ap.add_argument("query", help="natural-language question")
    ap.add_argument("--k", type=int, default=5, help="number of hits")
    ap.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K, help="dense recall depth before rerank")
    ap.add_argument("--full", action="store_true", help="print full chunk text")
    args = ap.parse_args()

    hits = search(args.query, args.k, args.fetch_k)
    print(f"\nQuery: {args.query!r}  (top {args.k})\n")
    for i, h in enumerate(hits, 1):
        loc = f"page {h.page}"
        if h.bloc:
            loc += f" | {h.rubrique or h.bloc}"
        print(f"[{i}] score={h.score:.3f}  {loc}  ({h.chunk_type})")
        if h.heading:
            print(f"    » {h.heading}")
        snippet = h.text if args.full else _snippet(h.text)
        print("    " + snippet.replace("\n", "\n    "))
        print()


def _snippet(text: str, n: int = 320) -> str:
    return text if len(text) <= n else text[:n].rstrip() + " …"


if __name__ == "__main__":
    main()
