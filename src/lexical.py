"""Lexical BM25 retrieval over the persisted chunks.

Dense embeddings blur exact tokens (rubrique codes, named organisations like
"France Travail"), leaving keyword-heavy and prose-table queries unreachable at
any vector-search depth. BM25 closes that recall gap. Fused with dense + the
code-aware exact match, then reranked.

The index is built once from the Chroma collection and cached for the process.
"""

from __future__ import annotations

import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from .ingest import CHROMA_DIR, COLLECTION

# Tokenizer: keep DSN codes (s21.g00.30.009 / s21.g00.30) as single tokens so
# exact code queries match lexically; otherwise split on word boundaries.
_TOK_RE = re.compile(r"s\d{2}\.g\d{2}(?:\.\d{2}){0,2}|[a-zà-ÿ0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOK_RE.findall(text.lower())


@lru_cache(maxsize=1)
def load_bm25():
    """Return (bm25, ids, docs, metas) built from the whole collection."""
    col = chromadb.PersistentClient(path=CHROMA_DIR).get_collection(COLLECTION)
    data = col.get()  # all docs + metadatas
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    bm25 = BM25Okapi([tokenize(d) for d in docs])
    return bm25, ids, docs, metas


def search_bm25(query: str, n: int) -> list[tuple[str, str, dict]]:
    """Top-n (id, doc, meta) by BM25 score for the query."""
    bm25, ids, docs, metas = load_bm25()
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    return [(ids[i], docs[i], metas[i]) for i in top if scores[i] > 0]
