# dsn-rag

Local-first RAG pipeline over the **DSN cahier technique** (NEODeS, ~392-page
French PDF), exposed to an LLM agent (Claude Code) through an **MCP server**.
The agent retrieves only the relevant chunks per query instead of loading the
whole document. Everything runs locally — no external API calls for embedding
or retrieval.

## Stack

| Concern | Choice |
|---------|--------|
| PDF extraction | `pymupdf` (fitz) — preserves page numbers + layout |
| Embeddings | `BAAI/bge-m3` (multilingual, strong French) via `sentence-transformers`, `device="mps"` |
| Reranker | `BAAI/bge-reranker-v2-m3` cross-encoder, `device="mps"` |
| Lexical recall | BM25 (`rank-bm25`) |
| Vector store | `chromadb`, embedded, persisted to `chroma/` |
| Agent interface | `mcp` SDK — one tool `search_dsn(query, top_k)` |
| Env / deps | `uv` |

Targets Apple Silicon (M-series) via the PyTorch **MPS** backend; falls back to
CPU with a warning if MPS is unavailable.

## Retrieval pipeline

Each query runs four stages, all local:

1. **Dense recall** — bge-m3 nearest `fetch_k` (default 25) chunks.
2. **Lexical recall** — BM25 top `fetch_k`. Closes the dense recall gap on
   keyword-heavy / prose queries (named organisations, exact codes) that the
   bi-encoder never surfaces.
3. **Code-aware exact match** — if the query cites a bloc/rubrique code
   (`S21.G00.30` / `S21.G00.30.009`), those chunks are pulled by metadata.
4. **Cross-encoder rerank** — bge-reranker-v2-m3 reorders the merged candidates
   and returns the top `k`. Falls back to dense order if the reranker can't load.

## Chunking

Structure-aware, tuned to the cahier technique's rigid hierarchy:

- Splits on **rubrique code** boundaries (`S21.G00.30.008`); the **bloc**
  (`S21.G00.30`) is carried from each page's header band as parent context.
- "Valeurs autorisées" enumerations (`01 - libellé`) stay attached to their
  rubrique. Oversized tables split by row-groups, repeating the heading + code
  so each piece is self-describing.
- Header/footer noise and TOC dotted-leaders are stripped; bloc-summary stubs
  are merged into a per-bloc overview chunk.
- Prose / matrix pages with no detectable rubrique fall back to a recursive
  character splitter (split at page boundaries so `page` metadata stays exact).

Every chunk carries metadata: `page` (1-indexed physical PDF page), `bloc`,
`rubrique`, `heading`, `chunk_type` (`text` | `table`).

## Setup

```bash
cd dsn-rag
uv sync                                  # create venv + install deps
mkdir -p data
curl -o data/cahier-technique.pdf \
  https://www.net-entreprises.fr/media/documentation/dsn-cahier-technique-2027.1.pdf
uv run python -m src.ingest              # ~150s on M-series
```

The PDF is **not** tracked in git (see `.gitignore`) — download it from
[net-entreprises](https://www.net-entreprises.fr/media/documentation/dsn-cahier-technique-2027.1.pdf)
to `data/cahier-technique.pdf` before ingesting.

## Usage

### 1. Ingest (run once, or after changing chunking)

```bash
uv run python -m src.ingest
```

Extracts → chunks → embeds (bge-m3 on MPS, batched) → persists to `chroma/`.
Idempotent: drops and recreates the collection each run. Prints page/chunk
counts and embed time (~150s for ~2700 chunks on an M-series GPU).

### 2. Manual retrieval eval (CLI)

```bash
uv run python -m src.query "comment déclarer le code postal d'un individu ?" --k 5
uv run python -m src.query "valeurs autorisées du sexe" --k 5 --fetch-k 40 --full
```

Prints each hit's score, page, bloc/rubrique, heading and a snippet.

### 3. Retrieval eval against gold pages

```bash
uv run python -m src.evaluate --k 5
```

Runs `eval/questions.yaml` (real DSN questions + expected physical PDF pages)
and reports recall@K — whether any expected page appears in the top-K.

### 4. MCP server

```bash
uv run python -m src.server
```

Exposes one tool:

```
search_dsn(query: str, top_k: int = 5)
  -> [{text, page, bloc, rubrique, chunk_type, score}, ...]
```

Responses are token-lean (snippets truncated). `page` is the physical PDF page.

#### Register with Claude Code

`.mcp.json` (project) or `~/.claude.json`:

```json
{
  "mcpServers": {
    "dsn-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/dsn-rag", "python", "-m", "src.server"]
    }
  }
}
```

Or:

```bash
claude mcp add dsn-rag -- uv run --directory /absolute/path/to/dsn-rag python -m src.server
```

> First `search_dsn` call in a fresh process loads two models
> (bge-m3 + reranker, ~4.5GB) and builds the BM25 index — a few seconds warmup,
> then fast. The server pays this once.

## Project layout

```
dsn-rag/
  pyproject.toml
  data/                 # cahier-technique.pdf (gitignored)
  chroma/               # persisted vector store (gitignored)
  src/
    extract.py          # PDF -> per-page text (pymupdf)
    chunk.py            # structure-aware splitting
    embed.py            # shared bge-m3 loader + MPS device pick
    rerank.py           # cross-encoder reranker loader
    lexical.py          # BM25 index over persisted chunks
    ingest.py           # extract -> chunk -> embed -> persist (run once)
    query.py            # hybrid retrieval + CLI
    evaluate.py         # recall@K over eval/questions.yaml
    server.py           # MCP server exposing search_dsn
  eval/questions.yaml   # DSN questions + expected pages
```

## Notes

- **French only** → multilingual embeddings (`bge-m3`); do not swap in an
  English-only model.
- **Page numbers**: the PDF header prints `N / 370`, but the document has 392
  physical pages (~22-page front-matter offset). All metadata + eval use the
  **physical PDF index** (what `pymupdf` returns, 1-indexed).
- Re-run `ingest` whenever `chunk.py` changes — `query`/`evaluate`/`server`
  read the persisted collection, and BM25 is rebuilt from it.
