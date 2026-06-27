"""MCP server exposing one tool: search_dsn(query, top_k).

Returns a token-lean list of {text, page, bloc, rubrique, score} so an agent
retrieves only the relevant chunks of the DSN cahier technique per query.

Run:  uv run python -m src.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .query import search

mcp = FastMCP("dsn-rag")

# Keep responses small: cap snippet length so the agent context stays lean.
SNIPPET_CHARS = 700


@mcp.tool()
def search_dsn(query: str, top_k: int = 5) -> list[dict]:
    """Recherche sémantique dans le cahier technique DSN (NEODeS).

    Renvoie les passages les plus pertinents (blocs/rubriques S__.G00.__.___,
    valeurs autorisées, règles déclaratives). Utiliser pour toute question sur
    la structure ou les règles DSN plutôt que de charger le document entier.

    Args:
        query: question en langage naturel (français de préférence).
        top_k: nombre de passages à renvoyer (défaut 5).

    Returns:
        Liste de {text, page, bloc, rubrique, chunk_type, score}, triée par
        score de pertinence décroissant. ``page`` est l'index physique du PDF.
    """
    hits = search(query, k=top_k)
    out = []
    for h in hits:
        text = h.text if len(h.text) <= SNIPPET_CHARS else h.text[:SNIPPET_CHARS].rstrip() + " …"
        out.append(
            {
                "text": text,
                "page": h.page,
                "bloc": h.bloc,
                "rubrique": h.rubrique,
                "chunk_type": h.chunk_type,
                "score": round(h.score, 3),
            }
        )
    return out


if __name__ == "__main__":
    mcp.run()
