"""PDF -> per-page text extraction for the DSN cahier technique.

Uses PyMuPDF (fitz). Preserves 1-indexed page numbers. Two extraction modes:

- ``extract_pages``: plain text per page (reading order via "text" mode).
- ``extract_page_blocks``: layout-aware block dump, useful for inspecting how
  tables ("valeurs autorisées") survive extraction during the Step 1 spike.

Run as a script to print raw output for a sample of pages:

    uv run python -m src.extract                # default sample
    uv run python -m src.extract 12 13 47 200   # explicit page numbers
    uv run python -m src.extract --blocks 47    # block-layout dump of page 47
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "cahier-technique.pdf"


@dataclass
class Page:
    """One extracted page. ``number`` is the physical 1-indexed PDF page (what
    PyMuPDF returns), NOT the "N / 370" number printed in the page header — the
    two differ by the ~22-page front-matter offset. All metadata uses this."""

    number: int
    text: str


def _open(pdf_path: Path | str = PDF_PATH) -> fitz.Document:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}. Drop the cahier technique there first."
        )
    return fitz.open(pdf_path)


def extract_pages(pdf_path: Path | str = PDF_PATH) -> list[Page]:
    """Return every page as plain text in natural reading order."""
    with _open(pdf_path) as doc:
        return [
            Page(number=i + 1, text=page.get_text("text"))
            for i, page in enumerate(doc)
        ]


def extract_page(number: int, pdf_path: Path | str = PDF_PATH) -> Page:
    """Extract a single 1-indexed page."""
    with _open(pdf_path) as doc:
        if not (1 <= number <= doc.page_count):
            raise IndexError(f"page {number} out of range 1..{doc.page_count}")
        return Page(number=number, text=doc[number - 1].get_text("text"))


def extract_page_blocks(number: int, pdf_path: Path | str = PDF_PATH) -> str:
    """Layout-aware block dump of a single page (for inspecting table structure)."""
    with _open(pdf_path) as doc:
        page = doc[number - 1]
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
        out = []
        for b in sorted(blocks, key=lambda b: (round(b[1]), round(b[0]))):
            x0, y0, x1, y1, btext = b[0], b[1], b[2], b[3], b[4]
            out.append(f"[block @ y={y0:.0f} x={x0:.0f}–{x1:.0f}]\n{btext.rstrip()}")
        return "\n".join(out)


def page_count(pdf_path: Path | str = PDF_PATH) -> int:
    with _open(pdf_path) as doc:
        return doc.page_count


# ---------------------------------------------------------------------------
# Step 1 spike runner
# ---------------------------------------------------------------------------

# Representative sample across the document. Tuned once we see real layout; for
# now spread across front-matter, bloc definitions, and likely table pages.
def _default_sample(total: int) -> list[int]:
    candidates = [1, 5, 10, 20, 40, 60, 90, 120, 160, 200, 250, 300, 340, 370, total]
    return sorted({p for p in candidates if 1 <= p <= total})


def _print_page(p: Page) -> None:
    bar = "=" * 78
    print(f"\n{bar}\nPAGE {p.number}  ({len(p.text)} chars)\n{bar}")
    print(p.text.rstrip() or "[EMPTY]")


def main(argv: list[str]) -> int:
    blocks_mode = "--blocks" in argv
    nums = [a for a in argv if a.isdigit()]

    try:
        total = page_count()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"PDF: {PDF_PATH}\nTotal pages: {total}")

    if blocks_mode:
        targets = [int(n) for n in nums] or [_default_sample(total)[5]]
        for n in targets:
            print(f"\n{'#' * 78}\nBLOCK DUMP PAGE {n}\n{'#' * 78}")
            print(extract_page_blocks(n))
        return 0

    targets = [int(n) for n in nums] if nums else _default_sample(total)
    print(f"Sampling pages: {targets}")
    for n in targets:
        _print_page(extract_page(n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
