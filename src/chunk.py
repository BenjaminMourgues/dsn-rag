"""Structure-aware chunking of the DSN cahier technique.

Tuned to the layout observed in the Step 1 extraction spike:

- Rubrique codes (``S21.G00.30.008``) survive extraction and mark the natural
  boundary between value definitions. We split on them.
- The bloc code (``S21.G00.30``) appears in each page's header band; we use it
  as the parent context for every chunk.
- "Valeurs autorisées" enumerations extract as clean ``NN - libellé`` lines
  directly under the rubrique. We keep them attached to their rubrique; when a
  table is too large we split it into row-groups, repeating the heading +
  rubrique code so each piece stays self-describing.
- Header/footer noise (date, ``N / 370``, running section title) and table-of-
  contents dotted leaders are stripped.
- Prose / matrix pages with no detectable rubrique fall back to a recursive
  character splitter so ingest never crashes.

The document is processed as ONE line stream (not page-by-page) so a rubrique
that spans a page break stays a single chunk. Each chunk's ``page`` is the
physical 1-indexed PDF page where its rubrique code (or first line) appears.

Every chunk carries metadata: page, bloc, rubrique (if applicable), heading,
chunk_type ("text" | "table").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import extract_pages

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAX_TOKENS = 800  # hard ceiling before we split
OVERLAP_RATIO = 0.15  # overlap for the recursive fallback splitter
PROSE_TARGET_TOKENS = 700  # preferred size when packing contiguous prose lines

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

RUBRIQUE_RE = re.compile(r"\bS\d{2}\.G\d{2}\.\d{2}\.\d{3}\b")
BLOC_RE = re.compile(r"\bS\d{2}\.G\d{2}\.\d{2}\b")
RUBRIQUE_ONLY_RE = re.compile(r"^\s*S\d{2}\.G\d{2}\.\d{2}\.\d{3}\s*$")
# Enumerated "valeur autorisée": "01 - Salaire réel", "CDD - ...", "99 - ..."
VALEUR_RE = re.compile(r"^[0-9A-Z]{1,4}\s*-\s+\S")
# Object path following a code, e.g. "Individu.CodePostal"
PATH_RE = re.compile(r"^[A-ZÀ-Ÿ][\w’'À-ÿ]*\.[A-Za-zÀ-ÿ][\w’'.À-ÿ]*\s*$")
# Control lines: CCH-11, CRE-11, CSL 00 ...
CONTROL_RE = re.compile(r"^(CCH|CRE|CSL|CSU|BR)[ -]?\d*\b")
# Length / cardinality constraint: [1,50] [5,5] [2,2] [1,*]
LENGTH_RE = re.compile(r"^\[\d+,\s*\d*\*?\]\s*$")
# Reference to an external value table: "Table HEX - Code postal"
TABLE_REF_RE = re.compile(r"^Table\b.*-")

# Header/footer noise.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s*$")
_PAGENUM_RE = re.compile(r"^\d+\s*/\s*\d+\s*$")
_RUNNING_TITLE_RE = re.compile(r"^\d{1,2}\s+[A-ZÀ-Ÿ].{0,60}$")
# Table-of-contents dotted leader: "S21.G00.06 ............ 142" — navigational
# noise, and the dot runs wildly inflate token estimates.
_DOTLEADER_RE = re.compile(r"\.{5,}")

# A line that cannot be part of a rubrique's label (used when walking up).
def _is_non_label(stripped: str) -> bool:
    return (
        not stripped
        or stripped == "X"
        or CONTROL_RE.match(stripped) is not None
        or LENGTH_RE.match(stripped) is not None
        or PATH_RE.match(stripped) is not None
        or VALEUR_RE.match(stripped) is not None
        or TABLE_REF_RE.match(stripped) is not None
        or RUBRIQUE_ONLY_RE.match(stripped) is not None
    )


@dataclass
class Chunk:
    id: str
    text: str
    page: int
    bloc: str | None
    rubrique: str | None
    heading: str
    chunk_type: str  # "text" | "table"


# ---------------------------------------------------------------------------
# Token estimate (heuristic; avoids loading a tokenizer at chunk time).
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def est_tokens(text: str) -> int:
    return int(len(_TOKEN_RE.findall(text)) * 1.3) + 1


# ---------------------------------------------------------------------------
# Line stream: (text, page, bloc) for every meaningful line in the document.
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    text: str  # rstripped, original indentation kept
    page: int
    bloc: str | None


def _page_bloc(raw_lines: list[str], carried: str | None) -> str | None:
    """Bloc context = first bloc code in the page header band, else carried."""
    for line in raw_lines[:8]:
        m = BLOC_RE.search(line)
        if m and not RUBRIQUE_RE.search(line):
            return m.group(0)
    return carried


def build_line_stream(pdf_path=None) -> list[_Line]:
    pages = extract_pages(pdf_path) if pdf_path else extract_pages()
    stream: list[_Line] = []
    carried: str | None = None
    for page in pages:
        raw = page.text.splitlines()
        bloc = _page_bloc(raw, carried)
        carried = bloc
        for r in raw:
            line = r.rstrip()
            s = line.strip()
            if not s:
                continue
            if _DATE_RE.match(s) or _PAGENUM_RE.match(s):
                continue
            if _RUNNING_TITLE_RE.match(s) and not s.endswith("."):
                continue
            if _DOTLEADER_RE.search(s):
                continue
            stream.append(_Line(text=line, page=page.number, bloc=bloc))
    return stream


# ---------------------------------------------------------------------------
# Segmentation: split the stream into rubrique segments + prose segments.
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    rubrique: str | None
    heading: str
    lines: list[_Line]

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines).strip()

    @property
    def page(self) -> int:
        return self.lines[0].page

    @property
    def bloc(self) -> str | None:
        return self.lines[0].bloc

    @property
    def n_value_lines(self) -> int:
        return sum(1 for ln in self.lines if VALEUR_RE.match(ln.text.strip()))


def _label_start(stream: list[_Line], code_idx: int) -> tuple[int, str]:
    """Index where the rubrique's label begins, and the label text.

    The label is the contiguous run of up-to-3 'label-eligible' lines directly
    above the code line. Stops at controls, paths, value lines, table refs."""
    label: list[str] = []
    start = code_idx
    j = code_idx - 1
    while j >= 0 and len(label) < 3:
        s = stream[j].text.strip()
        if _is_non_label(s):
            break
        label.insert(0, s)
        start = j
        j -= 1
    return start, " ".join(label).strip()


def segment_stream(stream: list[_Line]) -> list[_Segment]:
    code_positions = [
        i for i, ln in enumerate(stream) if RUBRIQUE_ONLY_RE.match(ln.text.strip())
    ]
    if not code_positions:
        return [_Segment(None, "", stream)] if stream else []

    # Precompute each rubrique's label-start so segments don't overlap.
    starts = []
    for ci in code_positions:
        s_idx, heading = _label_start(stream, ci)
        starts.append((s_idx, ci, heading))

    segments: list[_Segment] = []

    # Leading prose before the first rubrique label.
    first_start = starts[0][0]
    if first_start > 0:
        segments.append(_Segment(None, "", stream[:first_start]))

    for k, (s_idx, ci, heading) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(stream)
        rubrique = stream[ci].text.strip()
        segments.append(_Segment(rubrique, heading, stream[s_idx:end]))
    return segments


# ---------------------------------------------------------------------------
# Recursive character fallback splitter (prose / matrix pages).
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _hard_window(s: str, max_tokens: int) -> list[str]:
    """Last-resort split of a separator-less blob into fixed word windows."""
    words = s.split()
    per = max(1, int(max_tokens / 1.3))  # invert est_tokens' ~1.3 factor
    return [" ".join(words[i : i + per]) for i in range(0, len(words), per)] or [s]


def _recursive_split(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    if est_tokens(text) <= max_tokens:
        return [text]

    # Split to a body budget that leaves room for the prepended overlap, so the
    # final pieces still respect max_tokens.
    body_max = max_tokens - int(max_tokens * OVERLAP_RATIO)

    def _split(s: str, seps: list[str]) -> list[str]:
        if est_tokens(s) <= body_max:
            return [s]
        if not seps:
            return _hard_window(s, body_max)
        sep, rest = seps[0], seps[1:]
        parts = s.split(sep)
        if len(parts) == 1:
            return _split(s, rest)
        pieces, buf = [], ""
        for part in parts:
            cand = (buf + sep + part) if buf else part
            if est_tokens(cand) <= body_max:
                buf = cand
            else:
                if buf:
                    pieces.append(buf)
                if est_tokens(part) > body_max:
                    pieces.extend(_split(part, rest))
                    buf = ""
                else:
                    buf = part
        if buf:
            pieces.append(buf)
        return pieces

    pieces = _split(text, _SEPARATORS)
    ov_target = int(max_tokens * OVERLAP_RATIO)
    overlapped: list[str] = []
    for i, p in enumerate(pieces):
        if i == 0:
            overlapped.append(p)
        else:
            overlapped.append(f"{_tail_tokens(pieces[i - 1], ov_target)} {p}".strip())
    return overlapped


def _tail_tokens(text: str, target_tokens: int) -> str:
    """Trailing words of ``text`` totalling ~target_tokens (est_tokens budget)."""
    words = text.split()
    tail: list[str] = []
    for w in reversed(words):
        tail.insert(0, w)
        if est_tokens(" ".join(tail)) >= target_tokens:
            break
    return " ".join(tail)


def _pack_prose(seg: _Segment, idx: int) -> list[Chunk]:
    """Pack a prose segment into ~PROSE_TARGET_TOKENS chunks, split at page
    boundaries so each chunk's ``page`` metadata is exact (a prose run can span
    many pages — e.g. the front-matter before the first rubrique)."""
    from itertools import groupby

    out: list[Chunk] = []
    sub = 0
    for page, grp in groupby(seg.lines, key=lambda ln: ln.page):
        lines = list(grp)
        bloc = lines[0].bloc
        text = "\n".join(ln.text for ln in lines).strip()
        if not text:
            continue
        if est_tokens(text) <= MAX_TOKENS:
            out.append(Chunk(f"p{page}-txt-{idx}-{sub}", text, page, bloc, None, "", "text"))
        else:
            for i, piece in enumerate(_recursive_split(text, PROSE_TARGET_TOKENS)):
                out.append(Chunk(f"p{page}-txt-{idx}-{sub}-r{i}", piece, page, bloc, None, "", "text"))
        sub += 1
    return out


# Largest non-value prefix we will repeat verbatim across row-groups. Above
# this, the "header" isn't a real table header (it's a big messy enumerated
# domain like S21.G00.81 code cotisation) so we recursive-split instead.
_MAX_REPEATED_PREFIX = MAX_TOKENS // 3


def _oversized_split(seg: _Segment, idx: int, ctype: str) -> list[Chunk]:
    """Recursive char-split an oversized segment, repeating a short
    heading + rubrique-code prefix so each piece stays self-describing."""
    prefix = f"{seg.heading} {seg.rubrique or ''}".strip()
    base_id = f"p{seg.page}-{seg.rubrique or 'txt'}-{idx}"
    out = []
    for i, piece in enumerate(_recursive_split(seg.text)):
        body = piece if (seg.rubrique and seg.rubrique in piece[:160]) else f"{prefix}\n{piece}".strip()
        out.append(Chunk(f"{base_id}-r{i}", body, seg.page, seg.bloc, seg.rubrique, seg.heading, ctype))
    return out


def _split_value_table(seg: _Segment, idx: int) -> list[Chunk]:
    """Split an oversized valeurs-autorisées table by row-groups, repeating the
    heading + rubrique code prefix so each piece stays self-describing."""
    header_lines = [ln.text for ln in seg.lines if not VALEUR_RE.match(ln.text.strip())]
    value_lines = [ln.text for ln in seg.lines if VALEUR_RE.match(ln.text.strip())]
    prefix = "\n".join(header_lines).strip()
    # If the prefix is too large to repeat, this isn't a clean code|label table;
    # fall back to recursive splitting so we never emit a 16k-token chunk.
    if est_tokens(prefix) > _MAX_REPEATED_PREFIX or not value_lines:
        return _oversized_split(seg, idx, "table")
    budget = MAX_TOKENS - est_tokens(prefix)
    chunks, group, gi, cur = [], [], 0, 0
    for vl in value_lines:
        t = est_tokens(vl)
        if cur + t > budget and group:
            chunks.append(_table_chunk(seg, idx, gi, prefix, group))
            gi, group, cur = gi + 1, [], 0
        group.append(vl)
        cur += t
    if group:
        chunks.append(_table_chunk(seg, idx, gi, prefix, group))
    return chunks


def _table_chunk(seg, idx, gi, prefix, group) -> Chunk:
    body = "\n".join(group)
    return Chunk(
        f"p{seg.page}-{seg.rubrique}-{idx}-t{gi}",
        f"{prefix}\n{body}".strip(),
        seg.page, seg.bloc, seg.rubrique, seg.heading, "table",
    )


def _segment_to_chunks(seg: _Segment, idx: int) -> list[Chunk]:
    if not seg.text:
        return []
    if seg.rubrique is None:
        return _pack_prose(seg, idx)

    ctype = "table" if seg.n_value_lines >= 2 else "text"
    base_id = f"p{seg.page}-{seg.rubrique}-{idx}"
    if est_tokens(seg.text) <= MAX_TOKENS:
        return [Chunk(base_id, seg.text, seg.page, seg.bloc, seg.rubrique, seg.heading, ctype)]
    if ctype == "table":
        return _split_value_table(seg, idx)
    return _oversized_split(seg, idx, "text")


def _is_stub(seg: _Segment) -> bool:
    """A bloc-summary entry: rubrique label + code only, with no object path,
    no length constraint and no values (the full definition appears later)."""
    if seg.rubrique is None:
        return False
    has_path = any(PATH_RE.match(ln.text.strip()) for ln in seg.lines)
    has_len = any(LENGTH_RE.match(ln.text.strip()) for ln in seg.lines)
    return not has_path and not has_len and seg.n_value_lines == 0


def _merge_stub_run(run: list[_Segment], idx: int) -> Chunk:
    """Collapse consecutive summary-stub rubriques into one bloc-overview chunk."""
    bloc = run[0].bloc
    page = run[0].page
    text = "\n".join(seg.text for seg in run).strip()
    rubriques = ", ".join(seg.rubrique for seg in run if seg.rubrique)
    heading = f"{bloc} – liste des rubriques" if bloc else "liste des rubriques"
    body = f"{heading}\nRubriques : {rubriques}\n{text}"
    return Chunk(f"p{page}-{bloc or 'txt'}-overview-{idx}", body, page, bloc, None, heading, "text")


def chunk_document(pdf_path=None) -> list[Chunk]:
    stream = build_line_stream(pdf_path)
    segments = segment_stream(stream)
    chunks: list[Chunk] = []
    idx = 0
    stub_run: list[_Segment] = []

    def flush_stubs():
        nonlocal idx
        if stub_run:
            chunks.append(_merge_stub_run(stub_run, idx))
            idx += 1
            stub_run.clear()

    for seg in segments:
        if _is_stub(seg):
            stub_run.append(seg)
            continue
        flush_stubs()
        chunks.extend(_segment_to_chunks(seg, idx))
        idx += 1
    flush_stubs()
    return chunks


# ---------------------------------------------------------------------------
# CLI: inspect chunking quality
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    chunks = chunk_document()
    nums = {int(a) for a in sys.argv[1:] if a.isdigit()}

    if not nums:
        # summary stats
        from collections import Counter

        types = Counter(c.chunk_type for c in chunks)
        with_rub = sum(1 for c in chunks if c.rubrique)
        toks = [est_tokens(c.text) for c in chunks]
        print(f"total chunks : {len(chunks)}")
        print(f"  text/table : {types}")
        print(f"  w/ rubrique: {with_rub}  ({100*with_rub//len(chunks)}%)")
        print(f"  tokens     : min={min(toks)} med={sorted(toks)[len(toks)//2]} max={max(toks)} mean={sum(toks)//len(toks)}")
        big = sum(1 for t in toks if t > MAX_TOKENS)
        print(f"  > {MAX_TOKENS}tok : {big}")
        raise SystemExit(0)

    for c in chunks:
        if c.page not in nums:
            continue
        print(
            f"\n--- {c.id} | page={c.page} bloc={c.bloc} rubrique={c.rubrique} "
            f"type={c.chunk_type} ~{est_tokens(c.text)}tok\n    heading={c.heading!r}"
        )
        body = c.text if len(c.text) <= 800 else c.text[:800] + " …[trunc]"
        print(body)
