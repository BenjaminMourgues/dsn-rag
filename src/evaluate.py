"""Run eval/questions.yaml through retrieval and report recall@K.

    uv run python -m src.evaluate --k 5

For each question, checks whether any expected page appears in the top-K hits.
Reports per-question hit/miss + overall recall@K.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .query import search

QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description="DSN retrieval eval")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--file", default=str(QUESTIONS))
    args = ap.parse_args()

    questions = yaml.safe_load(Path(args.file).read_text())
    hits_count = 0
    print(f"Eval recall@{args.k}  ({len(questions)} questions)\n")

    for q in questions:
        expected = set(q.get("expected_pages", []))
        results = search(q["question"], k=args.k)
        got_pages = [h.page for h in results]
        hit = bool(expected & set(got_pages))
        hits_count += hit
        mark = "✓" if hit else "✗"
        print(f"{mark} {q['question'][:70]}")
        print(f"    expected={sorted(expected)} got={got_pages}")
        if not hit and results:
            top = results[0]
            print(f"    top hit: page {top.page} {top.rubrique or top.bloc} ({top.score:.3f})")
        print()

    print(f"recall@{args.k}: {hits_count}/{len(questions)} = {hits_count / len(questions):.0%}")


if __name__ == "__main__":
    main()
