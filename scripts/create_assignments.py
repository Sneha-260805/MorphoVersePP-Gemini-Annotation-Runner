#!/usr/bin/env python3
"""Split the assignable corpus deterministically among N teammates.

Reads corpus/source_manifest.json (built once, offline, from the
development repo's output_v3 — see corpus/README notes in
corpus_inventory.json's "source_root" field). Excludes:

  - poems in a PROFILE_MISSING language (see BLOCKED_LANGUAGES.md) — this
    script never invents a profile or falls back to a generic one;
  - the six pilot poems (PILOT_ALREADY_GENERATED) — regenerating one of
    those is an explicit, separate action, never a default assignment;
  - (optionally, --exclude-completed) poems that already have a MODEL_CANDIDATE
    file under outputs/model_candidates/.

Makes no network call. Writes one CSV per teammate under assignments/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = REPO_ROOT / "corpus" / "source_manifest.json"
ASSIGNMENTS_DIR = REPO_ROOT / "assignments"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "model_candidates"


def load_assignable_records(*, exclude_completed: bool) -> "list[dict]":
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records = []
    for r in manifest["records"]:
        if r["profile_status"] != "SUPPORTED_PILOT_VALIDATED":
            continue
        if r["pilot_status"] == "PILOT_ALREADY_GENERATED":
            continue
        if exclude_completed:
            candidate_path = OUTPUT_ROOT / r["language"] / f"{r['poem_id']}_vertex_model_candidate.json"
            if candidate_path.exists():
                continue
        records.append(r)
    return records


def split_balanced(records: "list[dict]", teammates: int) -> "list[list[dict]]":
    """Deterministic, language-balanced round-robin split.

    Groups by language (sorted for determinism), then deals each language's
    poems (sorted by poem_id) round-robin across teammate buckets, starting
    from the next bucket after where the previous language left off — this
    keeps any one teammate from being dealt every poem of a single language
    just because that language happens to sort first.
    """
    by_language: "dict[str, list[dict]]" = {}
    for r in records:
        by_language.setdefault(r["language"], []).append(r)

    buckets: "list[list[dict]]" = [[] for _ in range(teammates)]
    cursor = 0
    for language in sorted(by_language):
        poems = sorted(by_language[language], key=lambda r: r["poem_id"])
        for poem in poems:
            buckets[cursor % teammates].append(poem)
            cursor += 1
    return buckets


def write_assignment_csv(path: Path, records: "list[dict]", assignee: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["poem_id", "language", "assignee"])
        for r in sorted(records, key=lambda r: r["poem_id"]):
            writer.writerow([r["poem_id"], r["language"], assignee])


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teammates", type=int, required=True, help="Number of teammates to split the assignable corpus across.")
    parser.add_argument("--exclude-completed", action="store_true", help="Also exclude poems that already have a MODEL_CANDIDATE output file.")
    parser.add_argument("--prefix", type=str, default="teammate", help="Assignee/file name prefix (default: 'teammate').")
    args = parser.parse_args(argv)

    if args.teammates < 1:
        parser.error("--teammates must be >= 1")

    records = load_assignable_records(exclude_completed=args.exclude_completed)
    buckets = split_balanced(records, args.teammates)

    all_ids = [r["poem_id"] for bucket in buckets for r in bucket]
    assert len(all_ids) == len(set(all_ids)), "internal error: duplicate poem_id across assignment buckets"

    for i, bucket in enumerate(buckets, start=1):
        assignee = f"{args.prefix}_{i}"
        out_path = ASSIGNMENTS_DIR / f"{assignee}.csv"
        write_assignment_csv(out_path, bucket, assignee)
        langs = sorted({r["language"] for r in bucket})
        print(f"{out_path.relative_to(REPO_ROOT)}: {len(bucket)} poems across {len(langs)} language(s) {langs}")

    print(f"Total assignable poems: {len(records)}; total written: {len(all_ids)}; teammates: {args.teammates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
