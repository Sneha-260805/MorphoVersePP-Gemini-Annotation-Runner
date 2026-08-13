#!/usr/bin/env python3
"""Offline preflight check. Makes zero provider calls.

Reports: Python version, whether required dependencies import, corpus
count/language count/profile coverage, supported/blocked poem counts,
assignment file status (count + duplicate detection across ALL assignment
CSVs found under assignments/), existing valid outputs, missing inputs,
output-path conflicts, and safe (non-secret) Google Cloud config values plus
whether a credential is available — never its value.
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def check_python_version() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("  WARNING: this project targets Python 3.10+.")


def check_dependencies() -> bool:
    required = ["google.genai", "google.auth", "requests", "pytest"]
    ok = True
    for mod in required:
        try:
            importlib.import_module(mod)
            print(f"  OK   {mod}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"  MISSING {mod}: {exc}")
    print(f"Dependencies OK: {ok}")
    return ok


def check_corpus() -> dict:
    inv_path = REPO_ROOT / "corpus" / "corpus_inventory.json"
    if not inv_path.exists():
        print("Corpus inventory: MISSING (corpus/corpus_inventory.json not found)")
        return {}
    inv = json.loads(inv_path.read_text(encoding="utf-8"))
    print(f"Corpus records: {inv['total_poem_count']}")
    print(f"Languages: {inv['total_language_count']}")
    print(f"Duplicate poem IDs: {len(inv['duplicate_poem_ids'])}")
    print(f"Malformed records: {len(inv['malformed_records'])}")
    return inv


def check_release_status() -> dict:
    readiness_path = REPO_ROOT / "FULL_CORPUS_READINESS.json"
    if not readiness_path.exists():
        print("Release readiness: MISSING (FULL_CORPUS_READINESS.json not found)")
        return {}
    r = json.loads(readiness_path.read_text(encoding="utf-8"))
    print(f"Engineering-authorized languages: {r['engineering_valid_languages']}")
    print(f"Blocked languages: {len(r['blocked_languages'])} ({', '.join(r['blocked_languages'])})")
    print(f"Blocked poems: {r['blocked_poems']} ({', '.join(r['blocked_poem_ids'])})")
    print(f"Pilot already generated: {r['pilot_already_generated']}")
    print(f"New team generation targets: {r['new_supported_generations']}")
    print(f"Supported corpus ready: {r['supported_corpus_ready']}")
    print(f"Full corpus ready: {r['full_corpus_ready']}")
    return r


def check_assignments() -> None:
    assignments_dir = REPO_ROOT / "assignments"
    csvs = sorted(p for p in assignments_dir.glob("teammate_*.csv"))
    seen: Counter = Counter()
    total_rows = 0
    import csv as _csv
    for i, csv_path in enumerate(csvs, start=1):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        for row in rows:
            pid = (row.get("poem_id") or "").strip()
            if pid:
                seen[pid] += 1
        print(f"Assignment {i}: {len(rows)}")
        total_rows += len(rows)
    duplicates = {pid: n for pid, n in seen.items() if n > 1}
    print(f"Assignment rows total: {total_rows}")
    print(f"Duplicate assignments across files: {len(duplicates)}")
    if duplicates:
        for pid, n in list(duplicates.items())[:10]:
            print(f"  DUPLICATE {pid} assigned {n} times")


def check_outputs() -> None:
    output_root = REPO_ROOT / "outputs" / "model_candidates"
    existing = list(output_root.glob("**/*_vertex_model_candidate.json")) if output_root.exists() else []
    print(f"Existing valid-shaped outputs on disk: {len(existing)}")

    source_root = REPO_ROOT / "data" / "source_corpus"
    missing_inputs = 0
    conflicts = 0
    if source_root.exists():
        source_ids = {p.stem for p in source_root.glob("**/*.json")}
        output_ids = {p.stem.replace("_vertex_model_candidate", "") for p in existing}
        missing_inputs = len(output_ids - source_ids)
    print(f"Output-path conflicts (candidate with no matching source): {missing_inputs}")


def check_google_cloud_config() -> None:
    from morphoverse_gemini_pipeline.delivery.poem_annotator import gemini_backfill_executor_v1_1 as gx

    summary = gx.gemini_safe_config_summary()
    print(f"Google Cloud project: {summary['project']}")
    print(f"Google Cloud location: {summary['region']}")
    print(f"Vertex model: {summary['model']}")
    print(f"Google authentication available: {'YES' if summary['credential_available'] else 'NO'}")


def main() -> int:
    print("=" * 60)
    print("MorphoVerse++ Gemini Annotation Runner — preflight")
    print("=" * 60)
    check_python_version()
    print()
    deps_ok = check_dependencies()
    print()
    check_corpus()
    print()
    check_release_status()
    print()
    check_assignments()
    print()
    check_outputs()
    print()
    check_google_cloud_config()
    print()
    print("Provider calls during preflight: 0")
    print("=" * 60)
    return 0 if deps_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
