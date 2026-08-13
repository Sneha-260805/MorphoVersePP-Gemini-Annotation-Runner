"""Stage 5M.1 — corrects Stage 5M's mistake of treating output_v3 (166
records) as the full corpus. The authoritative source is now
data/raw/Indian_poem_dataset.xlsx (1,570 canonical poems). This file covers
the 20 required checks from the Stage 5M.1 task spec, in order.

No test in this file makes a network call, opens a Vertex/Gemini client, or
re-parses the Excel workbook (that happens once, offline, in
scripts/build_corpus_from_excel.py — these tests check its *output*).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from tests.conftest import REPO_ROOT, PROFILE_DIR

RAW_XLSX = REPO_ROOT / "data" / "raw" / "Indian_poem_dataset.xlsx"
EXPECTED_RAW_SHA256 = "d0dbc5c8c387d5d9bbc2c566f5dd4f9a5a665c7c72e96df5a9cf65c6314e0cd2"

PILOT_MAPPINGS = {
    "MV++_0011": ("Bengali", "O Amar Desher Mati"),
    "MV++_0073": ("Hindi", "Ek Chadar Maili Si"),
    "MV++_1118": ("Kannada", "Udayavagali Namma Cheluva Kannada Naadu"),
    "MV++_1153": ("Kashmiri", "Ghazal"),
    "MV++_1249": ("Sindhi", "Sufi Kafis"),
    "MV++_1443": ("Telugu", "Oh Jabilamma"),
}


def _inventory() -> dict:
    return json.loads((REPO_ROOT / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8"))


def _manifest() -> dict:
    return json.loads((REPO_ROOT / "corpus" / "source_manifest.json").read_text(encoding="utf-8"))


def _coverage() -> dict:
    return json.loads((REPO_ROOT / "corpus" / "language_profile_coverage.json").read_text(encoding="utf-8"))


# 1. authoritative Excel is used
def test_raw_excel_is_present_and_hash_matches():
    assert RAW_XLSX.exists()
    actual = hashlib.sha256(RAW_XLSX.read_bytes()).hexdigest()
    assert actual == EXPECTED_RAW_SHA256


# 2. output_v3 is not treated as corpus authority
def test_output_v3_not_present_and_not_referenced_as_authority():
    assert not (REPO_ROOT / "morphoverse_gemini_pipeline" / "delivery" / "output_v3").exists()
    manifest = _manifest()
    assert "output_v3" not in manifest["generated_from"]
    for path in (REPO_ROOT / "corpus" / "corpus_inventory.json",):
        text = path.read_text(encoding="utf-8")
        assert "output_v3" not in text


# 3. exact canonical corpus count = 1570
def test_canonical_corpus_count_is_1570():
    inv = _inventory()
    assert inv["total_poem_count"] == 1570
    manifest = _manifest()
    assert manifest["total_records"] == 1570
    assert len(list((REPO_ROOT / "data" / "source_corpus").glob("**/*.json"))) == 1570


# 4. exact language count = 21
def test_language_count_is_21():
    inv = _inventory()
    assert inv["total_language_count"] == 21
    assert len(inv["poems_per_language"]) == 21


# 5. embedded header is excluded
def test_embedded_header_excluded_from_every_poem_language_value():
    inv = _inventory()
    assert "Original Language" not in inv["poems_per_language"]
    for f in (REPO_ROOT / "data" / "source_corpus").glob("**/*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["language"] != "Original Language"


# 6. four trailing unlabeled records are excluded and audited
def test_trailing_unlabeled_records_audited_and_excluded():
    audit = json.loads((REPO_ROOT / "corpus" / "trailing_unlabeled_records.json").read_text(encoding="utf-8"))
    assert audit["total_trailing_records"] == 4
    for rec in audit["records"]:
        assert rec["classification"] == "TRAILING_UNLABELED_EXTRA_RECORD"
        assert "poem_id" not in rec  # never assigned an MV++ ID
    assert (REPO_ROOT / "corpus" / "TRAILING_RECORD_AUDIT.md").exists()
    # Titles alone can coincidentally repeat elsewhere in the corpus (110
    # duplicate (language, title) pairs exist) so use MV++ poem_id presence,
    # which is definitive: none of the trailing rows were ever assigned one,
    # so no source_corpus file's poem_id can trace back to a trailing row.
    manifest = _manifest()
    assert len(manifest["records"]) == 1570  # trailing rows never inflate the manifest


# 7. IDs are MV++_0001 through MV++_1570
def test_ids_span_0001_through_1570():
    manifest = _manifest()
    ids = {r["poem_id"] for r in manifest["records"]}
    expected = {f"MV++_{n:04d}" for n in range(1, 1571)}
    assert ids == expected


# 8. IDs are unique
def test_ids_unique():
    inv = _inventory()
    assert inv["duplicate_poem_ids"] == []
    assert inv["all_poem_ids_unique"] is True


# 9. all six pilot mappings preserved exactly
@pytest.mark.parametrize("poem_id,expected", PILOT_MAPPINGS.items())
def test_pilot_mapping_preserved(poem_id, expected):
    language, title = expected
    path = REPO_ROOT / "data" / "source_corpus" / language / f"{poem_id}.json"
    assert path.exists(), f"{poem_id} not found under language {language!r}"
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["poem_id"] == poem_id
    assert d["language"] == language
    assert d["poem_title"] == title


def test_all_six_pilots_are_a_disjoint_set_of_six():
    assert runner.PILOT_POEM_IDS == frozenset(PILOT_MAPPINGS.keys())
    assert len(runner.PILOT_POEM_IDS) == 6


# 10 & 11. source-only exports contain exactly five fields; no old-annotation leakage
def test_every_source_record_has_exactly_five_generation_fields():
    for f in (REPO_ROOT / "data" / "source_corpus").glob("**/*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert set(d.keys()) == set(runner.SOURCE_REQUIRED_KEYS)
        for forbidden in ("poet", "translator", "annotation", "cultural_entities", "metaphor_spans",
                           "theme", "emotional_arc", "review_items", "status", "candidate_status"):
            assert forbidden not in d


# 12. six language profiles remain byte-identical
EXPECTED_PROFILE_HASHES = {
    "bengali.json": "f8c8e849566db4e1c3ad83f9e60b84f8cd66047313de2d27149d7de6efaf31e0"[:64],
    "hindi.json": "cd72de93df1253f5402d062214c303c7f26978564ceb7fc07d0c4c8b816ff414"[:64],
    "kannada.json": "f5a4fd04c3bab33f0f836fb19c1727bc2ed131ca0a275a834dc60963dab0250c"[:64],
    "kashmiri.json": "55ae735a57dda4d36926af81fa78b97ce37c754defcbef8a70c2318b028f33df"[:64],
    "sindhi.json": "cc1b244f4299b98021af93bcee84c317786a4335993ebc82e08b8fa50117bea4"[:64],
    "telugu.json": "3bfcf6a8e0fd8c3efac8cb2fffebd3a69b818e75cd25810540b16530c07a4ce4"[:64],
}


@pytest.mark.parametrize("filename,expected_hash", EXPECTED_PROFILE_HASHES.items())
def test_language_profile_byte_identical(filename, expected_hash):
    path = PROFILE_DIR / filename
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected_hash


# 13 & 14. supported/blocked counts recomputed correctly (not hardcoded)
def test_supported_and_blocked_counts_recomputed_from_workbook():
    cov = _coverage()
    inv = _inventory()
    release = json.loads((REPO_ROOT / "corpus" / "execution_release_manifest.json").read_text(encoding="utf-8"))
    # recompute independently from the release manifest + per-language
    # inventory, rather than trusting language_profile_coverage.json's own
    # arithmetic. Authorization (not profile-file presence — all 21 profile
    # files exist on disk as of Stage 5M.2) is the only thing that decides
    # "supported" here.
    authorized = {lang for lang, v in release["languages"].items() if v["status"] == "AUTHORIZED_FOR_TEAM_GENERATION"}
    recomputed_supported = sum(c for lang, c in inv["poems_per_language"].items() if lang in authorized)
    recomputed_blocked = sum(c for lang, c in inv["poems_per_language"].items() if lang not in authorized)
    assert cov["supported_poem_count"] == recomputed_supported
    assert cov["blocked_poem_count"] == recomputed_blocked
    assert recomputed_supported + recomputed_blocked == 1570
    # Stage 5M.2: 20/21 languages authorized, Sanskrit's single poem blocked
    assert recomputed_supported == 1569
    assert recomputed_blocked == 1


# 15. six pilots excluded from default generation
def test_pilots_excluded_from_default_targets():
    manifest = _manifest()
    rows = runner.resolve_targets(
        repo_root=REPO_ROOT, assignment_csv=None, language=None, poem_id=None,
        max_poems=None, allow_pilot_regeneration=False, source_manifest=manifest,
    )
    ids = {r.poem_id for r in rows}
    assert ids.isdisjoint(runner.PILOT_POEM_IDS)
    assert len(rows) == 1570 - 6


# 16. assignments contain no duplicate IDs
def test_assignment_files_have_no_duplicate_ids_across_files():
    seen = set()
    assignment_dir = REPO_ROOT / "assignments"
    csvs = [p for p in assignment_dir.glob("*.csv") if p.name != "example_assignment.csv"]
    assert csvs, "expected regenerated teammate assignment CSVs to exist"
    for csv_path in csvs:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pid = row["poem_id"]
                assert pid not in seen, f"duplicate poem_id {pid} across assignment files"
                seen.add(pid)


# 17. blocked languages do not enter assignments
def test_blocked_languages_excluded_from_assignments():
    release = json.loads((REPO_ROOT / "corpus" / "execution_release_manifest.json").read_text(encoding="utf-8"))
    blocked_langs = {lang for lang, v in release["languages"].items() if v["status"] != "AUTHORIZED_FOR_TEAM_GENERATION"}
    assert blocked_langs == {"Sanskrit"}
    assignment_dir = REPO_ROOT / "assignments"
    for csv_path in [p for p in assignment_dir.glob("*.csv") if p.name != "example_assignment.csv"]:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                assert row["language"] not in blocked_langs
                assert row["poem_id"] != "MV++_1235"


# 18. dry-run makes zero provider calls (full corpus)
def test_full_corpus_dry_run_makes_zero_provider_calls(capsys):
    def exploding_factory():
        raise AssertionError("client_factory must never be called in --dry-run.")

    exit_code = runner.main(["--dry-run"], client_factory=exploding_factory)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.count("[DRY RUN]") == 1570 - 6  # pilots excluded by default


# 19. runner has no 166-record assumption
def test_runner_source_has_no_166_or_output_v3_assumption():
    import inspect
    src = inspect.getsource(runner)
    assert "166" not in src
    assert "output_v3" not in src


# 20. FULL_CORPUS_READY remains false while Sanskrit is blocked
def test_full_corpus_readiness_is_false():
    readiness = json.loads((REPO_ROOT / "FULL_CORPUS_READINESS.json").read_text(encoding="utf-8"))
    assert readiness["ready_for_full_corpus"] is False
    assert readiness["full_corpus_ready"] is False
    assert readiness["supported_corpus_ready"] is True
    assert readiness["total_poems"] == 1570
    assert readiness["pilot_already_generated"] == 6
    assert readiness["new_supported_generations"] == 1563
    assert readiness["blocked_poems"] == 1
    assert readiness["blocked_poem_ids"] == ["MV++_1235"]
    assert readiness["engineering_valid_languages"] == 20
    assert readiness["consistency_check"]["equals_total_poems"] is True
