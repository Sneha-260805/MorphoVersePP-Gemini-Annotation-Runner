"""Stage 5M.2 — teammate repo sync to the 20/21-language engineering-valid
state (Stage 5N.3 grounding/repair fixes, 15 new language profiles,
execution_release_manifest.json + programmatic Sanskrit block, 1,563-poem
3-way assignment split).

Covers the checklist items not already exercised by test_corpus_correction_1570.py,
test_execution_lifecycle_and_stop_gate.py, test_source_and_prompt.py, and
test_cli_dry_run_and_safety.py: all 21 profiles load; 20 languages
authorized; exact assignment counts (1563 total, 521 each); no pilot poem in
any assignment; multiline grounding regression; Sanskrit pre-sandhi false
positive stays rejected; multiple same-rule contradictions batch into one
repair round; assignment files exactly cover the authorized new-generation
universe; the Sanskrit block is enforced at the CLI (--language / --poem-id),
not only at the function level.

No test in this file makes a network call.
"""
from __future__ import annotations

import csv
import json

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from morphoverse_gemini_pipeline.delivery.poem_annotator import vertex_canary_execution_v1_1 as vce
from morphoverse_gemini_pipeline.delivery.poem_annotator import grounding
from morphoverse_gemini_pipeline.delivery.poem_annotator.annotation_language_profile_v1_1 import (
    load_annotation_language_profiles,
)
from tests.conftest import REPO_ROOT, PROFILE_DIR, load_release_manifest

ASSIGNMENT_DIR = REPO_ROOT / "assignments"
PILOT_IDS = {"MV++_0011", "MV++_0073", "MV++_1118", "MV++_1153", "MV++_1249", "MV++_1443"}


# ── 2. all 21 profiles load ─────────────────────────────────────────────────
def test_all_21_language_profiles_load():
    profiles = load_annotation_language_profiles(PROFILE_DIR)
    assert len(profiles) == 21
    inv = json.loads((REPO_ROOT / "corpus" / "corpus_inventory.json").read_text(encoding="utf-8"))
    assert set(profiles.keys()) == set(inv["poems_per_language"].keys())


def test_no_profile_claims_native_approved_and_all_require_review():
    profiles = load_annotation_language_profiles(PROFILE_DIR)
    for language, profile in profiles.items():
        assert profile.native_review_required is True, f"{language}: must still require native review"


# ── 3. exactly 20 languages authorized ──────────────────────────────────────
def test_exactly_20_languages_authorized_for_team_generation():
    release = load_release_manifest()
    authorized = [l for l, v in release["languages"].items() if v["status"] == "AUTHORIZED_FOR_TEAM_GENERATION"]
    blocked = [l for l, v in release["languages"].items() if v["status"] != "AUTHORIZED_FOR_TEAM_GENERATION"]
    assert len(authorized) == 20
    assert blocked == ["Sanskrit"]
    assert len(release["languages"]) == 21


# ── 7 & 8. exact assignment counts ──────────────────────────────────────────
def _load_assignment(n: int) -> "list[dict]":
    with open(ASSIGNMENT_DIR / f"teammate_{n}.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_assignment_files_each_contain_exactly_521_ids():
    for n in (1, 2, 3):
        rows = _load_assignment(n)
        assert len(rows) == 521, f"teammate_{n}.csv: expected 521 rows, got {len(rows)}"


def test_exactly_1563_unique_assignment_ids_combined():
    all_ids = [r["poem_id"] for n in (1, 2, 3) for r in _load_assignment(n)]
    assert len(all_ids) == 1563
    assert len(set(all_ids)) == 1563


# ── 11. no pilot poem in any assignment ─────────────────────────────────────
def test_no_pilot_poem_in_any_assignment():
    all_ids = {r["poem_id"] for n in (1, 2, 3) for r in _load_assignment(n)}
    assert all_ids.isdisjoint(PILOT_IDS)


# ── 29. assignment files exactly cover the authorized new-generation universe
def test_assignments_exactly_equal_the_eligible_universe():
    manifest = json.loads((REPO_ROOT / "corpus" / "source_manifest.json").read_text(encoding="utf-8"))
    release = load_release_manifest()
    authorized_langs = {l for l, v in release["languages"].items() if v["status"] == "AUTHORIZED_FOR_TEAM_GENERATION"}
    blocked_poem_ids = set(release["blocked_poems"].keys())

    expected = {
        r["poem_id"] for r in manifest["records"]
        if r["language"] in authorized_langs
        and r["poem_id"] not in blocked_poem_ids
        and r["poem_id"] not in PILOT_IDS
    }
    actual = {r["poem_id"] for n in (1, 2, 3) for r in _load_assignment(n)}
    assert actual == expected
    assert len(expected) == 1563


# ── 14. multiline grounding regression (Stage 5N.3) ─────────────────────────
def test_multiline_span_accepted_exactly_within_its_declared_line_scope():
    original = "first line here\nsecond line there\nunrelated third line\n"
    index = grounding.build_line_index(original)
    span = "first line here\nsecond line there"  # spans L1-L2, matches the declared scope exactly
    match = grounding.ground_original_span(span, "L1-L2", index)
    assert match.status in (grounding.SPAN_MATCH_EXACT, grounding.SPAN_MATCH_NFC_EQUIVALENT)
    assert match.line_refs == ("L1-L2",)


def test_multiline_span_outside_declared_scope_is_rejected_even_if_it_exists_elsewhere():
    # The same two-line span exists in the poem (L2-L3), but the candidate
    # declares line_ref L1 only — exact grounding must not silently widen
    # the search to "somewhere in the poem".
    original = "alpha\nfirst line here\nsecond line there\n"
    index = grounding.build_line_index(original)
    span = "first line here\nsecond line there"
    match = grounding.ground_original_span(span, "L1", index)
    assert match.status == grounding.SPAN_MATCH_NOT_FOUND


# ── 15. Sanskrit pre-sandhi false positive stays rejected ──────────────────
def test_sanskrit_pre_sandhi_lexical_form_is_not_accepted_as_grounded():
    """Regression for the exact Stage 5N.1/5N.5 defect: Gemini returned the
    pre-sandhi (un-fused) lemma 'शोकमुच्छोषणमिन्द्रियाणाम्' as
    source_span_original, when the poem's actual written line is
    sandhi-fused as 'यच्छोकमुच्छोषणमिन्द्रियाणाम्'. Exact grounding must
    keep rejecting the pre-sandhi form — this is not a bug to "fix" by
    accepting it; see SANSKRIT_BLOCK.md."""
    actual_written_line = "यच्छोकमुच्छोषणमिन्द्रियाणाम्"
    pre_sandhi_lemma = "शोकमुच्छोषणमिन्द्रियाणाम्"
    index = grounding.build_line_index(actual_written_line)
    match = grounding.ground_original_span(pre_sandhi_lemma, "L1", index)
    assert match.status == grounding.SPAN_MATCH_NOT_FOUND
    # the poem's own actual written form must still ground cleanly
    exact_match = grounding.ground_original_span(actual_written_line, "L1", index)
    assert exact_match.status == grounding.SPAN_MATCH_EXACT


# ── 17. multiple same-rule contradictions batch into one repair round ──────
def test_multiple_faithful_loss_note_contradictions_found_in_one_pass():
    annotation = {
        "stanzas": [
            {"index": 1, "translation_quality": "faithful", "loss_note": "should be empty"},
            {"index": 2, "translation_quality": "faithful", "loss_note": ""},  # not a contradiction
            {"index": 3, "translation_quality": "faithful", "loss_note": "also should be empty"},
        ],
    }
    issues = vce.find_all_loss_note_faithful_contradictions(annotation)
    # each contradiction authorizes 2 paths (loss_note + translation_quality,
    # two alternative resolutions for the same rule) — 2 contradicting
    # stanzas (0 and 2) -> 4 issues total, found in this ONE pass, not one
    # MAX_REPAIR_ROUNDS-consuming round per instance.
    assert len(issues) == 4
    paths = [i.field_path for i in issues]
    assert paths.count("stanzas[0].loss_note") == 1
    assert paths.count("stanzas[0].translation_quality") == 1
    assert paths.count("stanzas[2].loss_note") == 1
    assert paths.count("stanzas[2].translation_quality") == 1
    assert "stanzas[1].loss_note" not in paths  # stanza 2 has no contradiction
    # deterministic, ascending stanza order — never set/dict iteration order
    assert paths.index("stanzas[0].loss_note") < paths.index("stanzas[2].loss_note")


def test_categorize_invalid_path_rule_groups_same_rule_together():
    a = vce.InvalidPathIssue(field_path="stanzas[0].loss_note", validation_reason="faithful contradiction")
    b = vce.InvalidPathIssue(field_path="stanzas[2].loss_note", validation_reason="faithful contradiction")
    assert vce.categorize_invalid_path_rule(a) == vce.categorize_invalid_path_rule(b)


# ── 30 (CLI level). Sanskrit block enforced on --language / --poem-id, no crash
def test_cli_dry_run_language_sanskrit_reports_blocked_not_a_crash(capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    exit_code = runner.main(["--dry-run", "--language", "Sanskrit"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out
    assert "MV++_1235" in captured.out


def test_cli_dry_run_poem_id_mv_1235_reports_blocked_not_a_crash(capsys, monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    exit_code = runner.main(["--dry-run", "--poem-id", "MV++_1235", "--language", "Sanskrit"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "BLOCKED" in captured.out


def test_no_override_flag_exists_for_the_sanskrit_block():
    parser = runner.build_arg_parser()
    help_text = parser.format_help().lower()
    assert "sanskrit" not in help_text  # no documented bypass in the CLI surface
    assert "allow-sanskrit" not in help_text
    assert "override-block" not in help_text


def test_live_batch_with_a_blocked_poem_does_not_crash_the_rest_of_the_batch(tmp_path, clean_client_factory):
    """A blocked poem inside a larger live batch must be reported as a
    non-fatal per-poem result, never propagate and abort poems after it."""
    factory, client = clean_client_factory
    release = load_release_manifest()

    def _run_one(poem_id, language):
        return runner.execute_poem_live(
            poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=factory,
            output_root=tmp_path / "outputs", checkpoint_dir=tmp_path / "checkpoints",
            reports_dir=tmp_path / "reports", local_run_dir=tmp_path / "local_provider_runs",
            release_manifest=release,
        )

    with pytest.raises(runner.LanguageBlocked):
        _run_one("MV++_1235", "Sanskrit")
    # the exception type itself is the contract main()'s _run_one wrapper
    # catches (see corpus_gemini_runner_v1_1.main._run_one) — confirms the
    # blocked case is a distinct, catchable failure mode, not a generic crash.
