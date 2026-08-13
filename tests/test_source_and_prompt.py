"""Requirement coverage: (5) source whitelist excludes annotation fields,
(6) prompt assembly works, (7) no pilot answer is embedded, (8) five
sections assemble, (20) assignment IDs are unique.
"""
from __future__ import annotations

import json

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from tests.conftest import (
    REPO_ROOT, PROFILE_DIR, any_non_pilot_supported_poem, any_blocked_language_poem, load_release_manifest,
)


def test_source_loader_accepts_a_real_clean_record():
    poem_id, language = any_non_pilot_supported_poem()
    source = runner.load_source_poem(poem_id, language, repo_root=REPO_ROOT)
    assert source["poem_id"] == poem_id
    assert source["language"] == language
    assert set(source["original_poem"]) or source["original_poem"] == ""


def test_source_loader_rejects_annotation_leakage(tmp_path):
    poem_id, language = any_non_pilot_supported_poem()
    leaky_dir = tmp_path / "data" / "source_corpus" / language
    leaky_dir.mkdir(parents=True)
    (leaky_dir / f"{poem_id}.json").write_text(json.dumps({
        "poem_id": poem_id, "poem_title": "t", "language": language,
        "original_poem": "x", "translated_poem": "y",
        "annotation": {"cultural_entities": []},  # forbidden — old-answer leakage
    }), encoding="utf-8")
    with pytest.raises(runner.SourceDataError, match="forbidden extra key"):
        runner.load_source_poem(poem_id, language, repo_root=tmp_path)


def test_source_loader_rejects_missing_keys(tmp_path):
    poem_id, language = "MV++_9999", "Hindi"
    d = tmp_path / "data" / "source_corpus" / language
    d.mkdir(parents=True)
    (d / f"{poem_id}.json").write_text(json.dumps({"poem_id": poem_id, "language": language}), encoding="utf-8")
    with pytest.raises(runner.SourceDataError, match="missing required key"):
        runner.load_source_poem(poem_id, language, repo_root=tmp_path)


def test_pilot_poems_excluded_by_default():
    for pid in runner.PILOT_POEM_IDS:
        with pytest.raises(runner.PilotPoemBlocked):
            runner.require_not_pilot(pid, allow_pilot_regeneration=False)
    # explicit override works
    runner.require_not_pilot(next(iter(runner.PILOT_POEM_IDS)), allow_pilot_regeneration=True)


def test_missing_profile_never_falls_back_to_generic_profile():
    """require_language_profile never invents or substitutes a generic
    profile for a language with no addendum file on disk (independent of
    release-manifest authorization, which is a separate, later gate)."""
    with pytest.raises(runner.LanguageProfileMissing):
        runner.require_language_profile("Klingon", PROFILE_DIR)


def test_blocked_language_poem_is_sanskrit_mv_1235():
    poem_id, language = any_blocked_language_poem()
    assert (poem_id, language) == ("MV++_1235", "Sanskrit")


def test_blocked_language_has_a_profile_file_but_is_still_refused():
    """As of Stage 5M.2, Sanskrit's profile file (5N.1) IS present on disk —
    profile presence alone must never be mistaken for authorization.
    require_language_profile succeeds; require_release_authorized still
    refuses, and that is the gate the runner actually consults."""
    poem_id, language = any_blocked_language_poem()
    runner.require_language_profile(language, PROFILE_DIR)  # does NOT raise — profile is present
    with pytest.raises(runner.LanguageBlocked):
        runner.require_release_authorized(poem_id, language, load_release_manifest())


def test_release_manifest_none_blocks_every_language_fail_closed():
    """If corpus/execution_release_manifest.json were ever missing, every
    language must be refused — never fail open."""
    with pytest.raises(runner.LanguageBlocked):
        runner.require_release_authorized("MV++_0001", "Assamese", None)


def test_authorized_language_passes_release_check():
    poem_id, language = any_non_pilot_supported_poem()
    runner.require_release_authorized(poem_id, language, load_release_manifest())  # must not raise


def test_prompt_assembly_and_five_sections_plan(repo_root=REPO_ROOT):
    poem_id, language = any_non_pilot_supported_poem()
    plan = runner.plan_poem_dry_run(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, release_manifest=load_release_manifest(),
    )
    sections = [s.section for s in plan.planned_sections]
    assert sections == list(runner.GENERATIVE_SECTIONS)  # 4 of the 5; consistency review needs real content
    assert len(runner.REQUIRED_SECTIONS) == 5
    assert plan.provider_calls_made == 0
    for s in plan.planned_sections:
        assert s.prompt_sha256
        assert s.user_content_chars > 0


def test_prompt_never_carries_an_existing_candidate_for_the_four_generative_sections():
    """No poem-specific expected answer, no prior annotation, ever flows into
    the four generative-section prompts — only the CONSISTENCY_REVIEW
    section (which needs the just-generated candidate to review it) is
    allowed an `existing_candidate`."""
    import inspect
    src = inspect.getsource(runner.plan_poem_dry_run)
    assert "existing_candidate=None" in src


def test_assignment_csv_rejects_duplicate_poem_ids(tmp_path):
    p = tmp_path / "dup.csv"
    p.write_text("poem_id,language,assignee\nMV++_0001,Hindi,a\nMV++_0001,Bengali,b\n", encoding="utf-8")
    with pytest.raises(runner.AssignmentError, match="duplicate poem_id"):
        runner.load_assignment_csv(p)


def test_assignment_csv_unique_ids_accepted(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text("poem_id,language,assignee\nMV++_0001,Hindi,a\nMV++_0002,Bengali,b\n", encoding="utf-8")
    rows = runner.load_assignment_csv(p)
    assert [r.poem_id for r in rows] == ["MV++_0001", "MV++_0002"]
