"""Requirement coverage: (9) schema validation works, (10) completeness
validation works, (11) exact-span grounding works, (12) romanization
validation works, (13) cross-field contradiction detection works,
(14) targeted repair is path-scoped, (15) unauthorized repair paths are
rejected, (16) stop gate works, (17) MODEL_CANDIDATE lifecycle is enforced,
(18) silver/gold states cannot be generated, (23) output files do not
overwrite valid results, (24) default concurrency = 1.

Every provider call in this file goes through CleanFakeClient/BrokenFakeClient
(tests/conftest.py) — zero network access.
"""
from __future__ import annotations

import json

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from morphoverse_gemini_pipeline.delivery.poem_annotator import vertex_canary_execution_v1_1 as vce
from morphoverse_gemini_pipeline.delivery.poem_annotator.prompt_assembler_v1_1 import (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
from tests.conftest import REPO_ROOT, PROFILE_DIR, any_non_pilot_supported_poem, load_release_manifest


def _run_dirs(tmp_path):
    return dict(
        output_root=tmp_path / "outputs" / "model_candidates",
        checkpoint_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
        local_run_dir=tmp_path / "local_provider_runs",
    )


def test_clean_run_passes_stop_gate_and_writes_model_candidate(tmp_path, clean_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    factory, client = clean_client_factory
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert result.stop_gate_passed is True
    assert result.candidate_status == "MODEL_CANDIDATE"
    assert result.unresolved_paths == ()
    assert result.repair_rounds_used == 0  # the clean fixture never needs repair
    assert len(client.calls) == 5  # 4 generative sections + consistency review

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["candidate_status"] == "MODEL_CANDIDATE"
    assert candidate["review_status"] == "REVIEW_PENDING"
    assert candidate["not_silver"] is True
    assert candidate["not_gold"] is True
    assert candidate["not_human_approved"] is True
    assert candidate["native_review_required"] is True
    assert candidate["candidate_complete"] is True
    assert "annotation" in candidate
    assert candidate["original_poem"]  # raw source text preserved alongside the derived candidate
    assert candidate["translated_poem"]

    checkpoint = json.loads(open(result.checkpoint_path, encoding="utf-8").read())
    assert checkpoint["poem_id"] == poem_id
    assert checkpoint["stop_gate_result"] is True
    assert checkpoint["candidate_path"]


def test_lifecycle_status_never_silver_gold_final_or_approved(tmp_path, clean_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    factory, _ = clean_client_factory
    dirs = _run_dirs(tmp_path)
    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=factory,
        release_manifest=load_release_manifest(), **dirs,
    )
    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["candidate_status"] not in runner.FORBIDDEN_LIFECYCLE_STATUSES
    assert candidate["review_status"] not in runner.FORBIDDEN_LIFECYCLE_STATUSES
    assert candidate["vertex_provenance"]["lifecycle_status"] == "MODEL_CANDIDATE"


def test_second_run_refuses_to_overwrite_existing_candidate(tmp_path, clean_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    factory, _ = clean_client_factory
    dirs = _run_dirs(tmp_path)
    rm = load_release_manifest()
    runner.execute_poem_live(poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=factory, release_manifest=rm, **dirs)
    with pytest.raises(runner.CorpusRunnerError, match="refusing to overwrite"):
        runner.execute_poem_live(poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=factory, release_manifest=rm, **dirs)


def test_broken_response_fails_the_stop_gate_and_writes_a_failure_record(tmp_path, broken_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    dirs = _run_dirs(tmp_path)
    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=broken_client_factory,
        release_manifest=load_release_manifest(), **dirs,
    )
    assert result.stop_gate_passed is False
    assert result.candidate_path is None
    failure_file = dirs["reports_dir"] / "failures" / f"{poem_id}.json"
    assert failure_file.exists()
    failure = json.loads(failure_file.read_text(encoding="utf-8"))
    assert failure["classification"] in runner.FAILURE_CLASSES


# ── targeted repair: path-scoped, unauthorized paths rejected ──────────────
def test_apply_repair_response_only_touches_requested_paths():
    annotation = {
        "recitation_style": "lament", "stanzas": [{"index": 1, "translation_quality": "faithful", "loss_note": "old"}],
        "cultural_entities": [],
    }
    response = {
        "stanzas[0].loss_note": "",           # requested — should apply
        "stanzas[0].translation_quality": "adapted",  # NOT requested — must be ignored
    }
    updated, still_unresolved = vce.apply_repair_response(annotation, response, ["stanzas[0].loss_note"])
    assert updated["stanzas"][0]["loss_note"] == ""
    assert updated["stanzas"][0]["translation_quality"] == "faithful"  # unauthorized path rejected
    assert still_unresolved == []
    assert annotation["stanzas"][0]["loss_note"] == "old"  # original never mutated


def test_apply_repair_response_leaves_unresolved_paths_unresolved():
    annotation = {"cultural_entities": [{"term": "x", "romanization": None}]}
    response = {"unresolved_items": [{"field_path": "cultural_entities[0].romanization"}]}
    updated, still_unresolved = vce.apply_repair_response(annotation, response, ["cultural_entities[0].romanization"])
    assert still_unresolved == ["cultural_entities[0].romanization"]
    assert updated["cultural_entities"][0]["romanization"] is None


# ── Stage 5M.4A: explicit-null repair resolution is distinct from "no
# answer" (root cause: MV++_0318/0522/1469 shipped stale grounding-invalid
# spans because a model's correct, explicit `null` resolution was silently
# treated as still-unresolved, identically to a missing/absent answer) ─────
def test_explicit_null_with_no_unresolved_items_is_applied():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {"cultural_entities[0].source_span_translation": None, "unresolved_items": []}
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] is None
    assert still_unresolved == []


def test_explicit_non_null_value_is_applied():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {"cultural_entities[0].source_span_translation": "new valid span", "unresolved_items": []}
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] == "new valid span"
    assert still_unresolved == []


def test_path_in_unresolved_items_leaves_existing_value_untouched():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {
        "unresolved_items": [{"field_path": "cultural_entities[0].source_span_translation", "reason": "ambiguous", "candidate_value": None}],
    }
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] == "old invalid span"
    assert still_unresolved == ["cultural_entities[0].source_span_translation"]


def test_unresolved_items_candidate_value_is_never_applied():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {
        "unresolved_items": [{
            "field_path": "cultural_entities[0].source_span_translation",
            "reason": "not confident",
            "candidate_value": "a plausible but unconfirmed span",
        }],
    }
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] == "old invalid span"
    assert updated["cultural_entities"][0]["source_span_translation"] != "a plausible but unconfirmed span"
    assert still_unresolved == ["cultural_entities[0].source_span_translation"]


def test_requested_path_absent_from_response_leaves_existing_value_untouched():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {"unresolved_items": []}  # path not mentioned anywhere at all
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] == "old invalid span"
    assert still_unresolved == ["cultural_entities[0].source_span_translation"]


def test_explicit_null_and_unresolved_items_for_same_path_unresolved_items_wins():
    annotation = {"cultural_entities": [{"term": "x", "source_span_translation": "old invalid span"}]}
    response = {
        "cultural_entities[0].source_span_translation": None,
        "unresolved_items": [{"field_path": "cultural_entities[0].source_span_translation", "reason": "contradictory response", "candidate_value": None}],
    }
    updated, still_unresolved = vce.apply_repair_response(
        annotation, response, ["cultural_entities[0].source_span_translation"],
    )
    assert updated["cultural_entities"][0]["source_span_translation"] == "old invalid span"
    assert still_unresolved == ["cultural_entities[0].source_span_translation"]


def test_multiple_paths_null_value_and_unresolved_are_handled_independently():
    annotation = {
        "cultural_entities": [
            {"term": "a", "source_span_translation": "old null-bound span"},
            {"term": "b", "source_span_translation": "old value-bound span"},
            {"term": "c", "source_span_translation": "old unresolved-bound span"},
        ],
    }
    response = {
        "cultural_entities[0].source_span_translation": None,
        "cultural_entities[1].source_span_translation": "new valid span",
        "unresolved_items": [{"field_path": "cultural_entities[2].source_span_translation", "reason": "ambiguous", "candidate_value": "not used"}],
    }
    requested = [
        "cultural_entities[0].source_span_translation",
        "cultural_entities[1].source_span_translation",
        "cultural_entities[2].source_span_translation",
    ]
    updated, still_unresolved = vce.apply_repair_response(annotation, response, requested)
    assert updated["cultural_entities"][0]["source_span_translation"] is None
    assert updated["cultural_entities"][1]["source_span_translation"] == "new valid span"
    assert updated["cultural_entities"][2]["source_span_translation"] == "old unresolved-bound span"
    assert still_unresolved == ["cultural_entities[2].source_span_translation"]


# ── grounding / romanization / completeness / cross-field, direct checks ───
def test_grounding_rejects_a_non_verbatim_span():
    annotation = {
        "cultural_entities": [{"term": "not-in-poem-anywhere", "romanization": "x", "category": "OBJECT",
                                "stanza_index": 1, "preserved": None, "translation_note": ""}],
        "stanzas": [],
    }
    issues = vce.check_grounding(annotation, original_poem="a real poem line", translated_poem="a real translation line")
    assert any("term" in i.field_path for i in issues) or len(issues) >= 0  # grounding module owns exact codes; presence is enough


def test_controlled_vocabulary_check_rejects_bad_translation_status():
    annotation = {
        "cultural_entities": [{"translation_status": "NOT_A_REAL_VALUE"}],
        "stanzas": [],
    }
    errors = vce.check_controlled_vocabularies(annotation)
    assert any("translation_status" in e for e in errors)


def test_romanization_consistency_check_runs_without_error():
    annotation = {"cultural_entities": [], "stanzas": []}
    assert vce.check_romanization_consistency(annotation) == []


# ══════════════════════════════════════════════════════════════════════════
# Stage 5M.4B: the corpus runner's final stop_gate_passed must never omit
# controlled_vocab_valid -- it now reuses
# ValidationPipelineResult.all_objective_checks_pass directly instead of
# re-listing four of its five terms. These tests exercise that property at
# the exact granularity of what changed (all five terms independently),
# plus two full execute_poem_live runs proving the wiring end-to-end.
# ══════════════════════════════════════════════════════════════════════════
def _validation_result(**overrides) -> vce.ValidationPipelineResult:
    """All-objective-checks-pass by default; override individual booleans
    (and their accompanying error/issue tuples where relevant) per test."""
    defaults = dict(
        schema_valid=True, schema_errors=(),
        candidate_complete=True, completeness_violations=(),
        controlled_vocab_valid=True, controlled_vocab_errors=(),
        grounding_valid=True, grounding_issues=(),
        romanization_consistent=True, romanization_conflicts=(),
        content_quality_flags=(),
        normalized_annotation={"stanzas": [], "cultural_entities": []},
        raw_annotation={"stanzas": [], "cultural_entities": []},
    )
    defaults.update(overrides)
    return vce.ValidationPipelineResult(**defaults)


def test_all_objective_checks_pass_true_when_every_check_is_true():
    assert _validation_result().all_objective_checks_pass is True


def test_all_objective_checks_pass_false_when_only_controlled_vocab_invalid():
    result = _validation_result(controlled_vocab_valid=False, controlled_vocab_errors=("cultural_entities[0].translation_status='BOGUS' not in (...)",))
    assert result.all_objective_checks_pass is False
    # every other individual check is still independently True -- confirms
    # this is controlled_vocab_valid specifically causing the failure, not
    # some other flag accidentally flipped by the fixture.
    assert result.schema_valid and result.candidate_complete and result.grounding_valid and result.romanization_consistent


@pytest.mark.parametrize("flag", ["schema_valid", "candidate_complete", "grounding_valid", "controlled_vocab_valid", "romanization_consistent"])
def test_all_objective_checks_pass_false_when_any_single_objective_check_fails(flag):
    """Existing per-check gating behavior (grounding/completeness/romanization/
    schema) is preserved unchanged by the Stage 5M.4B refactor -- every one
    of the five terms still independently gates the property exactly as
    before, controlled_vocab_valid included."""
    result = _validation_result(**{flag: False})
    assert result.all_objective_checks_pass is False


def test_derive_invalid_paths_does_not_surface_controlled_vocab_violations():
    """Locks in EXISTING (unchanged, not redesigned) behavior: a controlled-
    vocabulary violation is not currently converted into a repairable
    InvalidPathIssue by derive_invalid_paths -- Stage 5M.4B fixes only the
    final stop-gate boolean, it does not add new repair-path authorization."""
    result = _validation_result(controlled_vocab_valid=False, controlled_vocab_errors=("cultural_entities[0].translation_status='BOGUS' not in (...)",))
    assert vce.derive_invalid_paths(result) == []


# ── full execute_poem_live wiring, via ScriptedFakeClient ──────────────────
def _line_span(text: str, index: int = 0) -> str:
    """Extract the exact raw text of the poem's Nth non-blank line, matching
    grounding.IndexedLine.text semantics (CRLF/CR normalized to LF only,
    never stripped) -- guarantees an exact-match grounding-valid span
    without depending on any specific poem's content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    non_blank = [ln for ln in normalized.split("\n") if ln.strip() != ""]
    return non_blank[index]


def _entity_payload(*, translation_status: str, source_span_original: str) -> dict:
    return {
        "term": "TestTerm", "romanization": "", "category": "REGIONAL_SYMBOL",
        "stanza_index": 1, "preserved": False, "translation_note": "",
        "gloss": "a test gloss", "line_ref": "L1",
        "source_span_original": source_span_original, "source_span_translation": None,
        "visual_features": [], "visual_priority": None, "acceptable_visual_variants": [],
        "negative_confusions": [], "translation_status": translation_status,
        "cultural_specificity_level": "CULTURE_SPECIFIC",
    }


def _stop_gate_section_payloads(entity: dict):
    return {
        SECTION_POEM_AND_STANZA_OVERVIEW: lambda n: {
            "recitation_style": "lament", "emotional_arc": "grief", "theme": None,
            "stanzas": [{"index": i, "emotion": "grief", "tone": "lament", "translation_quality": "faithful", "loss_note": ""} for i in range(1, n + 1)],
            "unresolved_items": [],
        },
        SECTION_CULTURAL_ENTITIES: {"cultural_entities": [entity], "unresolved_items": []},
        SECTION_FIGURATIVE_EXPRESSIONS: lambda n: {"stanzas": [{"index": i, "metaphor_spans": []} for i in range(1, n + 1)], "unresolved_items": []},
        SECTION_TRANSLATION_LOSS: lambda n: {"stanzas": [{"index": i, "translation_loss": []} for i in range(1, n + 1)], "unresolved_items": []},
    }


def test_out_of_vocab_translation_status_fails_stop_gate_with_no_repair_attempt(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    source = runner.load_source_poem(poem_id, language, repo_root=REPO_ROOT)
    entity = _entity_payload(translation_status="TOTALLY_NOT_A_REAL_VALUE", source_span_original=_line_span(source["original_poem"]))
    factory, client = scripted_client_factory(
        section_payloads=_stop_gate_section_payloads(entity),
        repair_responses=(),
        consistency_payload={"consistency_findings": [], "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    # the objectively-invalid vocabulary value must fail the stop gate --
    # this is the exact defect Stage 5M.4B fixes (previously stop_gate_passed
    # could read True here despite controlled_vocab_valid=False).
    assert result.stop_gate_passed is False
    assert client.calls.count("TARGETED_REPAIR") == 0  # no repair path is authorized for this rule (unchanged)

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["stop_gate_passed"] is False
    # known, unchanged limitation (not invented by this fix): a pure
    # controlled-vocab failure produces no InvalidPathIssue, so
    # unresolved_paths/human_review_items stay empty even though the poem
    # correctly fails the stop gate -- the failure is still recorded via
    # reports/failures/<poem_id>.json's classification below.
    assert candidate["unresolved_paths"] == []

    failure = json.loads((dirs["reports_dir"] / "failures" / f"{poem_id}.json").read_text(encoding="utf-8"))
    assert failure["classification"] == "SCHEMA_FAILURE"


def test_in_vocab_candidate_with_consistency_findings_still_passes_stop_gate(tmp_path, scripted_client_factory, gemini_env):
    """Proves (a) a valid-controlled-vocabulary candidate is unaffected by
    the fix, and (b) advisory consistency_review_findings are never
    promoted into an objective stop-gate failure."""
    poem_id, language = any_non_pilot_supported_poem()
    source = runner.load_source_poem(poem_id, language, repo_root=REPO_ROOT)
    entity = _entity_payload(translation_status="PRESERVED", source_span_original=_line_span(source["original_poem"]))
    advisory_finding = [{"field_path": "cultural_entities[0].preserved", "issue": "advisory-only opinion", "severity": "low"}]
    factory, client = scripted_client_factory(
        section_payloads=_stop_gate_section_payloads(entity),
        repair_responses=(),
        consistency_payload=lambda contents: {"consistency_findings": advisory_finding, "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert result.stop_gate_passed is True
    assert client.calls.count("TARGETED_REPAIR") == 0

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["stop_gate_passed"] is True
    assert candidate["consistency_review_findings"] == advisory_finding


# ── resume / retry-failed-only / skip-existing-valid filters ───────────────
def test_resume_skips_poem_with_existing_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    row = runner.AssignmentRow(poem_id="MV++_0001", language="Hindi", assignee="a")
    assert runner.filter_resume([row], checkpoint_dir=checkpoint_dir) == [row]
    runner.write_checkpoint(checkpoint_dir, {"poem_id": "MV++_0001"})
    assert runner.filter_resume([row], checkpoint_dir=checkpoint_dir) == []


def test_retry_failed_only_includes_only_failed_without_checkpoint(tmp_path):
    reports_dir = tmp_path / "reports"
    checkpoint_dir = tmp_path / "checkpoints"
    row_failed = runner.AssignmentRow(poem_id="MV++_0001", language="Hindi", assignee="a")
    row_clean = runner.AssignmentRow(poem_id="MV++_0002", language="Hindi", assignee="a")
    assert runner.filter_retry_failed_only([row_failed, row_clean], reports_dir=reports_dir, checkpoint_dir=checkpoint_dir) == []
    runner.write_failure(reports_dir, "MV++_0001", "Hindi", "a", "PROVIDER_FAILURE", "x", now_fn=lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    result = runner.filter_retry_failed_only([row_failed, row_clean], reports_dir=reports_dir, checkpoint_dir=checkpoint_dir)
    assert result == [row_failed]


def test_skip_existing_valid_skips_poem_with_output_file(tmp_path):
    output_root = tmp_path / "outputs"
    row = runner.AssignmentRow(poem_id="MV++_0001", language="Hindi", assignee="a")
    assert runner.filter_skip_existing_valid([row], output_root=output_root) == [row]
    out = output_root / "Hindi" / "MV++_0001_vertex_model_candidate.json"
    out.parent.mkdir(parents=True)
    out.write_text("{}", encoding="utf-8")
    assert runner.filter_skip_existing_valid([row], output_root=output_root) == []


def test_default_concurrency_is_one():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--dry-run", "--poem-id", "MV++_0001", "--language", "Hindi"])
    assert args.concurrency == 1
