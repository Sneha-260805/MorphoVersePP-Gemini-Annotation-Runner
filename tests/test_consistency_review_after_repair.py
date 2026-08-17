"""Stage 5M.3 regression tests: CONSISTENCY_REVIEW must run exactly once,
against the FINAL post-repair candidate -- never before the objective
validation/targeted-repair loop, and never a second time after.

Root-cause evidence: the live MV++_0001 Assamese smoke candidate
(outputs/model_candidates/Assamese/MV++_0001_vertex_model_candidate.json)
shipped a stale consistency_review_findings entry ("translation_quality is
marked as 'faithful', but the stanza contains a non-empty loss_note ...")
describing a pre-repair annotation state -- the final annotation actually
has loss_note = "" -- because the CONSISTENCY_REVIEW call ran BEFORE the
objective validation/targeted-repair loop and its findings were never
refreshed against the repaired candidate.

Every provider call in this file goes through ScriptedFakeClient
(tests/conftest.py) -- zero network access, zero real Vertex/Gemini calls.
"""
from __future__ import annotations

import json
import re

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from morphoverse_gemini_pipeline.delivery.poem_annotator.prompt_assembler_v1_1 import (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
from tests.conftest import REPO_ROOT, PROFILE_DIR, any_non_pilot_supported_poem, load_release_manifest

EXPECTED_GENERATIVE_ORDER = [
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS,
]


def _run_dirs(tmp_path):
    return dict(
        output_root=tmp_path / "outputs" / "model_candidates",
        checkpoint_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports",
        local_run_dir=tmp_path / "local_provider_runs",
    )


def _overview_payload(n, *, contradiction_stanza=None, loss_note_text="a real, evidence-supported loss"):
    stanzas = []
    for i in range(1, n + 1):
        loss_note = loss_note_text if i == contradiction_stanza else ""
        stanzas.append({
            "index": i, "emotion": "grief", "tone": "lament",
            "translation_quality": "faithful", "loss_note": loss_note,
        })
    return {"recitation_style": "lament", "emotional_arc": "grief", "theme": None, "stanzas": stanzas, "unresolved_items": []}


def _entities_payload(_n=None):
    return {"cultural_entities": [], "unresolved_items": []}


def _figurative_payload(n):
    return {"stanzas": [{"index": i, "metaphor_spans": []} for i in range(1, n + 1)], "unresolved_items": []}


def _translation_loss_payload(n):
    return {"stanzas": [{"index": i, "translation_loss": []} for i in range(1, n + 1)], "unresolved_items": []}


def _clean_section_payloads(contradiction_stanza=None):
    return {
        SECTION_POEM_AND_STANZA_OVERVIEW: lambda n: _overview_payload(n, contradiction_stanza=contradiction_stanza),
        SECTION_CULTURAL_ENTITIES: _entities_payload,
        SECTION_FIGURATIVE_EXPRESSIONS: _figurative_payload,
        SECTION_TRANSLATION_LOSS: _translation_loss_payload,
    }


def _extract_existing_candidate(contents: str) -> dict:
    m = re.search(
        r"===== BEGIN EXISTING CANDIDATE \(untrusted context only.*?=====\n(.*?)\n===== END EXISTING CANDIDATE",
        contents, re.DOTALL,
    )
    assert m, "no EXISTING CANDIDATE block found in CONSISTENCY_REVIEW prompt contents"
    return json.loads(m.group(1))


# ══════════════════════════════════════════════════════════════════════════
# 1. No-repair case
# ══════════════════════════════════════════════════════════════════════════
def test_no_repair_runs_consistency_review_exactly_once_after_generation(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    findings = [{"field_path": "stanzas[0].emotion", "issue": "advisory only", "severity": "low"}]
    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=None),
        repair_responses=(),
        consistency_payload=lambda contents: {"consistency_findings": findings, "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert client.calls == EXPECTED_GENERATIVE_ORDER + [SECTION_CONSISTENCY_REVIEW]
    assert len(client.raw_calls) == 5
    assert client.repair_call_index == 0
    assert result.repair_rounds_used == 0
    assert result.stop_gate_passed is True

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["consistency_review_findings"] == findings
    assert candidate["stop_gate_passed"] is True

    # consistency review received the exact final assembled candidate
    assert len(client.consistency_request_contents) == 1
    embedded = _extract_existing_candidate(client.consistency_request_contents[0])
    assert embedded == candidate["annotation"]


# ══════════════════════════════════════════════════════════════════════════
# 2. One-repair case -- critical regression
# ══════════════════════════════════════════════════════════════════════════
def test_one_repair_runs_consistency_review_only_after_repair(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    stale_loss_note = "a real, evidence-supported loss"

    def repair_response(contents):
        return {"stanzas[0].loss_note": ""}

    findings = [{"field_path": "stanzas[0].tone", "issue": "advisory only", "severity": "low"}]
    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=1),
        repair_responses=(repair_response,),
        consistency_payload=lambda contents: {"consistency_findings": findings, "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    # exactly the expected call order: 4 generation, 1 repair, 1 consistency
    assert client.calls == EXPECTED_GENERATIVE_ORDER + ["TARGETED_REPAIR", SECTION_CONSISTENCY_REVIEW]
    assert len(client.raw_calls) == 6
    assert client.repair_call_index == 1
    assert result.repair_rounds_used == 1

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["annotation"]["stanzas"][0]["loss_note"] == ""
    assert candidate["annotation"]["stanzas"][0]["translation_quality"] == "faithful"
    assert result.stop_gate_passed is True
    assert candidate["unresolved_paths"] == []

    # the candidate embedded in the CONSISTENCY_REVIEW request must be the
    # REPAIRED candidate -- the stale pre-repair loss_note text must be
    # entirely absent from that request's contents.
    assert len(client.consistency_request_contents) == 1
    consistency_contents = client.consistency_request_contents[0]
    assert stale_loss_note not in consistency_contents
    embedded = _extract_existing_candidate(consistency_contents)
    assert embedded["stanzas"][0]["loss_note"] == ""
    assert embedded == candidate["annotation"]

    # final consistency_review_findings reflect the model's post-repair
    # opinion, never a stale pre-repair contradiction finding
    assert candidate["consistency_review_findings"] == findings
    assert not any("loss_note" in f.get("issue", "") for f in candidate["consistency_review_findings"])


# ══════════════════════════════════════════════════════════════════════════
# 3. Two-repair-round case
# ══════════════════════════════════════════════════════════════════════════
def test_two_repair_rounds_runs_consistency_review_exactly_once_after_final_repair(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()

    def repair_round_1_leaves_unresolved(contents):
        # Round 1: model fails to resolve either requested path -> still
        # unresolved -> forces a second repair round.
        return {"unresolved_items": [
            {"field_path": "stanzas[0].loss_note", "reason": "uncertain", "candidate_value": None},
            {"field_path": "stanzas[0].translation_quality", "reason": "uncertain", "candidate_value": None},
        ]}

    def repair_round_2_resolves(contents):
        return {"stanzas[0].loss_note": ""}

    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=1),
        repair_responses=(repair_round_1_leaves_unresolved, repair_round_2_resolves),
        consistency_payload={"consistency_findings": [], "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert client.calls == EXPECTED_GENERATIVE_ORDER + ["TARGETED_REPAIR", "TARGETED_REPAIR", SECTION_CONSISTENCY_REVIEW]
    assert len(client.raw_calls) == 4 + 2 + 1
    assert result.repair_rounds_used == 2
    assert client.calls.count(SECTION_CONSISTENCY_REVIEW) == 1
    # consistency review must be the LAST call, after both repair rounds
    assert client.calls[-1] == SECTION_CONSISTENCY_REVIEW
    assert client.calls.index(SECTION_CONSISTENCY_REVIEW) > max(
        i for i, c in enumerate(client.calls) if c == "TARGETED_REPAIR"
    )

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["annotation"]["stanzas"][0]["loss_note"] == ""
    assert result.stop_gate_passed is True


# ══════════════════════════════════════════════════════════════════════════
# 4. Call order (explicit)
# ══════════════════════════════════════════════════════════════════════════
def test_one_repair_call_order_is_generation_then_repair_then_consistency(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=1),
        repair_responses=(lambda contents: {"stanzas[0].loss_note": ""},),
        consistency_payload={"consistency_findings": [], "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert client.calls == [
        SECTION_POEM_AND_STANZA_OVERVIEW,
        SECTION_CULTURAL_ENTITIES,
        SECTION_FIGURATIVE_EXPRESSIONS,
        SECTION_TRANSLATION_LOSS,
        "TARGETED_REPAIR",
        SECTION_CONSISTENCY_REVIEW,
    ]
    consistency_index = client.calls.index(SECTION_CONSISTENCY_REVIEW)
    repair_index = client.calls.index("TARGETED_REPAIR")
    assert consistency_index > repair_index, "CONSISTENCY_REVIEW must not occur before TARGETED_REPAIR"


# ══════════════════════════════════════════════════════════════════════════
# 5. Consistency findings remain non-blocking
# ══════════════════════════════════════════════════════════════════════════
def test_consistency_findings_never_trigger_repair_or_change_stop_gate(tmp_path, scripted_client_factory, gemini_env):
    poem_id, language = any_non_pilot_supported_poem()
    advisory_finding = [{
        "field_path": "cultural_entities[0].line_ref",
        "issue": "low-severity opinion that should never gate anything",
        "severity": "low",
    }]
    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=None),
        repair_responses=(),
        consistency_payload=lambda contents: {"consistency_findings": advisory_finding, "unresolved_items": []},
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    assert result.stop_gate_passed is True  # determined purely by objective validators
    assert result.candidate_status == "MODEL_CANDIDATE"
    assert client.repair_call_index == 0  # no repair triggered merely by a consistency finding
    assert client.calls.count("TARGETED_REPAIR") == 0

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["candidate_status"] == "MODEL_CANDIDATE"
    assert candidate["stop_gate_passed"] is True
    assert candidate["consistency_review_findings"] == advisory_finding  # stored as review metadata only


# ══════════════════════════════════════════════════════════════════════════
# 6. Provider failure during final consistency review
# ══════════════════════════════════════════════════════════════════════════
def test_consistency_review_provider_failure_does_not_corrupt_candidate(tmp_path, scripted_client_factory, gemini_env):
    """Existing contract (preserved, not invented): a CONSISTENCY_REVIEW
    provider-call failure is tolerated -- the poem still completes, the
    failed attempt is preserved in provenance/section_records for audit,
    and consistency_review_findings falls back to [] rather than an
    optimistic fabricated "consistency-clean" result. The objective stop
    gate is entirely unaffected, since it never reads consistency findings."""
    poem_id, language = any_non_pilot_supported_poem()
    factory, client = scripted_client_factory(
        section_payloads=_clean_section_payloads(contradiction_stanza=None),
        repair_responses=(),
        consistency_payload="FAIL",
    )
    dirs = _run_dirs(tmp_path)

    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR,
        client_factory=factory, assignee="teammate_1", release_manifest=load_release_manifest(), **dirs,
    )

    # the CONSISTENCY_REVIEW call was attempted (and only attempted once)
    assert client.calls == EXPECTED_GENERATIVE_ORDER + [SECTION_CONSISTENCY_REVIEW]

    # the poem is NOT failed on account of an advisory-only call failing;
    # the objective stop gate (schema/completeness/grounding/romanization)
    # still passes and a candidate is still written.
    assert result.stop_gate_passed is True
    assert result.candidate_path is not None

    candidate = json.loads(open(result.candidate_path, encoding="utf-8").read())
    assert candidate["consistency_review_findings"] == []  # no fabricated optimistic findings
    assert candidate["stop_gate_passed"] is True
    assert candidate["candidate_status"] == "MODEL_CANDIDATE"

    # provenance preserves the failed attempt for audit (not silently dropped)
    attempts = candidate["vertex_provenance"]["section_generation_attempts"]
    consistency_attempts = [a for a in attempts if a["section"] == SECTION_CONSISTENCY_REVIEW]
    assert len(consistency_attempts) == 1
    assert consistency_attempts[0]["outcome"] == "provider_call_failed"
