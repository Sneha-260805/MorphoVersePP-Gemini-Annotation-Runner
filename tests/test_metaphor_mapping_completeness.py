"""Stage 5M.4F — metaphor_mapping is no longer required merely because
vehicle and tenor are populated.

Root cause (Stage 5M.4E audit): completeness_validator_v1_1.py's
check_figurative_expression_completeness required metaphor_mapping
unconditionally whenever vehicle+tenor were both non-blank, contradicting
models.py::validate_metaphor_mapping_v1_1 ("not every expression type
needs one") and shared_full_schema_prompt_v1_1.py's own field definition
("included only when both concepts are clearly evidenced, never invented
for every expression"). Corpus-wide (Stage 5M.4E), 1371/1373 figurative
expressions already carry a mapping regardless of expression_type; the
only 2 exceptions (MV++_0252, MV++_1502) are non-"metaphor" expression
types where the repair model correctly declined via unresolved_items
rather than invent a structured relation.

This file exercises the corrected rule purely through the public
completeness_validator_v1_1 API, plus a read-only regression check against
the two named real candidates (used only as evidence, never as
poem-specific production logic). No provider/network call anywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

from morphoverse_gemini_pipeline.delivery.poem_annotator import completeness_validator_v1_1 as cv
from morphoverse_gemini_pipeline.delivery.poem_annotator import models
from morphoverse_gemini_pipeline.delivery.poem_annotator import shared_full_schema_prompt_v1_1 as sfp

REPO_ROOT = Path(__file__).resolve().parents[1]

_BASE_EXPR = {
    "source_term": "x", "abstract_meaning": "y",
    "expression_type": "metaphor", "visualization_difficulty": "LOW",
    "line_ref": "L1", "source_span_original": "x",
}


def _expr(**overrides) -> dict:
    return {**_BASE_EXPR, "vehicle": "a vehicle", "tenor": "a tenor", "metaphor_mapping": None, **overrides}


# ── 1. vehicle+tenor populated, metaphor_mapping=null -> no violation ──────
def test_metaphor_mapping_null_with_vehicle_and_tenor_is_not_a_violation():
    expr = _expr()
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert not any(v.field_path.endswith(".metaphor_mapping") for v in violations)


# ── 2. metaphor_mapping present and valid -> remains complete ─────────────
def test_metaphor_mapping_present_and_valid_remains_complete():
    expr = _expr(metaphor_mapping={"vehicle_concept": "v", "tenor_concept": "t", "transferred_attributes": []})
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert not any(v.field_path.endswith(".metaphor_mapping") for v in violations)


# ── 3. structurally malformed mapping still rejected by SCHEMA validation ──
def test_structurally_malformed_metaphor_mapping_still_rejected_by_schema():
    # completeness never re-validates internal shape -- models.py does,
    # and this fix must not weaken that in any way.
    malformed = {"vehicle_concept": "", "tenor_concept": "t", "transferred_attributes": []}
    try:
        models.validate_metaphor_mapping_v1_1(malformed, "metaphor_mapping", "figurative_expression", 0)
        raised = False
    except models.SchemaValidationError:
        raised = True
    assert raised, "an empty vehicle_concept must still be rejected by schema validation"

    unexpected_key = {"vehicle_concept": "v", "tenor_concept": "t", "transferred_attributes": [], "extra": "nope"}
    try:
        models.validate_metaphor_mapping_v1_1(unexpected_key, "metaphor_mapping", "figurative_expression", 0)
        raised2 = False
    except models.SchemaValidationError:
        raised2 = True
    assert raised2, "an unexpected key inside metaphor_mapping must still be rejected"


# ── 4. vehicle missing -> existing violation remains ───────────────────────
def test_missing_vehicle_violation_remains():
    expr = _expr(vehicle=None)
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert any(v.field_path.endswith(".vehicle") for v in violations)


def test_missing_vehicle_does_not_also_require_metaphor_mapping():
    expr = _expr(vehicle="")
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert not any(v.field_path.endswith(".metaphor_mapping") for v in violations)


# ── 5. tenor missing -> existing violation remains ─────────────────────────
def test_missing_tenor_violation_remains():
    expr = _expr(tenor=None)
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert any(v.field_path.endswith(".tenor") for v in violations)


# ── 6. unrelated figurative completeness requirements unchanged ────────────
def test_unrelated_completeness_requirements_still_enforced():
    expr = _expr(line_ref=None, source_span_original=None, visualization_difficulty=None)
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    paths = {v.field_path for v in violations}
    assert any(p.endswith(".line_ref") for p in paths)
    assert any(p.endswith(".source_span_original") for p in paths)
    assert any(p.endswith(".visualization_difficulty") for p in paths)
    # metaphor_mapping was also null throughout and must still not appear
    assert not any(p.endswith(".metaphor_mapping") for p in paths)


def test_wordplay_still_exempt_from_visualization_difficulty():
    expr = _expr(expression_type="wordplay", visualization_difficulty=None)
    violations = cv.check_figurative_expression_completeness(expr, stanza_index=0, index=0)
    assert not any(v.field_path.endswith(".visualization_difficulty") for v in violations)


# ── 7/8. MV++_0252 and MV++_1502 offline revalidation regression ──────────
def _load_candidate(language: str, poem_id: str) -> dict:
    path = REPO_ROOT / "outputs" / "model_candidates" / language / f"{poem_id}_vertex_model_candidate.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_mv_0252_no_longer_fails_completeness_solely_on_metaphor_mapping():
    candidate = _load_candidate("Hindi", "MV++_0252")
    violations = cv.check_candidate_completeness(candidate["annotation"])
    assert not any(v.field_path.endswith(".metaphor_mapping") for v in violations)


def test_mv_1502_no_longer_fails_completeness_solely_on_metaphor_mapping():
    candidate = _load_candidate("Telugu", "MV++_1502")
    violations = cv.check_candidate_completeness(candidate["annotation"])
    assert not any(v.field_path.endswith(".metaphor_mapping") for v in violations)


# ── 9. no poem-specific production logic exists ────────────────────────────
def test_completeness_validator_source_has_no_poem_specific_logic():
    source = (REPO_ROOT / "morphoverse_gemini_pipeline" / "delivery" / "poem_annotator" / "completeness_validator_v1_1.py").read_text(encoding="utf-8")
    for forbidden in ("MV++_0252", "MV++_1502", "MV++_"):
        assert forbidden not in source


# ── contract version bump sanity ────────────────────────────────────────────
def test_completeness_contract_version_bumped_shared_prompt_unchanged():
    assert sfp.COMPLETENESS_CONTRACT_VERSION == "5K.2.1"
    assert sfp.SHARED_PROMPT_CONTRACT_VERSION == "5K.2.0"
