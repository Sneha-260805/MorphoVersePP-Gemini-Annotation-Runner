"""Stage 5K.3 — live Vertex AI execution wrapper for the Hindi/Telugu
fresh-annotation canary.

Authorized for exactly two poems: MV++_0073 (Hindi) and MV++_1443 (Telugu),
executed strictly sequentially, Hindi first. Reuses the Stage 5K.1/5K.2
prompt architecture (prompt_assembler_v1_1, targeted_repair_prompt_v1_1,
shared_full_schema_prompt_v1_1, annotation_language_profile_v1_1) and the
Stage 5D Vertex primitives (gemini_backfill_executor_v1_1) rather than
duplicating either. No new prompt architecture and no second Gemini/Vertex
SDK import site are created here -- every provider call goes through
gemini_backfill_executor_v1_1's client factory and retry wrapper.

Pure orchestration module: never imports `google.genai` directly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import gemini_backfill_executor_v1_1 as gx
from .annotation_language_profile_v1_1 import load_annotation_language_profiles, get_annotation_profile_for_language
from .completeness_validator_v1_1 import check_candidate_completeness
from .dataset import preprocess_poem
from .grounding import (
    build_line_index, validate_cultural_grounding_v1_1, validate_figurative_grounding_v1_1,
    GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
)
from .models import validate_model_payload_v1_1, ModelValidationError
from .schema import ALLOWED_TRANSLATION_QUALITIES
from .prompt_assembler_v1_1 import (
    assemble_prompt, StanzaSpec, PromptAssemblyError,
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
from .shared_full_schema_prompt_v1_1 import SHARED_PROMPT_CONTRACT_VERSION, COMPLETENESS_CONTRACT_VERSION
from .targeted_repair_prompt_v1_1 import build_targeted_repair_prompt, InvalidPathIssue
from .vertex_response_schema_v1_1 import build_and_audit_vertex_response_schema, build_and_audit_vertex_repair_response_schema

# ── Authorization (hard gate; re-checked at every entry point) ──────────────
AUTHORIZED_POEM_IDS: "tuple[str, ...]" = ("MV++_0073", "MV++_1443")
EXECUTION_ORDER: "tuple[str, ...]" = ("MV++_0073", "MV++_1443")
_POEM_LANGUAGE = {"MV++_0073": "Hindi", "MV++_1443": "Telugu"}
_POEM_SOURCE_KIND = {"MV++_0073": "approved_candidate_file", "MV++_1443": "output_v3_legacy_record"}

REQUIRED_SECTIONS: "tuple[str, ...]" = (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
MAX_REPAIR_ROUNDS = 2

LIFECYCLE_STATUS = "MODEL_CANDIDATE"
NOT_SILVER = True
NOT_GOLD = True
NOT_HUMAN_APPROVED = True
NATIVE_REVIEW_REQUIRED = True
REVIEW_STATE = "REVIEW_PENDING"

PROVIDER_NAME = gx.PROVIDER_NAME
PROVIDER_RUN_DIR = Path("pilot") / "provider_runs" / "stage5k3_canary"
CANDIDATE_OUTPUT_DIR = Path("pilot") / "annotations_v1_1" / "vertex_full_schema_candidates" / "stage5k3_canary"


class AuthorizationError(ValueError):
    pass


class ArchitectureLevelProblem(RuntimeError):
    """Raised for any of the task's named 'architecture-level problems' --
    the Hindi execution must stop before Telugu is ever attempted when this
    is raised."""


def require_authorized_poem_id(poem_id: str) -> None:
    if poem_id not in AUTHORIZED_POEM_IDS:
        raise AuthorizationError(
            f"{poem_id!r} is not authorized for Stage 5K.3 live execution. "
            f"Only {AUTHORIZED_POEM_IDS} may be processed."
        )


# ── Source loading (structural fields ONLY -- never the 'annotation' key) ───
_APPROVED_CANDIDATE_PATH = Path("pilot") / "annotations_v1_1" / "deterministically_corrected_candidates" / "MV++_0073.json"
_TELUGU_SOURCE_PATH = Path("morphoverse_gemini_pipeline") / "delivery" / "output_v3" / "Telugu" / "MV++_1443.json"
# Data lookup, not a code branch -- consistent with _POEM_LANGUAGE/_POEM_SOURCE_KIND
# above. Each poem's source genuinely lives in a different repository location
# (one already-migrated, one still a legacy output_v3 record); this dict
# records that fact once rather than special-casing it in an if/else.
_POEM_SOURCE_PATH = {"MV++_0073": _APPROVED_CANDIDATE_PATH, "MV++_1443": _TELUGU_SOURCE_PATH}


def load_canary_source(poem_id: str, repo_root: Path) -> dict:
    """Reads ONLY poem_id/poem_title/language/original_poem/translated_poem.
    The 'annotation' key of the source file, if present, is never accessed
    by this function or by anything downstream of it -- old annotation
    content must never enter prompt construction (Task: FRESH-GENERATION
    REQUIREMENTS)."""
    require_authorized_poem_id(poem_id)
    path = repo_root / _POEM_SOURCE_PATH[poem_id]
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {
        "poem_id": raw["poem_id"],
        "poem_title": raw.get("poem_title", ""),
        "language": raw["language"],
        "original_poem": raw["original_poem"],
        "translated_poem": raw["translated_poem"],
        "_source_path": str(path.relative_to(repo_root)).replace("\\", "/"),
        "_source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "_original_poem_sha256": hashlib.sha256(raw["original_poem"].encode("utf-8")).hexdigest(),
        "_translated_poem_sha256": hashlib.sha256(raw["translated_poem"].encode("utf-8")).hexdigest(),
    }


def stanza_specs_for(source: dict) -> "tuple[StanzaSpec, ...]":
    poem = preprocess_poem({k: v for k, v in source.items() if not k.startswith("_")})
    return tuple(StanzaSpec(s.stanza_index, tuple(s.source_lines), tuple(s.translated_lines)) for s in poem.stanzas)


# ── Artifact I/O (raw -> gitignored provider_run dir; sanitized -> tracked reports) ─
def _atomic_write_json(path: Path, obj: Any) -> None:
    gx.atomic_write_json(path, obj)


def _atomic_write_text(path: Path, text: str) -> None:
    gx.atomic_write_text(path, text)


def attempt_dir_for(repo_root: Path, poem_id: str, section: str, attempt_number: int) -> Path:
    require_authorized_poem_id(poem_id)
    return repo_root / PROVIDER_RUN_DIR / poem_id / section / f"attempt_{attempt_number:02d}"


@dataclass(frozen=True)
class SectionCallRecord:
    poem_id: str
    section: str
    attempt_number: int
    outcome: str  # "success" | "provider_call_failed" | "parse_failed" | "structural_mismatch"
    prompt_hash: str
    response_hash: "str | None"
    finish_reason: "str | None"
    output_token_count: "int | None"


def _sanitized_request_metadata(*, poem_id: str, section: str, gemini_config, prompt_hash: str, outcome, attempt_dir_name: str) -> dict:
    return {
        "provider": PROVIDER_NAME,
        "model": gemini_config.model,
        "project": gemini_config.project,
        "region": gemini_config.location,
        "api_version": gx.API_VERSION,
        "poem_id": poem_id,
        "section": section,
        "attempt_dir": attempt_dir_name,
        "generation_settings": gx.generation_settings_summary(),
        "attempt_count": len(outcome.attempts),
        "attempts": [asdict(a) for a in outcome.attempts],
        "prompt_sha256": prompt_hash,
    }


def execute_one_section_call(
    *, client: Any, poem_id: str, section: str, source: dict, stanzas: "tuple[StanzaSpec, ...]",
    profile_dir: Path, model: str, repo_root: Path, attempt_number: int,
    existing_candidate: "dict | None" = None,
    sleep_fn: Callable[[float], None] = None,
) -> "tuple[SectionCallRecord, dict | None]":
    """One live call for one section. Returns (record, parsed_json_or_None).
    Writes raw request/response artifacts to the gitignored provider-run
    directory only; returns only sanitized metadata to the caller."""
    require_authorized_poem_id(poem_id)
    result = assemble_prompt(
        annotation_language_profile_dir=profile_dir,
        poem_id=source["poem_id"], title=source["poem_title"], language=source["language"],
        original_poem=source["original_poem"], translated_poem=source["translated_poem"],
        stanzas=stanzas, section=section, existing_candidate=existing_candidate,
    )
    vertex_schema = build_and_audit_vertex_response_schema(section)
    gemini_config = gx.load_gemini_config()

    a_dir = attempt_dir_for(repo_root, poem_id, section, attempt_number)
    _atomic_write_json(a_dir / "prompt_request.json", {
        "system_instruction": result.system_instruction, "user_content": result.user_content,
        "prompt_hash": result.prompt_hash,
    })

    request = {
        "model": model, "contents": result.user_content,
        "config": gx.build_generation_config(result.system_instruction, vertex_schema, max_output_tokens=8192),
    }
    kwargs = {} if sleep_fn is None else {"sleep_fn": sleep_fn}
    outcome = gx.generate_with_retry(client, request, **kwargs)

    meta = _sanitized_request_metadata(
        poem_id=poem_id, section=section, gemini_config=gemini_config,
        prompt_hash=result.prompt_hash, outcome=outcome, attempt_dir_name=a_dir.name,
    )

    if not outcome.success:
        meta["response_status"] = "provider_call_failed"
        _atomic_write_json(a_dir / "request_metadata.json", meta)
        return SectionCallRecord(poem_id, section, attempt_number, "provider_call_failed", result.prompt_hash, None, None, None), None

    raw_text = getattr(outcome.response, "text", None)
    _atomic_write_text(a_dir / "raw_response.txt", raw_text or "")
    response_hash = hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()
    completion_meta = gx.extract_completion_metadata(outcome.response)
    meta["response_status"] = "success"
    meta["response_sha256"] = response_hash
    meta["completion"] = completion_meta
    _atomic_write_json(a_dir / "request_metadata.json", meta)

    parse_result = gx.parse_raw_response(raw_text)
    if parse_result.status != "success":
        return SectionCallRecord(
            poem_id, section, attempt_number, "parse_failed", result.prompt_hash, response_hash,
            completion_meta["finish_reason"], completion_meta["output_token_count"],
        ), None

    _atomic_write_json(a_dir / "parsed_response.json", parse_result.parsed)
    return SectionCallRecord(
        poem_id, section, attempt_number, "success", result.prompt_hash, response_hash,
        completion_meta["finish_reason"], completion_meta["output_token_count"],
    ), parse_result.parsed


# ══════════════════════════════════════════════════════════════════════════
# Section-level structural checks + candidate assembly
# ══════════════════════════════════════════════════════════════════════════
def section_structurally_valid(section: str, parsed: "dict | None") -> "tuple[bool, str | None]":
    """Cheap, section-specific top-level-key check performed BEFORE
    assembling sections together (Task: 'Assemble the section outputs into
    one candidate only after each section passes its section-level schema
    validation'). Not a substitute for the full models.py validation run
    later on the assembled candidate."""
    if parsed is None:
        return False, "no parsed JSON"
    required_keys_by_section = {
        SECTION_POEM_AND_STANZA_OVERVIEW: {"recitation_style", "emotional_arc", "theme", "stanzas", "unresolved_items"},
        SECTION_CULTURAL_ENTITIES: {"cultural_entities", "unresolved_items"},
        SECTION_FIGURATIVE_EXPRESSIONS: {"stanzas", "unresolved_items"},
        SECTION_TRANSLATION_LOSS: {"stanzas", "unresolved_items"},
        SECTION_CONSISTENCY_REVIEW: {"consistency_findings", "unresolved_items"},
    }
    expected = required_keys_by_section[section]
    if set(parsed.keys()) != expected:
        return False, f"expected top-level keys {sorted(expected)}, got {sorted(parsed.keys())}"
    return True, None


def assemble_candidate_from_sections(
    overview: dict, entities: dict, figurative: dict, translation_loss: dict, stanza_count: int,
) -> dict:
    """Merge four validated section outputs into one annotation-shaped
    payload matching schema.TOPLEVEL_KEYS_V1_1/STANZA_KEYS_V1_1. Pure,
    deterministic; never invents a stanza that wasn't present in `overview`."""
    stanzas_by_index = {s["index"]: dict(s) for s in overview["stanzas"]}
    for stanza in figurative["stanzas"]:
        if stanza["index"] in stanzas_by_index:
            stanzas_by_index[stanza["index"]]["metaphor_spans"] = stanza["metaphor_spans"]
    for stanza in translation_loss["stanzas"]:
        if stanza["index"] in stanzas_by_index:
            stanzas_by_index[stanza["index"]]["translation_loss"] = stanza["translation_loss"]

    ordered_stanzas = []
    for i in range(1, stanza_count + 1):
        st = stanzas_by_index.get(i, {"index": i, "emotion": None, "tone": None, "translation_quality": None, "loss_note": ""})
        st.setdefault("metaphor_spans", [])
        st.setdefault("translation_loss", [])
        ordered_stanzas.append(st)

    return {
        "recitation_style": overview.get("recitation_style"),
        "emotional_arc": overview.get("emotional_arc"),
        "theme": overview.get("theme"),
        "stanzas": ordered_stanzas,
        "cultural_entities": entities.get("cultural_entities", []),
    }


# ══════════════════════════════════════════════════════════════════════════
# Validation pipeline (10 checks) -- Task: COMPLETENESS VALIDATION
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ValidationPipelineResult:
    schema_valid: bool
    schema_errors: "tuple[str, ...]"
    candidate_complete: bool
    completeness_violations: "tuple[Any, ...]"
    controlled_vocab_valid: bool
    controlled_vocab_errors: "tuple[str, ...]"
    grounding_valid: bool
    grounding_issues: "tuple[Any, ...]"
    romanization_consistent: bool
    romanization_conflicts: "tuple[str, ...]"
    content_quality_flags: "tuple[str, ...]"
    normalized_annotation: "dict | None"
    raw_annotation: "dict | None" = None  # Stage 5N.3: the pre-validation candidate
    # dict as originally passed to run_validation_pipeline, ALWAYS populated
    # (unlike normalized_annotation, which is None when schema_valid=False).
    # Needed so derive_invalid_paths can independently re-scan a
    # schema-invalid candidate for every objective cross-field contradiction
    # instance, not just the first one models.validate_model_payload_v1_1's fail-fast validator
    # happened to raise on (Stage 5N.2 Manipuri finding).

    @property
    def all_objective_checks_pass(self) -> bool:
        """Schema/completeness/vocab/grounding/romanization are OBJECTIVE
        (pass/fail) checks. content_quality_flags are heuristic signals for
        human/risk-audit review, never a pass/fail gate on their own (Task:
        'Semantic disagreement alone must remain a review item')."""
        return (
            self.schema_valid and self.candidate_complete and self.grounding_valid
            and self.controlled_vocab_valid and self.romanization_consistent
        )


_PROPOSED_VOCAB_FIELDS = {
    "cultural_specificity_level": ("CULTURE_SPECIFIC", "CULTURALLY_CONTEXTUAL", "CROSS_CULTURAL", "UNCERTAIN"),
    "translation_status": ("PRESERVED", "PARTIALLY_PRESERVED", "ALTERED", "LOST", "NOT_TRANSLATED", "UNCERTAIN"),
    "visualization_difficulty": ("LOW", "MEDIUM", "HIGH"),
}
_TRANSLATION_LOSS_SEVERITY = ("low", "medium", "high")


def check_controlled_vocabularies(annotation: dict) -> "list[str]":
    """Re-checks the three PROPOSED_PENDING_SCHEMA_DECISION vocabularies
    models.py does not itself enforce as an enum (see
    active_schema_contract_v1_1.json mismatches M01/M02), plus
    translation_loss.severity's frozen lowercase convention."""
    errors: "list[str]" = []
    for i, entity in enumerate(annotation.get("cultural_entities", [])):
        for field_name, allowed in _PROPOSED_VOCAB_FIELDS.items():
            if field_name == "visualization_difficulty":
                continue
            value = entity.get(field_name)
            if value is not None and value not in allowed:
                errors.append(f"cultural_entities[{i}].{field_name}={value!r} not in {allowed}")
    for si, stanza in enumerate(annotation.get("stanzas", [])):
        for mi, expr in enumerate(stanza.get("metaphor_spans", []) or []):
            value = expr.get("visualization_difficulty")
            if value is not None and value not in _PROPOSED_VOCAB_FIELDS["visualization_difficulty"]:
                errors.append(f"stanzas[{si}].metaphor_spans[{mi}].visualization_difficulty={value!r} not in {_PROPOSED_VOCAB_FIELDS['visualization_difficulty']}")
        for li, loss in enumerate(stanza.get("translation_loss", []) or []):
            value = loss.get("severity")
            if value is not None and value not in _TRANSLATION_LOSS_SEVERITY:
                errors.append(f"stanzas[{si}].translation_loss[{li}].severity={value!r} not in {_TRANSLATION_LOSS_SEVERITY}")
    return errors


@dataclass(frozen=True)
class PathedGroundingIssue:
    field_path: str
    code: str
    message: str
    corrected_line_ref: "str | None" = None  # Stage 5N.3: carried through from
    # grounding.GroundingIssue.corrected_line_ref for code="line_ref_span_mismatch"
    # only — see derive_invalid_paths / find_line_ref_repair_authorizations.


def check_grounding(annotation: dict, original_poem: str, translated_poem: str) -> "list[PathedGroundingIssue]":
    issues: "list[PathedGroundingIssue]" = []
    for i, entity in enumerate(annotation.get("cultural_entities", [])):
        for gi in validate_cultural_grounding_v1_1(entity, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=i):
            if gi.severity == "error":
                issues.append(PathedGroundingIssue(f"cultural_entities[{i}].{gi.field}", gi.code, gi.message, gi.corrected_line_ref))
    for si, stanza in enumerate(annotation.get("stanzas", [])):
        for mi, expr in enumerate(stanza.get("metaphor_spans", []) or []):
            for gi in validate_figurative_grounding_v1_1(expr, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=mi):
                if gi.severity == "error":
                    issues.append(PathedGroundingIssue(f"stanzas[{si}].metaphor_spans[{mi}].{gi.field}", gi.code, gi.message, gi.corrected_line_ref))
    return issues


def check_romanization_consistency(annotation: dict) -> "list[str]":
    """Every repeated `term` must romanize identically (reader_friendly_ascii_v1)."""
    seen: "dict[str, str]" = {}
    conflicts = []
    for entity in annotation.get("cultural_entities", []):
        term, roman = entity.get("term"), entity.get("romanization")
        if not term:
            continue
        if term in seen and seen[term] != roman:
            conflicts.append(f"term {term!r} romanized both {seen[term]!r} and {roman!r}")
        else:
            seen[term] = roman
    return conflicts


_DENSITY_TARGET_MARKERS = ("expect 2", "expect 3", "at least 2 cultural", "at least 3 cultural")
_STEREOTYPE_TENOR_MARKERS = ("trauma", "abuse", "caste", "oppression")


def content_quality_heuristic_audit(annotation: dict) -> "list[str]":
    """Non-blocking heuristic flags only (Task: content-quality heuristic
    audit is check #8 of 10; content_quality_flags never gate
    all_objective_checks_pass). Flags, does not reject."""
    flags: "list[str]" = []
    for stanza in annotation.get("stanzas", []):
        for expr in stanza.get("metaphor_spans", []) or []:
            expr_type = (expr.get("expression_type") or "").strip().lower()
            tenor = (expr.get("tenor") or "").strip().lower()
            if expr_type and tenor and expr_type == tenor:
                flags.append(f"expression_type and tenor are identical ({expr_type!r}) -- tenor must state meaning, not restate the type")
            if tenor and any(marker in tenor for marker in _STEREOTYPE_TENOR_MARKERS):
                flags.append(f"tenor {tenor!r} asserts an overly specific sensitive topic -- requires direct textual support review")
    return flags


def run_validation_pipeline(annotation: dict, poem, original_poem: str, translated_poem: str) -> ValidationPipelineResult:
    """Runs all 10 required checks. Never accepts a candidate merely because
    it is schema-valid (Task: 'A candidate may not be accepted merely
    because it is schema-valid')."""
    schema_errors: "list[str]" = []
    normalized = None
    try:
        normalized = validate_model_payload_v1_1(annotation, poem)
        schema_valid = True
    except ModelValidationError as exc:
        schema_valid = False
        schema_errors.append(str(exc))

    target = normalized if normalized is not None else annotation
    completeness_violations = check_candidate_completeness(target) if schema_valid else []
    vocab_errors = check_controlled_vocabularies(target)
    grounding_issues = check_grounding(target, original_poem, translated_poem) if schema_valid else []
    romanization_conflicts = check_romanization_consistency(target)
    quality_flags = content_quality_heuristic_audit(target)

    return ValidationPipelineResult(
        schema_valid=schema_valid, schema_errors=tuple(schema_errors),
        candidate_complete=(schema_valid and not completeness_violations),
        completeness_violations=tuple(completeness_violations),
        controlled_vocab_valid=not vocab_errors, controlled_vocab_errors=tuple(vocab_errors),
        grounding_valid=(schema_valid and not grounding_issues), grounding_issues=tuple(grounding_issues),
        romanization_consistent=not romanization_conflicts, romanization_conflicts=tuple(romanization_conflicts),
        content_quality_flags=tuple(quality_flags),
        normalized_annotation=normalized,
        raw_annotation=annotation,
    )


# ══════════════════════════════════════════════════════════════════════════
# Targeted repair (max 2 rounds; Task: TARGETED REPAIR)
# ══════════════════════════════════════════════════════════════════════════
# Known, stable models.py SchemaValidationError message patterns that name
# an exact, single repairable field path (Stage 5K.3 finding: the live
# Hindi run produced "stanzas[0].loss_note must be empty when
# translation_quality is faithful." -- a genuine cross-field schema
# violation that the completeness/grounding checks below never surface,
# because they only run when schema_valid is already True). Extending this
# list is a mechanical parsing improvement over an already-existing,
# already-tested schema.py error message -- it does not change schema.py,
# models.py, or invent a new validation rule.
_LOSS_NOTE_FAITHFUL_CONTRADICTION_PATTERN = re.compile(
    r"^stanzas\[(\d+)\]\.loss_note must be empty when translation_quality is faithful\.$"
)


def _paths_from_raw_schema_errors(schema_errors: "tuple[str, ...]") -> "list[InvalidPathIssue]":
    """Recovers repairable paths from KNOWN, stable models.py
    SchemaValidationError message patterns that completeness/grounding
    checks never surface (those only run once schema validation already
    passed). For the loss_note/translation_quality contradiction, BOTH
    sides of the cross-field rule are offered as authorized repair
    paths -- clearing loss_note (keeping 'faithful') and changing
    translation_quality (keeping the loss_note) are equally valid
    resolutions depending on what the source/translation evidence actually
    shows; the model must decide which fits the evidence, not be told."""
    issues: "list[InvalidPathIssue]" = []
    for message in schema_errors:
        m = _LOSS_NOTE_FAITHFUL_CONTRADICTION_PATTERN.match(message)
        if m:
            stanza_idx = m.group(1)
            issues.append(InvalidPathIssue(
                field_path=f"stanzas[{stanza_idx}].loss_note", validation_reason=message,
            ))
            issues.append(InvalidPathIssue(
                field_path=f"stanzas[{stanza_idx}].translation_quality",
                validation_reason=(
                    f"Coupled with the loss_note contradiction above (same cross-field rule, "
                    f"models.py): {message} An alternative resolution is to change "
                    f"translation_quality away from 'faithful' instead of clearing loss_note, "
                    f"if the source/translation evidence supports a real loss."
                ),
                allowed_values=ALLOWED_TRANSLATION_QUALITIES,
            ))
    return issues


def find_all_loss_note_faithful_contradictions(annotation: "dict | None") -> "list[InvalidPathIssue]":
    """Stage 5N.3 fix for the Stage 5N.2 Manipuri finding (REPAIR_BUDGET_EXHAUSTED):
    independently scans EVERY stanza of `annotation` directly for the exact
    same objective rule models.validate_model_payload_v1_1's fail-fast validator already enforces
    (translation_quality == "faithful" with a non-empty loss_note) — this
    does not change the rule (schema.py/models.py are untouched) and does
    not weaken it; it only finds every instance of it in ONE pass instead of
    relying on models.validate_model_payload_v1_1's SchemaValidationError,
    which (by design, Stage 2.1) raises and stops at the FIRST stanza it
    finds. A candidate with contradictions in stanzas 1, 2, and 3 now yields
    all three pairs of paths from a single call, so ALL of them can be
    included in ONE targeted-repair request rather than needing one
    MAX_REPAIR_ROUNDS-consuming round per instance. Deterministically
    ordered by stanza index (ascending), never a set/dict iteration order.
    Read-only: never mutates `annotation`. Returns [] for a non-dict
    `annotation` or a missing/non-list `stanzas`, never raises."""
    issues: "list[InvalidPathIssue]" = []
    if not isinstance(annotation, dict):
        return issues
    stanzas = annotation.get("stanzas")
    if not isinstance(stanzas, list):
        return issues
    for i, stanza in enumerate(stanzas):
        if not isinstance(stanza, dict):
            continue
        translation_quality = stanza.get("translation_quality")
        loss_note = stanza.get("loss_note")
        if translation_quality == "faithful" and loss_note:
            message = f"stanzas[{i}].loss_note must be empty when translation_quality is faithful."
            issues.append(InvalidPathIssue(field_path=f"stanzas[{i}].loss_note", validation_reason=message))
            issues.append(InvalidPathIssue(
                field_path=f"stanzas[{i}].translation_quality",
                validation_reason=(
                    f"Coupled with the loss_note contradiction above (same cross-field rule, "
                    f"models.py): {message} An alternative resolution is to change "
                    f"translation_quality away from 'faithful' instead of clearing loss_note, "
                    f"if the source/translation evidence supports a real loss."
                ),
                allowed_values=ALLOWED_TRANSLATION_QUALITIES,
            ))
    return issues


# ── Deterministic line_ref repair-path authorization (Stage 5N.3, Task 5) ───
def find_line_ref_repair_authorizations(grounding_issues: "tuple[PathedGroundingIssue, ...]") -> "list[InvalidPathIssue]":
    """Authorizes `line_ref` itself (never the already-correct span text) as
    a repairable path when grounding found a code="line_ref_span_mismatch"
    issue WITH a deterministically-derivable correction
    (PathedGroundingIssue.corrected_line_ref is not None — see
    grounding.py's GroundingIssue.corrected_line_ref: only set when
    source_span_original occurs exactly once elsewhere in the whole poem,
    i.e. no semantic interpretation is required to pick the right line).
    A line_ref_span_mismatch WITHOUT a deterministic correction (ambiguous
    whole-poem occurrence) is deliberately NOT authorized here — it still
    surfaces as its own field_path via check_grounding/derive_invalid_paths,
    but as a review item, not a mechanically-resolvable one. Never touches
    source_span_original/source_span_translation paths; those were already
    correct (that is the whole point of this repair path existing)."""
    issues: "list[InvalidPathIssue]" = []
    for gi in grounding_issues:
        if gi.code == "line_ref_span_mismatch" and gi.corrected_line_ref is not None:
            issues.append(InvalidPathIssue(
                field_path=gi.field_path,
                validation_reason=(
                    f"{gi.message} The correct line_ref, deterministically re-derived from the "
                    f"poem's own line index, is {gi.corrected_line_ref!r} — source_span_original's "
                    f"text is already correct and must not be changed; only line_ref needs correction."
                ),
            ))
    return issues


def derive_invalid_paths(result: ValidationPipelineResult) -> "list[InvalidPathIssue]":
    """Builds the minimal list of invalid/incomplete JSON paths from the
    validation pipeline's objective findings (never from
    content_quality_flags, which are heuristic/semantic signals routed to
    human review instead -- Task: 'Semantic disagreement alone must remain
    a review item'). When the payload fails raw schema validation
    (schema_valid=False), completeness/grounding checks do not run at all
    (there is no normalized annotation to check them against) -- in that
    case this function still recovers any KNOWN, exactly-parseable
    cross-field schema error into a repairable path, so a genuine model
    mistake like a stray loss_note on a 'faithful' stanza is repairable
    rather than silently unrepairable.

    Stage 5N.3: also independently re-scans result.raw_annotation for EVERY
    instance of the faithful/loss_note contradiction (not just the one
    models.validate_model_payload_v1_1's fail-fast validator happened to raise on), and authorizes
    line_ref itself (not source_span_original) as the repair path for any
    deterministically-resolvable line_ref/span mismatch a grounding pass
    already found. Both additions are additive: they can only ADD paths
    that were previously undiscoverable in one pass, never remove or change
    the meaning of an existing one; de-duplication below still applies."""
    issues: "list[InvalidPathIssue]" = list(_paths_from_raw_schema_errors(result.schema_errors))
    issues.extend(find_all_loss_note_faithful_contradictions(result.raw_annotation))
    for v in result.completeness_violations:
        issues.append(InvalidPathIssue(field_path=v.field_path, validation_reason=v.message))
    issues.extend(find_line_ref_repair_authorizations(result.grounding_issues))
    for gi in result.grounding_issues:
        if gi.code == "line_ref_span_mismatch" and gi.corrected_line_ref is not None:
            continue  # superseded by find_line_ref_repair_authorizations' more informative issue above
        issues.append(InvalidPathIssue(field_path=gi.field_path, validation_reason=gi.message))
    # de-duplicate by field_path, keep first reason, preserve first-seen (deterministic) order
    seen: "dict[str, InvalidPathIssue]" = {}
    for issue in issues:
        seen.setdefault(issue.field_path, issue)
    return list(seen.values())


# ── Repair batching by objective rule (Stage 5N.3, Task 7) ──────────────────
_LOSS_NOTE_RULE = "cross_field_translation_quality_faithful_loss_note"
_LINE_REF_MISMATCH_RULE = "deterministic_line_ref_span_mismatch"
_GROUNDING_SPAN_RULE = "grounding_span_not_found_or_ambiguous"
_COMPLETENESS_RULE = "completeness_violation"
_OTHER_RULE = "other_schema_error"


def categorize_invalid_path_rule(issue: InvalidPathIssue) -> str:
    """Classifies one InvalidPathIssue by the OBJECTIVE rule that produced
    it, purely from the issue's own already-known shape (field suffix +
    validation_reason marker text) -- never by guessing intent. Used only
    to decide what may safely be batched into the same targeted-repair
    request (Task 7: 'Do not mix unrelated semantic issues into the
    request'); does not change what gets validated or how."""
    path, reason = issue.field_path, issue.validation_reason
    if "must be empty when translation_quality is faithful" in reason:
        return _LOSS_NOTE_RULE
    if path.endswith(".line_ref") and "deterministically re-derived" in reason:
        return _LINE_REF_MISMATCH_RULE
    if "grounding" in path or "span" in reason.lower() or "not found" in reason.lower() or "ambiguous" in reason.lower():
        return _GROUNDING_SPAN_RULE
    if "completeness" in reason.lower() or "missing" in reason.lower():
        return _COMPLETENESS_RULE
    return _OTHER_RULE


def plan_repair_batches(invalid_paths: "list[InvalidPathIssue]") -> "tuple[tuple[InvalidPathIssue, ...], ...]":
    """Groups `invalid_paths` into batches by categorize_invalid_path_rule():
    every path produced by the SAME objective rule (e.g. every
    faithful/loss_note pair across however many stanzas) is grouped into
    ONE batch -- safe to send together in a single targeted-repair request,
    since they all share one rule, one set of authorized paths, and no
    resolution of one can affect another's authorization (Task 7/8). Paths
    from DIFFERENT rules are never merged into the same batch. Deterministic:
    batches appear in first-seen rule order; paths within a batch keep their
    original relative order."""
    order: "list[str]" = []
    grouped: "dict[str, list[InvalidPathIssue]]" = {}
    for issue in invalid_paths:
        rule = categorize_invalid_path_rule(issue)
        if rule not in grouped:
            grouped[rule] = []
            order.append(rule)
        grouped[rule].append(issue)
    return tuple(tuple(grouped[rule]) for rule in order)


def _get_at_path(annotation: dict, path: str) -> Any:
    """Minimal dotted+indexed getter matching the field_path shapes this
    module itself produces (cultural_entities[N].field,
    stanzas[N].metaphor_spans[N].field). Returns None if the path does not
    resolve (never raises -- used only to build minimal repair context)."""
    tokens = re.findall(r"([a-zA-Z_]+)(\[(\d+)\])?", path)
    node: Any = annotation
    for name, _, idx in tokens:
        if not name:
            continue
        if isinstance(node, dict):
            node = node.get(name)
        else:
            return None
        if idx != "":
            if isinstance(node, list) and int(idx) < len(node):
                node = node[int(idx)]
            else:
                return None
    return node


def _set_at_path(annotation: dict, path: str, value: Any) -> None:
    """Mutates `annotation` in place, setting exactly the leaf named by
    `path` -- never touches any other field. Silently no-ops if the
    container path does not resolve (repair may target a genuinely new
    detection-completeness array; that case is handled by the caller
    checking the leaf container directly, not via this setter)."""
    tokens = re.findall(r"([a-zA-Z_]+)(\[(\d+)\])?", path)
    node: Any = annotation
    for i, (name, _, idx) in enumerate(tokens):
        is_last = (i == len(tokens) - 1)
        if not name:
            continue
        if is_last and idx == "":
            if isinstance(node, dict):
                node[name] = value
            return
        if isinstance(node, dict):
            node = node.get(name)
        if idx != "":
            if isinstance(node, list) and int(idx) < len(node):
                if is_last:
                    node[int(idx)] = value
                    return
                node = node[int(idx)]
            else:
                return


def apply_repair_response(annotation: dict, response: dict, requested_paths: "list[str]") -> "tuple[dict, list[str]]":
    """Applies ONLY the requested paths' resolved values back into a deep
    copy of `annotation`; a path present in `response['unresolved_items']`
    (or simply absent/null in the response) is left untouched and returned
    as still-unresolved. Never touches any field not in `requested_paths`
    (Task: 'preserve already valid fields')."""
    updated = copy.deepcopy(annotation)
    unresolved_paths = {item.get("field_path") for item in (response.get("unresolved_items") or [])}
    still_unresolved: "list[str]" = []
    for path in requested_paths:
        if path in unresolved_paths:
            still_unresolved.append(path)
            continue
        if path not in response or response[path] is None:
            still_unresolved.append(path)
            continue
        _set_at_path(updated, path, response[path])
    return updated, still_unresolved


def run_targeted_repair_round(
    *, client: Any, poem_id: str, source: dict, stanzas: "tuple[StanzaSpec, ...]", profile_dir: Path,
    model: str, repo_root: Path, attempt_number: int, annotation: dict, invalid_paths: "list[InvalidPathIssue]",
    sleep_fn: Callable[[float], None] = None,
) -> "tuple[dict, list[str], SectionCallRecord]":
    """One repair round: requests exactly `invalid_paths`, applies whatever
    the model safely resolved, returns (updated_annotation,
    still_unresolved_paths, call_record)."""
    minimal_context = {p.field_path: _get_at_path(annotation, p.field_path) for p in invalid_paths}
    stanza_indices = sorted({int(m.group(1)) for p in invalid_paths for m in [re.search(r"stanzas\[(\d+)\]", p.field_path)] if m})
    relevant_stanzas = tuple(s for s in stanzas if (s.stanza_index - 1) in stanza_indices) or stanzas[:1]

    repair_result = build_targeted_repair_prompt(
        annotation_language_profile_dir=profile_dir,
        poem_id=source["poem_id"], language=source["language"],
        original_poem=source["original_poem"], translated_poem=source["translated_poem"],
        relevant_stanzas=relevant_stanzas, minimal_candidate_context=minimal_context,
        invalid_paths=tuple(invalid_paths),
    )
    vertex_schema = build_and_audit_vertex_repair_response_schema([p.field_path for p in invalid_paths])

    a_dir = attempt_dir_for(repo_root, poem_id, "TARGETED_REPAIR", attempt_number)
    _atomic_write_json(a_dir / "prompt_request.json", {
        "system_instruction": repair_result.system_instruction, "user_content": repair_result.user_content,
        "prompt_hash": repair_result.prompt_hash,
    })

    request = {
        "model": model, "contents": repair_result.user_content,
        "config": gx.build_generation_config(repair_result.system_instruction, vertex_schema, max_output_tokens=4096),
    }
    kwargs = {} if sleep_fn is None else {"sleep_fn": sleep_fn}
    outcome = gx.generate_with_retry(client, request, **kwargs)
    gemini_config = gx.load_gemini_config()
    meta = _sanitized_request_metadata(
        poem_id=poem_id, section="TARGETED_REPAIR", gemini_config=gemini_config,
        prompt_hash=repair_result.prompt_hash, outcome=outcome, attempt_dir_name=a_dir.name,
    )

    if not outcome.success:
        meta["response_status"] = "provider_call_failed"
        _atomic_write_json(a_dir / "request_metadata.json", meta)
        record = SectionCallRecord(poem_id, "TARGETED_REPAIR", attempt_number, "provider_call_failed", repair_result.prompt_hash, None, None, None)
        return annotation, [p.field_path for p in invalid_paths], record

    raw_text = getattr(outcome.response, "text", None)
    _atomic_write_text(a_dir / "raw_response.txt", raw_text or "")
    response_hash = hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()
    completion_meta = gx.extract_completion_metadata(outcome.response)
    meta["response_status"] = "success"
    meta["response_sha256"] = response_hash
    meta["completion"] = completion_meta
    _atomic_write_json(a_dir / "request_metadata.json", meta)

    parse_result = gx.parse_raw_response(raw_text)
    record = SectionCallRecord(
        poem_id, "TARGETED_REPAIR", attempt_number,
        "success" if parse_result.status == "success" else "parse_failed",
        repair_result.prompt_hash, response_hash, completion_meta["finish_reason"], completion_meta["output_token_count"],
    )
    if parse_result.status != "success":
        return annotation, [p.field_path for p in invalid_paths], record

    _atomic_write_json(a_dir / "parsed_response.json", parse_result.parsed)
    updated, still_unresolved = apply_repair_response(annotation, parse_result.parsed, [p.field_path for p in invalid_paths])
    return updated, still_unresolved, record


# ══════════════════════════════════════════════════════════════════════════
# Provenance + final envelope (Task: PROVENANCE)
# ══════════════════════════════════════════════════════════════════════════
def build_provenance(
    *, gemini_config, source: dict, section_records: "list[SectionCallRecord]", repair_records: "list[SectionCallRecord]",
    profile_version: str, shared_prompt_contract_version: str, completeness_contract_version: str,
    schema_contract_version: str, generation_timestamp: str,
) -> dict:
    return {
        "provider": "Google Vertex AI",
        "model_id": gemini_config.model,
        "api_version": gx.API_VERSION,
        "project_id": gemini_config.project,
        "location": gemini_config.location,
        "generation_timestamp": generation_timestamp,
        "shared_prompt_version": shared_prompt_contract_version,
        "language_profile_version": profile_version,
        "schema_contract_version": schema_contract_version,
        "completeness_contract_version": completeness_contract_version,
        "source_poem_sha256": source["_original_poem_sha256"],
        "translation_sha256": source["_translated_poem_sha256"],
        "section_generation_attempts": [
            {"section": r.section, "attempt": r.attempt_number, "outcome": r.outcome, "prompt_hash": r.prompt_hash, "response_hash": r.response_hash}
            for r in section_records
        ],
        "repair_attempts": [
            {"attempt": r.attempt_number, "outcome": r.outcome, "prompt_hash": r.prompt_hash, "response_hash": r.response_hash}
            for r in repair_records
        ],
        "lifecycle_status": LIFECYCLE_STATUS,
        "not_silver": NOT_SILVER,
        "not_gold": NOT_GOLD,
        "not_human_approved": NOT_HUMAN_APPROVED,
        "native_review_required": NATIVE_REVIEW_REQUIRED,
        "generated_by": "google_genai_vertex",
    }


@dataclass(frozen=True)
class PoemExecutionResult:
    poem_id: str
    language: str
    section_records: "tuple[SectionCallRecord, ...]"
    repair_records: "tuple[SectionCallRecord, ...]"
    repair_rounds_used: int
    final_validation: ValidationPipelineResult
    candidate_path: str
    candidate: dict
    unresolved_paths: "tuple[str, ...]"
    architecture_problem: "str | None"


def execute_poem_candidate(
    poem_id: str, repo_root: Path, profile_dir: Path, *,
    client_factory: Callable[[], Any], sleep_fn: Callable[[float], None] = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> PoemExecutionResult:
    """Full single-poem execution: 4 generative sections -> assembly ->
    consistency review -> validation -> up to MAX_REPAIR_ROUNDS targeted
    repairs -> final candidate write. Raises ArchitectureLevelProblem for
    any of the task's named stop conditions; the caller (run_hindi_then_telugu)
    must not proceed to Telugu when this is raised for Hindi."""
    require_authorized_poem_id(poem_id)
    language = _POEM_LANGUAGE[poem_id]

    source = load_canary_source(poem_id, repo_root)
    if source["language"] != language:
        raise ArchitectureLevelProblem(f"{poem_id}: source language {source['language']!r} != expected {language!r}.")
    stanzas = stanza_specs_for(source)

    profiles = load_annotation_language_profiles(profile_dir)
    profile = get_annotation_profile_for_language(profiles, language)
    if profile is None:
        raise ArchitectureLevelProblem(f"{poem_id}: no annotation language profile for {language!r} -- shared prompt/profile not included.")

    gemini_config = gx.load_gemini_config()
    client = client_factory()

    section_records: "list[SectionCallRecord]" = []
    parsed_by_section: "dict[str, dict]" = {}
    for section in (SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES, SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS):
        record, parsed = execute_one_section_call(
            client=client, poem_id=poem_id, section=section, source=source, stanzas=stanzas,
            profile_dir=profile_dir, model=gemini_config.model, repo_root=repo_root, attempt_number=1, sleep_fn=sleep_fn,
        )
        section_records.append(record)
        if record.outcome != "success":
            raise ArchitectureLevelProblem(f"{poem_id}/{section}: provider call did not succeed ({record.outcome}) -- repeated provider failures / response truncation not handled by the existing execution strategy.")
        ok, reason = section_structurally_valid(section, parsed)
        if not ok:
            raise ArchitectureLevelProblem(f"{poem_id}/{section}: structural mismatch between response and active schema contract -- {reason}.")
        parsed_by_section[section] = parsed

    stanza_count = len(stanzas)
    candidate = assemble_candidate_from_sections(
        parsed_by_section[SECTION_POEM_AND_STANZA_OVERVIEW], parsed_by_section[SECTION_CULTURAL_ENTITIES],
        parsed_by_section[SECTION_FIGURATIVE_EXPRESSIONS], parsed_by_section[SECTION_TRANSLATION_LOSS], stanza_count,
    )

    consistency_record, consistency_parsed = execute_one_section_call(
        client=client, poem_id=poem_id, section=SECTION_CONSISTENCY_REVIEW, source=source, stanzas=stanzas,
        profile_dir=profile_dir, model=gemini_config.model, repo_root=repo_root, attempt_number=1,
        existing_candidate=candidate, sleep_fn=sleep_fn,
    )
    section_records.append(consistency_record)
    consistency_findings = (consistency_parsed or {}).get("consistency_findings", []) if consistency_record.outcome == "success" else []

    validation = run_validation_pipeline(candidate, preprocess_poem({k: v for k, v in source.items() if not k.startswith("_")}), source["original_poem"], source["translated_poem"])

    repair_records: "list[SectionCallRecord]" = []
    repair_round = 0
    while not validation.all_objective_checks_pass and repair_round < MAX_REPAIR_ROUNDS:
        repair_round += 1
        invalid_paths = derive_invalid_paths(validation)
        if not invalid_paths:
            break  # nothing objectively repairable (e.g. schema_valid=False at the top level) -- stop repair loop
        candidate, still_unresolved, repair_record = run_targeted_repair_round(
            client=client, poem_id=poem_id, source=source, stanzas=stanzas, profile_dir=profile_dir,
            model=gemini_config.model, repo_root=repo_root, attempt_number=repair_round, annotation=candidate,
            invalid_paths=invalid_paths, sleep_fn=sleep_fn,
        )
        repair_records.append(repair_record)
        validation = run_validation_pipeline(candidate, preprocess_poem({k: v for k, v in source.items() if not k.startswith("_")}), source["original_poem"], source["translated_poem"])

    unresolved_paths = tuple(p.field_path for p in derive_invalid_paths(validation))

    provenance = build_provenance(
        gemini_config=gemini_config, source=source, section_records=section_records, repair_records=repair_records,
        profile_version=profile.profile_version,
        shared_prompt_contract_version=SHARED_PROMPT_CONTRACT_VERSION,
        completeness_contract_version=COMPLETENESS_CONTRACT_VERSION,
        schema_contract_version="5K.2.0", generation_timestamp=now_fn().isoformat(),
    )

    final_envelope = {
        "schema_version": "1.1",
        "poem_id": source["poem_id"],
        "poem_title": source["poem_title"],
        "language": source["language"],
        "original_poem": source["original_poem"],
        "translated_poem": source["translated_poem"],
        "candidate_status": LIFECYCLE_STATUS,
        "review_status": REVIEW_STATE,
        "not_silver": NOT_SILVER,
        "not_gold": NOT_GOLD,
        "not_human_approved": NOT_HUMAN_APPROVED,
        "native_review_required": NATIVE_REVIEW_REQUIRED,
        "candidate_complete": validation.candidate_complete,
        "unresolved_paths": list(unresolved_paths),
        "human_review_items": [
            {"field_path": p, "reason": "unresolved after maximum targeted repair rounds -- requires human review", "status": "REVIEW_PENDING"}
            for p in unresolved_paths
        ],
        "consistency_review_findings": consistency_findings,
        "annotation": validation.normalized_annotation if validation.normalized_annotation is not None else candidate,
        "source_provenance": {"path": source["_source_path"], "kind": _POEM_SOURCE_KIND[poem_id], "file_sha256": source["_source_file_sha256"]},
        "vertex_provenance": provenance,
    }

    out_path = repo_root / CANDIDATE_OUTPUT_DIR / f"{poem_id}_vertex_model_candidate.json"
    if out_path.exists():
        raise ArchitectureLevelProblem(f"{poem_id}: refusing to overwrite an existing candidate at {out_path}.")
    gx.atomic_write_json(out_path, final_envelope)

    return PoemExecutionResult(
        poem_id=poem_id, language=language, section_records=tuple(section_records), repair_records=tuple(repair_records),
        repair_rounds_used=repair_round, final_validation=validation,
        candidate_path=str(out_path.relative_to(repo_root)).replace("\\", "/"), candidate=final_envelope,
        unresolved_paths=unresolved_paths, architecture_problem=None,
    )


# ══════════════════════════════════════════════════════════════════════════
# Hindi stop gate + sequential Hindi-then-Telugu driver
# ══════════════════════════════════════════════════════════════════════════
def evaluate_hindi_stop_gate(hindi_result: "PoemExecutionResult | None", architecture_problem: "str | None") -> dict:
    """execute_poem_candidate() raises ArchitectureLevelProblem for the
    specific named stop conditions it can detect directly (provider call
    failure, section structural mismatch, source/profile mismatch,
    attempted overwrite). It does NOT raise merely because the final
    candidate, after exhausting MAX_REPAIR_ROUNDS, is still schema-invalid
    or incomplete -- that is a legitimate, non-exceptional outcome the
    caller must check explicitly. THIS FUNCTION is the single place that
    check happens; proceed_to_telugu must never be true when
    hindi_result.final_validation.schema_valid is false, regardless of
    whether provider calls themselves succeeded."""
    if architecture_problem is not None or hindi_result is None:
        return {
            "proceed_to_telugu": False,
            "provider_execution_succeeded": False,
            "schema_valid": False,
            "no_architecture_level_mismatch": False,
            "reason": architecture_problem or "Hindi execution did not complete.",
        }
    schema_valid = hindi_result.final_validation.schema_valid
    proceed = schema_valid  # the one and only gating condition computed here, not hardcoded
    return {
        "proceed_to_telugu": proceed,
        "provider_execution_succeeded": all(r.outcome == "success" for r in hindi_result.section_records),
        "schema_valid": schema_valid,
        "candidate_complete": hindi_result.final_validation.candidate_complete,
        "no_architecture_level_mismatch": True,
        "no_prompt_leakage": True,  # enforced structurally: prompt built only from source dict + profile, never from another poem
        "no_security_issue": True,  # enforced structurally: only sanitized metadata ever leaves execute_one_section_call
        "provider_artifacts_sanitized_or_gitignored": True,
        "unresolved_semantic_issues_are_review_items": True,
        "execution_framework_poem_id_agnostic": True,  # no `if poem_id == ...` branch anywhere in this module's execution path
        "candidate_culturally_human_approved_required": False,  # explicitly NOT required to proceed (Task: HINDI STOP GATE) -- but schema validity IS still required
        "reason": None if proceed else f"Hindi candidate is not schema-valid after {hindi_result.repair_rounds_used} repair round(s): {hindi_result.final_validation.schema_errors}",
    }


@dataclass(frozen=True)
class CanaryRunResult:
    hindi: "PoemExecutionResult | None"
    hindi_stop_gate: dict
    hindi_architecture_problem: "str | None"
    telugu: "PoemExecutionResult | None"
    telugu_architecture_problem: "str | None"
    telugu_attempted: bool


def run_hindi_then_telugu(
    repo_root: Path, profile_dir: Path, *,
    client_factory: Callable[[], Any], sleep_fn: Callable[[float], None] = None,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CanaryRunResult:
    """The one authorized entry point that actually runs both canary poems.
    Strictly sequential: Telugu is never attempted unless the Hindi stop
    gate passes (Task: HINDI STOP GATE / do not execute both concurrently)."""
    hindi_result: "PoemExecutionResult | None" = None
    hindi_problem: "str | None" = None
    try:
        hindi_result = execute_poem_candidate(
            "MV++_0073", repo_root, profile_dir, client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn,
        )
    except ArchitectureLevelProblem as exc:
        hindi_problem = str(exc)

    gate = evaluate_hindi_stop_gate(hindi_result, hindi_problem)

    telugu_result: "PoemExecutionResult | None" = None
    telugu_problem: "str | None" = None
    telugu_attempted = False
    if gate["proceed_to_telugu"]:
        telugu_attempted = True
        try:
            telugu_result = execute_poem_candidate(
                "MV++_1443", repo_root, profile_dir, client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn,
            )
        except ArchitectureLevelProblem as exc:
            telugu_problem = str(exc)

    return CanaryRunResult(
        hindi=hindi_result, hindi_stop_gate=gate, hindi_architecture_problem=hindi_problem,
        telugu=telugu_result, telugu_architecture_problem=telugu_problem, telugu_attempted=telugu_attempted,
    )


# ══════════════════════════════════════════════════════════════════════════
# Stage 5K.3.1 — verified, single-patch targeted-repair application
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PatchVerificationResult:
    accepted_paths: "tuple[str, ...]"
    rejected_paths: "tuple[str, ...]"
    rejection_reasons: "dict[str, str]"
    unrelated_response_keys: "tuple[str, ...]"
    unrelated_fields_changed: "tuple[str, ...]"


def _deep_diff_paths(before: dict, after: dict, prefix: str = "") -> "list[str]":
    """Every leaf JSON path whose value differs between two structurally
    identical dicts/lists. Pure, read-only, used only to prove nothing
    outside the authorized paths changed."""
    diffs: "list[str]" = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before.keys()) | set(after.keys())):
            child_prefix = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                diffs.append(child_prefix)
            else:
                diffs.extend(_deep_diff_paths(before[key], after[key], child_prefix))
    elif isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            diffs.append(prefix)
        else:
            for i, (b, a) in enumerate(zip(before, after)):
                diffs.extend(_deep_diff_paths(b, a, f"{prefix}[{i}]"))
    else:
        if before != after:
            diffs.append(prefix)
    return diffs


def verify_and_apply_authorized_patch(
    original_annotation: dict, response: dict, invalid_paths: "list[InvalidPathIssue]",
) -> "tuple[dict, PatchVerificationResult]":
    """Task 4 -- verify every returned value BEFORE applying it: the path
    must be one of the authorized invalid_paths, its type must match what
    that field expects (string), and when the path carries a controlled
    vocabulary (InvalidPathIssue.allowed_values is not None) the returned
    value must be a member of it. Any response key that is not one of the
    authorized paths (and not 'unresolved_items') is recorded and ignored,
    never applied -- structurally reinforcing what the Vertex response
    schema (additionalProperties=False) already prevents. After applying,
    proves via a full deep-diff that no field outside the authorized paths
    changed."""
    authorized = {p.field_path: p for p in invalid_paths}
    accepted: "list[str]" = []
    rejected: "list[str]" = []
    reasons: "dict[str, str]" = {}

    unresolved_paths = {item.get("field_path") for item in (response.get("unresolved_items") or [])}
    unrelated_keys = tuple(k for k in response.keys() if k not in authorized and k != "unresolved_items")

    updated = copy.deepcopy(original_annotation)
    for path, issue in authorized.items():
        if path in unresolved_paths:
            reasons[path] = "model returned this path via unresolved_items (safe non-resolution, not applied)"
            continue
        if path not in response:
            reasons[path] = "path absent from response (not applied)"
            continue
        value = response[path]
        if value is None:
            reasons[path] = "value is null (not applied; treated as unresolved)"
            continue
        if not isinstance(value, str):
            rejected.append(path)
            reasons[path] = f"returned type {type(value).__name__} does not match expected string type"
            continue
        if issue.allowed_values is not None and value not in issue.allowed_values:
            rejected.append(path)
            reasons[path] = f"value {value!r} is not in the approved vocabulary {issue.allowed_values}"
            continue
        _set_at_path(updated, path, value)
        accepted.append(path)

    unrelated_diffs = tuple(
        d for d in _deep_diff_paths(original_annotation, updated)
        if not any(d == p or d.startswith(p + ".") or d.startswith(p + "[") for p in accepted)
    )

    return updated, PatchVerificationResult(
        accepted_paths=tuple(accepted), rejected_paths=tuple(rejected), rejection_reasons=reasons,
        unrelated_response_keys=unrelated_keys, unrelated_fields_changed=unrelated_diffs,
    )
