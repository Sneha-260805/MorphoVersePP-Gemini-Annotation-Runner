"""Stage 5K.2 (Task 5) — an executable checker for
pilot/reports/schema_freeze_stage5k2/completeness_contract_v1_1.json.

This module implements the FROZEN completeness contract as callable checks,
so "candidate-complete" is a testable property, not only a JSON
specification. It is strictly ADDITIVE on top of schema.py's structural
validation (models.py) -- it never replaces or relaxes schema validation,
and it introduces no new schema field. A payload can be schema-valid while
still failing completeness (e.g. an empty romanization is schema-valid but
completeness-invalid — see completeness_contract_v1_1.json mismatch M01).

Pure, offline: takes an already-validated annotation dict (e.g. the output
of models.validate_model_payload_v1_1) and returns structured violations.
Never calls a model, network, or provider.
"""
from __future__ import annotations

from dataclasses import dataclass

# The one figurative expression_type documented in
# completeness_contract_v1_1.json as an example of a non-visual expression
# (visualization_difficulty not applicable). Not an exhaustive list by
# design -- a purely phonetic/aural device has no visual dimension to rate.
_NON_VISUAL_EXPRESSION_TYPES = frozenset({"wordplay"})


@dataclass(frozen=True)
class CompletenessViolation:
    field_path: str
    rule: str
    message: str


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _term_is_non_latin(term: str) -> bool:
    """True if `term` contains at least one alphabetic character outside
    the ASCII Latin range -- i.e. romanization is APPLICABLE for it."""
    return any(ch.isalpha() and ord(ch) > 0x7A for ch in term)


def check_cultural_entity_completeness(entity: dict, index: int) -> "list[CompletenessViolation]":
    violations: "list[CompletenessViolation]" = []
    path = f"cultural_entities[{index}]"

    term = entity.get("term", "")
    if _term_is_non_latin(term) and _is_blank(entity.get("romanization")):
        violations.append(CompletenessViolation(
            f"{path}.romanization", "no_missing_romanization_for_applicable_non_latin_term",
            "romanization is empty/whitespace-only for a non-Latin-script term.",
        ))

    if entity.get("cultural_specificity_level") is None:
        violations.append(CompletenessViolation(
            f"{path}.cultural_specificity_level", "cultural_specificity_for_every_retained_entity",
            "cultural_specificity_level is null for a retained cultural entity.",
        ))

    if entity.get("translation_status") is None:
        violations.append(CompletenessViolation(
            f"{path}.translation_status", "translation_status_for_every_retained_entity",
            "translation_status is null for a retained cultural entity.",
        ))

    if entity.get("line_ref") is None:
        violations.append(CompletenessViolation(
            f"{path}.line_ref", "valid_global_line_references",
            "line_ref is null for a retained cultural entity.",
        ))

    if entity.get("source_span_original") is None:
        violations.append(CompletenessViolation(
            f"{path}.source_span_original", "exact_source_spans",
            "source_span_original is null for a retained cultural entity.",
        ))

    status = entity.get("translation_status")
    if status in {"PARTIALLY_PRESERVED", "ALTERED", "LOST", "NOT_TRANSLATED"} and _is_blank(entity.get("translation_note")):
        violations.append(CompletenessViolation(
            f"{path}.translation_note", "translation_note_for_partial_altered_lost_or_untranslated",
            f"translation_note is empty while translation_status is {status!r}.",
        ))

    visual_priority = entity.get("visual_priority")
    if visual_priority is not None and visual_priority != "non_visual":
        if not entity.get("visual_features"):
            violations.append(CompletenessViolation(
                f"{path}.visual_features", "visual_arrays_required_unless_non_visual",
                f"visual_features is empty while visual_priority is {visual_priority!r}.",
            ))

    return violations


def check_figurative_expression_completeness(expr: dict, stanza_index: int, index: int) -> "list[CompletenessViolation]":
    violations: "list[CompletenessViolation]" = []
    path = f"stanzas[{stanza_index}].metaphor_spans[{index}]"

    # Stage 5M.4F: metaphor_mapping is NOT required merely because vehicle
    # and tenor are both populated -- that unconditional pairing was the
    # completeness rule's own overreach relative to models.py's own
    # contract (validate_metaphor_mapping_v1_1: "metaphor_mapping is
    # optional/nullable at the containing figurative expression (not every
    # expression type needs one)") and shared_full_schema_prompt_v1_1.py's
    # field definition ("included only when both concepts are clearly
    # evidenced, never invented for every expression"). A structured
    # vehicle_concept/tenor_concept/transferred_attributes elaboration is a
    # STRONGER, separately-evidenced claim than the legacy free-text
    # vehicle/tenor fields, and forcing it whenever those are non-blank
    # required the model to either invent unsupported structure or be
    # marked incomplete for correctly declining to (Stage 5M.4E audit:
    # two real, non-"metaphor" expression_type candidates where the repair
    # model explicitly declined via unresolved_items rather than invent
    # one -- exactly the behavior UNCERTAINTY_HANDLING_TEXT asks for; this
    # rule is corpus-generic and applies identically regardless of poem,
    # language, or expression_type). metaphor_mapping's own INTERNAL
    # shape, when present, is still fully validated -- by models.py's
    # validate_metaphor_mapping_v1_1, never duplicated here.
    vehicle_blank = _is_blank(expr.get("vehicle"))
    tenor_blank = _is_blank(expr.get("tenor"))
    if vehicle_blank:
        violations.append(CompletenessViolation(f"{path}.vehicle", "vehicle_tenor_mapping_for_metaphors", "vehicle is null/empty."))
    if tenor_blank:
        violations.append(CompletenessViolation(f"{path}.tenor", "vehicle_tenor_mapping_for_metaphors", "tenor is null/empty."))

    expr_type = expr.get("expression_type")
    if expr_type not in _NON_VISUAL_EXPRESSION_TYPES and expr.get("visualization_difficulty") is None:
        violations.append(CompletenessViolation(
            f"{path}.visualization_difficulty", "visualization_difficulty_for_every_applicable_expression",
            f"visualization_difficulty is null for an applicable expression_type ({expr_type!r}).",
        ))

    if expr.get("line_ref") is None:
        violations.append(CompletenessViolation(f"{path}.line_ref", "valid_global_line_references", "line_ref is null."))
    if expr.get("source_span_original") is None:
        violations.append(CompletenessViolation(f"{path}.source_span_original", "exact_source_spans", "source_span_original is null."))

    return violations


def check_stanza_completeness(stanza: dict, position: int) -> "list[CompletenessViolation]":
    violations: "list[CompletenessViolation]" = []
    index = position - 1
    translation_quality = stanza.get("translation_quality")
    loss_note = stanza.get("loss_note", "")
    if translation_quality != "faithful" and _is_blank(loss_note):
        violations.append(CompletenessViolation(
            f"stanzas[{index}].loss_note", "loss_note_required_when_not_faithful",
            f"loss_note is empty while translation_quality is {translation_quality!r}.",
        ))
    for m_index, expr in enumerate(stanza.get("metaphor_spans", []) or []):
        violations.extend(check_figurative_expression_completeness(expr, index, m_index))
    return violations


def check_candidate_completeness(annotation: dict) -> "list[CompletenessViolation]":
    """Full candidate-complete check over one already schema-valid
    'annotation' object (models.validate_model_payload_v1_1's return
    value). [] cultural_entities / [] metaphor_spans / [] translation_loss
    are NEVER flagged -- they are legitimate per completeness_contract_v1_1.json."""
    violations: "list[CompletenessViolation]" = []
    for index, entity in enumerate(annotation.get("cultural_entities", []) or []):
        violations.extend(check_cultural_entity_completeness(entity, index))
    for position, stanza in enumerate(annotation.get("stanzas", []) or [], start=1):
        violations.extend(check_stanza_completeness(stanza, position))
    return violations


def is_candidate_complete(annotation: dict) -> bool:
    return not check_candidate_completeness(annotation)
