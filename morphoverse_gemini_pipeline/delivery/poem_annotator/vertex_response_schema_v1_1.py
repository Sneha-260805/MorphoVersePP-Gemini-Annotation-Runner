"""Stage 5K.3 — Vertex-AI-compatible response JSON schemas for the Stage
5K.1/5K.2 section-specific prompt architecture.

`prompt_assembler_v1_1.response_schema_for_section()` produces a lightweight,
human-readable schema description for prompt PREVIEWS (Stage 5K.1/5K.2,
offline only). This module builds the STRICTER, Vertex-`response_json_schema`-
compatible variant actually sent on a live request: every property has an
explicit `type`/`anyOf`/`enum` (no bare `{}`), and nullable fields use the
same `{"anyOf": [<shape>, {"type": "null"}]}` pattern already proven correct
by `gemini_backfill_executor_v1_1.build_patch_response_json_schema` (the
Stage 5D.1 fix for a real, previously-encountered Vertex 400 rejection of an
untyped required property). No schema field name here is invented — every
leaf shape corresponds exactly to a field already frozen in
`shared_full_schema_prompt_v1_1.py`'s FIELD_DEFINITIONS dicts and
`pilot/reports/schema_freeze_stage5k2/active_schema_contract_v1_1.json`.

Pure, offline: builds a schema; never calls the provider. Reuses
`_audit_schema_node`-equivalent self-checking so a malformed schema is
caught locally before ever being sent.
"""
from __future__ import annotations

from typing import Any

from .prompt_assembler_v1_1 import (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
from .schema import (
    ALLOWED_RECITATION_STYLES, ALLOWED_EMOTIONS, ALLOWED_TONES, ALLOWED_TRANSLATION_QUALITIES,
    ALLOWED_ENTITY_CATEGORIES, ALLOWED_VISUAL_PRIORITIES, ALLOWED_EXPRESSION_TYPES,
)
from .shared_full_schema_prompt_v1_1 import (
    PROPOSED_CULTURAL_SPECIFICITY_LEVELS, PROPOSED_TRANSLATION_STATUSES, PROPOSED_VISUALIZATION_DIFFICULTIES,
)

_TRANSLATION_LOSS_SEVERITY = ("low", "medium", "high")  # frozen in controlled_vocabularies_v1_1.json

_STRING: dict = {"type": "string"}
_NULL: dict = {"type": "null"}
_LINE_REF_PATTERN = r"^L\d+(-L\d+)?$"


def _nullable(shape: dict) -> dict:
    return {"anyOf": [shape, _NULL]}


def _string_array() -> dict:
    return {"type": "array", "items": _STRING}


def _enum(values: "tuple[str, ...]") -> dict:
    return {"type": "string", "enum": list(values)}


_UNRESOLVED_ITEM_SHAPE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["field_path", "reason", "candidate_value"],
    "properties": {
        "field_path": _STRING,
        "reason": _STRING,
        "candidate_value": _nullable(_STRING),
    },
}
_UNRESOLVED_ITEMS_ARRAY_SHAPE = {"type": "array", "items": _UNRESOLVED_ITEM_SHAPE}

_METAPHOR_MAPPING_SHAPE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["vehicle_concept", "tenor_concept", "transferred_attributes"],
    "properties": {
        "vehicle_concept": _STRING,
        "tenor_concept": _STRING,
        "transferred_attributes": _string_array(),
    },
}

_LINE_REF_SHAPE = {"type": "string", "pattern": _LINE_REF_PATTERN}

_CULTURAL_ENTITY_PROPERTIES = {
    "term": _STRING,
    "romanization": _STRING,
    "category": _enum(ALLOWED_ENTITY_CATEGORIES),
    "stanza_index": {"type": "integer"},
    "preserved": _nullable({"type": "boolean"}),
    "translation_note": _STRING,
    "gloss": _nullable(_STRING),
    "line_ref": _nullable(_LINE_REF_SHAPE),
    "source_span_original": _nullable(_STRING),
    "source_span_translation": _nullable(_STRING),
    "visual_features": _string_array(),
    "visual_priority": _nullable(_enum(ALLOWED_VISUAL_PRIORITIES)),
    "acceptable_visual_variants": _string_array(),
    "negative_confusions": _string_array(),
    "translation_status": _nullable(_enum(PROPOSED_TRANSLATION_STATUSES)),
    "cultural_specificity_level": _nullable(_enum(PROPOSED_CULTURAL_SPECIFICITY_LEVELS)),
}
_CULTURAL_ENTITY_REQUIRED = list(_CULTURAL_ENTITY_PROPERTIES.keys())
_CULTURAL_ENTITY_SHAPE = {
    "type": "object", "additionalProperties": False,
    "required": _CULTURAL_ENTITY_REQUIRED, "properties": _CULTURAL_ENTITY_PROPERTIES,
}

_FIGURATIVE_EXPRESSION_PROPERTIES = {
    "source_term": _STRING,
    "abstract_meaning": _STRING,
    "source_span_original": _nullable(_STRING),
    "source_span_translation": _nullable(_STRING),
    "expression_type": _nullable(_enum(ALLOWED_EXPRESSION_TYPES)),
    "literal_meaning": _nullable(_STRING),
    "vehicle": _nullable(_STRING),
    "tenor": _nullable(_STRING),
    "metaphor_mapping": _nullable(_METAPHOR_MAPPING_SHAPE),
    "line_ref": _nullable(_LINE_REF_SHAPE),
    "literalization_risk": _nullable(_STRING),
    "visualization_strategy": _nullable(_STRING),
    "acceptable_visual_variants": _string_array(),
    "visualization_difficulty": _nullable(_enum(PROPOSED_VISUALIZATION_DIFFICULTIES)),
}
_FIGURATIVE_EXPRESSION_REQUIRED = list(_FIGURATIVE_EXPRESSION_PROPERTIES.keys())
_FIGURATIVE_EXPRESSION_SHAPE = {
    "type": "object", "additionalProperties": False,
    "required": _FIGURATIVE_EXPRESSION_REQUIRED, "properties": _FIGURATIVE_EXPRESSION_PROPERTIES,
}

_TRANSLATION_LOSS_ITEM_PROPERTIES = {
    "what_was_lost": _STRING,
    "where": _nullable(_LINE_REF_SHAPE),
    "severity": _nullable(_enum(_TRANSLATION_LOSS_SEVERITY)),
}
_TRANSLATION_LOSS_ITEM_SHAPE = {
    "type": "object", "additionalProperties": False,
    "required": list(_TRANSLATION_LOSS_ITEM_PROPERTIES.keys()), "properties": _TRANSLATION_LOSS_ITEM_PROPERTIES,
}

_STANZA_OVERVIEW_PROPERTIES = {
    "index": {"type": "integer"},
    "emotion": _enum(ALLOWED_EMOTIONS),
    "tone": _enum(ALLOWED_TONES),
    "translation_quality": _enum(ALLOWED_TRANSLATION_QUALITIES),
    "loss_note": _STRING,
}
_STANZA_OVERVIEW_SHAPE = {
    "type": "object", "additionalProperties": False,
    "required": list(_STANZA_OVERVIEW_PROPERTIES.keys()), "properties": _STANZA_OVERVIEW_PROPERTIES,
}


def build_vertex_response_schema(section: str) -> dict:
    """The strict, Vertex-request-ready response_json_schema for one
    section. Deterministic: same section -> same schema, always."""
    if section == SECTION_POEM_AND_STANZA_OVERVIEW:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["recitation_style", "emotional_arc", "theme", "stanzas", "unresolved_items"],
            "properties": {
                "recitation_style": _enum(ALLOWED_RECITATION_STYLES),
                "emotional_arc": _STRING,
                "theme": _nullable(_STRING),
                "stanzas": {"type": "array", "items": _STANZA_OVERVIEW_SHAPE},
                "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE,
            },
        }
    if section == SECTION_CULTURAL_ENTITIES:
        return {
            "type": "object", "additionalProperties": False,
            "required": ["cultural_entities", "unresolved_items"],
            "properties": {
                "cultural_entities": {"type": "array", "items": _CULTURAL_ENTITY_SHAPE},
                "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE,
            },
        }
    if section == SECTION_FIGURATIVE_EXPRESSIONS:
        stanza_expr_shape = {
            "type": "object", "additionalProperties": False,
            "required": ["index", "metaphor_spans"],
            "properties": {"index": {"type": "integer"}, "metaphor_spans": {"type": "array", "items": _FIGURATIVE_EXPRESSION_SHAPE}},
        }
        return {
            "type": "object", "additionalProperties": False,
            "required": ["stanzas", "unresolved_items"],
            "properties": {"stanzas": {"type": "array", "items": stanza_expr_shape}, "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE},
        }
    if section == SECTION_TRANSLATION_LOSS:
        stanza_loss_shape = {
            "type": "object", "additionalProperties": False,
            "required": ["index", "translation_loss"],
            "properties": {"index": {"type": "integer"}, "translation_loss": {"type": "array", "items": _TRANSLATION_LOSS_ITEM_SHAPE}},
        }
        return {
            "type": "object", "additionalProperties": False,
            "required": ["stanzas", "unresolved_items"],
            "properties": {"stanzas": {"type": "array", "items": stanza_loss_shape}, "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE},
        }
    if section == SECTION_CONSISTENCY_REVIEW:
        finding_shape = {
            "type": "object", "additionalProperties": False,
            "required": ["field_path", "issue", "severity"],
            "properties": {"field_path": _STRING, "issue": _STRING, "severity": _nullable(_STRING)},
        }
        return {
            "type": "object", "additionalProperties": False,
            "required": ["consistency_findings", "unresolved_items"],
            "properties": {"consistency_findings": {"type": "array", "items": finding_shape}, "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE},
        }
    raise ValueError(f"No Vertex response schema defined for section {section!r}.")


def audit_schema_node(node: Any, *, node_path: str = "$", errors: "list[str] | None" = None) -> "list[str]":
    """Generic, offline, recursive self-audit -- every node must declare
    type/anyOf/enum, no empty {} schema, no callables. Mirrors
    gemini_backfill_executor_v1_1._audit_schema_node's generic checks
    (that function's outer audit_provider_response_schema is patch-specific
    and not reused here; this generic inner-check logic is re-implemented
    rather than importing a private name across modules)."""
    if errors is None:
        errors = []
    if not isinstance(node, dict):
        return errors
    if node == {}:
        errors.append(f"{node_path}: empty schema dict is not permitted.")
        return errors
    for key, value in node.items():
        if callable(value):
            errors.append(f"{node_path}.{key}: schema must not contain a callable/executable value.")
    if not ("type" in node or "anyOf" in node or "enum" in node):
        errors.append(f"{node_path}: property has no type, anyOf, or enum.")
    if "anyOf" in node:
        if not isinstance(node["anyOf"], list) or not node["anyOf"]:
            errors.append(f"{node_path}.anyOf: must be a non-empty list.")
        else:
            for i, branch in enumerate(node["anyOf"]):
                audit_schema_node(branch, node_path=f"{node_path}.anyOf[{i}]", errors=errors)
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        for prop_name, prop_schema in node["properties"].items():
            audit_schema_node(prop_schema, node_path=f"{node_path}.properties.{prop_name}", errors=errors)
    if node.get("type") == "array" and "items" in node:
        audit_schema_node(node["items"], node_path=f"{node_path}.items", errors=errors)
    return errors


# ── Targeted repair (Task 15) — one leaf-name -> Vertex shape lookup,
# covering both single-value leaves and whole-array/whole-object container
# paths a repair might target (e.g. a detection-completeness path like
# "stanzas[1].metaphor_spans" requests the WHOLE array, not one scalar).
_REPAIR_LEAF_SHAPES: "dict[str, dict]" = {
    **{k: v for k, v in _CULTURAL_ENTITY_PROPERTIES.items()},
    **{k: v for k, v in _FIGURATIVE_EXPRESSION_PROPERTIES.items()},
    **{k: v for k, v in _STANZA_OVERVIEW_PROPERTIES.items()},
    "recitation_style": _enum(ALLOWED_RECITATION_STYLES),
    "emotional_arc": _STRING,
    "theme": _nullable(_STRING),
    "what_was_lost": _STRING,
    "where": _nullable(_LINE_REF_SHAPE),
    "severity": _nullable(_enum(_TRANSLATION_LOSS_SEVERITY)),
    # Whole-array/whole-object container targets (detection-completeness paths).
    "cultural_entities": {"type": "array", "items": _CULTURAL_ENTITY_SHAPE},
    "metaphor_spans": {"type": "array", "items": _FIGURATIVE_EXPRESSION_SHAPE},
    "translation_loss": {"type": "array", "items": _TRANSLATION_LOSS_ITEM_SHAPE},
    "metaphor_mapping": _nullable(_METAPHOR_MAPPING_SHAPE),
}


def _repair_leaf_name(path: str) -> str:
    """'cultural_entities[0].gloss' -> 'gloss'; 'stanzas[1].metaphor_spans'
    -> 'metaphor_spans'; a bare 'theme' -> 'theme'. Pure string parsing, no
    poem/path-instance-specific logic."""
    tail = path.rstrip()
    if "." in tail:
        tail = tail.rsplit(".", 1)[-1]
    if "[" in tail:
        tail = tail.split("[", 1)[0]
    return tail


def build_vertex_repair_response_schema(paths: "list[str] | tuple[str, ...]") -> dict:
    """Vertex-compatible response schema for a targeted-repair request over
    exactly `paths`. Each path's value is `anyOf[<leaf shape>, null]` (null
    meaning: resolved as explicitly not applicable / left to
    unresolved_items) -- the model may instead satisfy any path solely via
    an `unresolved_items` entry, so no path's value is ever forced non-null
    when safe resolution is not possible. Raises ValueError for a path whose
    leaf name has no known shape (fails loudly rather than silently
    accepting an untyped, Vertex-incompatible property)."""
    if not paths:
        raise ValueError("build_vertex_repair_response_schema requires at least one path.")
    properties: "dict[str, dict]" = {}
    for path in paths:
        leaf = _repair_leaf_name(path)
        if leaf not in _REPAIR_LEAF_SHAPES:
            raise ValueError(f"No Vertex repair response shape is defined for leaf field {leaf!r} (path {path!r}).")
        shape = _REPAIR_LEAF_SHAPES[leaf]
        properties[path] = shape if shape.get("anyOf") else _nullable(shape)
    return {
        "type": "object", "additionalProperties": False,
        "required": list(properties.keys()) + ["unresolved_items"],
        "properties": {**properties, "unresolved_items": _UNRESOLVED_ITEMS_ARRAY_SHAPE},
    }


def build_and_audit_vertex_repair_response_schema(paths: "list[str] | tuple[str, ...]") -> dict:
    schema = build_vertex_repair_response_schema(paths)
    errors = audit_schema_node(schema)
    if errors:
        raise ValueError(f"Vertex repair response schema failed self-audit: {errors}")
    return schema


def build_and_audit_vertex_response_schema(section: str) -> dict:
    """build_vertex_response_schema(), then self-audit before ever sending
    a request built from it. Raises ValueError with every problem found if
    the schema is malformed -- never sends a schema this audit rejects."""
    schema = build_vertex_response_schema(section)
    errors = audit_schema_node(schema)
    if errors:
        raise ValueError(f"Vertex response schema for section {section!r} failed self-audit: {errors}")
    return schema
