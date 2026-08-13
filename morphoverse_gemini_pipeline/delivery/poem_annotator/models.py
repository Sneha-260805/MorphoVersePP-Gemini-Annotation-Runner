from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from api import LLMProxyClient, LLMProxyError
from .config import REQUEST_TEMPERATURE, max_tokens_for, GEMINI_PRIMARY, GEMINI_FALLBACK
from .dataset import PreprocessedPoem
from .schema import (
    ALLOWED_RECITATION_STYLES,
    ALLOWED_EMOTIONS,
    ALLOWED_TONES,
    ALLOWED_TRANSLATION_QUALITIES,
    ALLOWED_ENTITY_CATEGORIES,
    TOPLEVEL_KEYS,
    STANZA_KEYS,
    METAPHOR_KEYS,
    ENTITY_KEYS,
    # Schema v1.1 (Stage 2) — additive; legacy names above are unchanged.
    ALLOWED_VISUAL_PRIORITIES,
    ALLOWED_EXPRESSION_TYPES,
    TOPLEVEL_KEYS_V1_1,
    STANZA_KEYS_V1_1,
    METAPHOR_KEYS_V1_1,
    ENTITY_KEYS_V1_1,
    TRANSLATION_LOSS_KEYS,
    METAPHOR_MAPPING_KEYS,
    MORPHOVERSE_SCHEMA_VERSION,
)
# Schema v1.1 (Stage 3) — only the pure-syntax line_ref parser is used here.
# Textual grounding itself (resolving a span against real poem text) stays
# out of models.py entirely; see grounding.py and docs/GROUNDING_AND_LINE_REFERENCES.md.
from .grounding import parse_line_ref, LineRefError


class ModelValidationError(ValueError):
    """Raised when a model response has the wrong schema."""


class StanzaCountMismatch(ModelValidationError):
    """Raised when a model response changes stanza segmentation."""


# ── Abbreviated key expansion ────────────────────────────────────────────────
def expand_abbreviated_keys(payload: Any) -> Any:
    """Expand abbreviated model output keys to full names before schema validation.

    The prompt asks the model to output compact keys (rs/ea/st/ce/i/em/to/tq/ln/ms
    etc.) to fit within the proxy's ~37-token completion cap.  This function
    expands them back to full field names so the rest of the pipeline is unchanged.
    Full-key dicts pass through unchanged (idempotent).
    """
    from .schema import ABBREV_TOPLEVEL, ABBREV_STANZA, ABBREV_METAPHOR, ABBREV_ENTITY
    if not isinstance(payload, dict):
        return payload
    expanded = {ABBREV_TOPLEVEL.get(k, k): v for k, v in payload.items()}
    stanzas = expanded.get("stanzas")
    if isinstance(stanzas, list):
        new_stanzas = []
        for st in stanzas:
            if not isinstance(st, dict):
                new_stanzas.append(st)
                continue
            exp_st = {ABBREV_STANZA.get(k, k): v for k, v in st.items()}
            spans = exp_st.get("metaphor_spans")
            if isinstance(spans, list):
                exp_st["metaphor_spans"] = [
                    {ABBREV_METAPHOR.get(k, k): v for k, v in m.items()}
                    if isinstance(m, dict) else m
                    for m in spans
                ]
            new_stanzas.append(exp_st)
        expanded["stanzas"] = new_stanzas
    entities = expanded.get("cultural_entities")
    if isinstance(entities, list):
        expanded["cultural_entities"] = [
            {ABBREV_ENTITY.get(k, k): v for k, v in e.items()}
            if isinstance(e, dict) else e
            for e in entities
        ]
    return expanded


# ── JSON extraction ──────────────────────────────────────────────────────────
def _repair_truncated_json(s: str) -> Any:
    """Best-effort repair of truncated JSON from proxy completion cutoff.

    Strategy (least to most invasive):
    1. Try a set of standard close-bracket suffixes.
    2. Trim up to 2000 chars from the right and retry suffixes.
    3. If cultural_entities is present but truncated, replace it with [].
    4. If stanzas is the truncated part, close it with format-aware suffixes.
    """
    if not s.startswith("{"):
        raise json.JSONDecodeError("not a JSON object", s, 0)

    generic_closings = [
        "",
        "}",
        "]}",
        "}]}",
        '"]}',
        '"}]}',
        '"}}]}',
    ]
    ce_closings = [
        '],"ce":[]}',
        '],"cultural_entities":[]}',
        '}],"ce":[]}',
        '}],"cultural_entities":[]}',
        '"}],"ce":[]}',
        '"}],"cultural_entities":[]}',
        '""}],"ce":[]}',
        '","ms":[]}],"ce":[]}',
        '[],"ms":[]}],"ce":[]}',
    ]
    all_closings = generic_closings + ce_closings

    for suffix in all_closings:
        try:
            result = json.loads(s + suffix)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Trim from the right (up to 2000 chars) until JSON closes cleanly.
    for trim in range(1, min(2000, len(s))):
        candidate = s[:-trim]
        for suffix in all_closings[1:]:
            try:
                result = json.loads(candidate + suffix)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    # Structural recovery: if cultural_entities is truncated, replace with [].
    # Handles: {"recitation_style":...,"stanzas":[...],"cultural_entities":[{...cut
    ce_markers = ['"cultural_entities":', '"ce":']
    for marker in ce_markers:
        idx = s.rfind(marker)
        if idx != -1:
            truncated_at_ce = s[:idx] + '"cultural_entities":[]}'
            try:
                result = json.loads(truncated_at_ce)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("repair exhausted", s, 0)


def extract_json_payload(raw_text: str) -> Any:
    stripped = (raw_text or "").strip()
    if not stripped:
        raise json.JSONDecodeError("empty model response", "", 0)
    # Strip a leading/trailing code fence anywhere in the text (not only at the ends).
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        stripped = fence.group(1).strip() or stripped
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Last resort: try to repair truncated JSON (proxy completion cutoff)
        candidate = stripped[start:] if start != -1 else stripped
        return _repair_truncated_json(candidate)


# ── Primitive validators ─────────────────────────────────────────────────────
def require_string(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ModelValidationError(f"{field_name} must be a string")
    if not allow_empty and not value.strip():
        raise ModelValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_enum(value: Any, field_name: str, allowed: tuple[str, ...]) -> str:
    normalized = require_string(value, field_name)
    allowed_lookup = {item.casefold(): item for item in allowed}
    key = normalized.casefold()
    if key not in allowed_lookup:
        raise ModelValidationError(f"{field_name} must be one of {list(allowed)}")
    return allowed_lookup[key]


def ensure_only_keys(obj: dict[str, Any], allowed_keys: frozenset[str] | set[str], field_name: str) -> None:
    extra_keys = sorted(set(obj) - set(allowed_keys))
    if extra_keys:
        raise ModelValidationError(f"{field_name} contains unexpected keys: {extra_keys}")


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip()).casefold()


def coerce_to_text(value: Any) -> str:
    """Free-text fields (e.g. emotional_arc) may come back as a list or null.
    Normalize to a string instead of rejecting the whole annotation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return " → ".join(parts)
    return str(value).strip()


def _strip_ascii_gloss(term: str) -> str:
    """Remove trailing English gloss the model sometimes appends, e.g. ' (cold wind)'.
    Strips only ASCII-only parentheticals so Indic script in parens is untouched."""
    return re.sub(r"\s*\([A-Za-z0-9\s,\-']+\)\s*$", "", term).strip()


def term_in_source(term: str, source_text: str) -> bool:
    """Verbatim, whitespace-insensitive presence check (handles compounds/scripts).

    Also tries stripping trailing ASCII gloss the model sometimes appends to
    source_term (e.g. 'చల్లగా వీచే గాలి (cold wind)' → 'చల్లగా వీచే గాలి').
    """
    t = normalize_term(term)
    s = normalize_term(source_text)
    if t and t in s:
        return True
    if t and t.replace(" ", "") in s.replace(" ", ""):
        return True
    # Try after stripping any ASCII annotation gloss
    t2 = normalize_term(_strip_ascii_gloss(term))
    if t2 and t2 != t:
        if t2 in s:
            return True
        if t2.replace(" ", "") in s.replace(" ", ""):
            return True
    return False


# ── Metaphor span validation (no visual_motifs) ──────────────────────────────
def validate_metaphor_spans(value: Any, field_name: str) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ModelValidationError(f"{field_name} must be an array")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ModelValidationError(f"{field_name}[{index}] must be an object")
        ensure_only_keys(item, METAPHOR_KEYS, f"{field_name}[{index}]")
        source_term = require_string(item.get("source_term"), f"{field_name}[{index}].source_term")
        abstract_meaning = require_string(item.get("abstract_meaning"), f"{field_name}[{index}].abstract_meaning")
        normalized.append({"source_term": source_term, "abstract_meaning": abstract_meaning})
    return normalized


# ── Full schema validation ───────────────────────────────────────────────────
def validate_model_payload(payload: Any, poem: PreprocessedPoem) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ModelValidationError("Model output must be a JSON object.")
    ensure_only_keys(payload, TOPLEVEL_KEYS, "payload")

    recitation_style = require_enum(payload.get("recitation_style"), "recitation_style", ALLOWED_RECITATION_STYLES)
    emotional_arc = coerce_to_text(payload.get("emotional_arc"))

    stanzas = payload.get("stanzas")
    if not isinstance(stanzas, list):
        raise ModelValidationError("stanzas must be a JSON array.")
    if len(stanzas) != len(poem.stanzas):
        raise StanzaCountMismatch(f"Expected {len(poem.stanzas)} stanzas but model returned {len(stanzas)}.")

    normalized_stanzas: list[dict[str, Any]] = []
    for i, stanza in enumerate(stanzas, start=1):
        if not isinstance(stanza, dict):
            raise ModelValidationError(f"stanzas[{i-1}] must be an object.")
        ensure_only_keys(stanza, STANZA_KEYS, f"stanzas[{i-1}]")
        if stanza.get("index") != i:
            raise ModelValidationError(f"stanzas[{i-1}].index must be {i}")
        translation_quality = require_enum(
            stanza.get("translation_quality"), f"stanzas[{i-1}].translation_quality", ALLOWED_TRANSLATION_QUALITIES,
        )
        loss_note = require_string(stanza.get("loss_note"), f"stanzas[{i-1}].loss_note", allow_empty=True)
        if translation_quality == "faithful" and loss_note:
            raise ModelValidationError(f"stanzas[{i-1}].loss_note must be empty when translation_quality is faithful")
        metaphor_spans = validate_metaphor_spans(stanza.get("metaphor_spans", []), f"stanzas[{i-1}].metaphor_spans")
        normalized_stanzas.append({
            "index": i,
            "emotion": require_enum(stanza.get("emotion"), f"stanzas[{i-1}].emotion", ALLOWED_EMOTIONS),
            "tone": require_enum(stanza.get("tone"), f"stanzas[{i-1}].tone", ALLOWED_TONES),
            "translation_quality": translation_quality,
            "loss_note": loss_note,
            "metaphor_spans": metaphor_spans,
        })

    cultural_entities = payload.get("cultural_entities", [])
    if not isinstance(cultural_entities, list):
        raise ModelValidationError("cultural_entities must be a JSON array.")

    normalized_entities: list[dict[str, Any]] = []
    for j, entity in enumerate(cultural_entities):
        if not isinstance(entity, dict):
            raise ModelValidationError(f"cultural_entities[{j}] must be an object.")
        ensure_only_keys(entity, ENTITY_KEYS, f"cultural_entities[{j}]")
        stanza_index_raw = entity.get("stanza_index", 1)
        try:
            stanza_index = int(stanza_index_raw)
        except (TypeError, ValueError):
            raise ModelValidationError(f"cultural_entities[{j}].stanza_index must be an integer.") from None
        stanza_index = min(max(stanza_index, 1), len(poem.stanzas))
        normalized_entities.append({
            "term": require_string(entity.get("term"), f"cultural_entities[{j}].term"),
            "romanization": require_string(entity.get("romanization"), f"cultural_entities[{j}].romanization", allow_empty=True),
            "category": require_enum(entity.get("category"), f"cultural_entities[{j}].category", ALLOWED_ENTITY_CATEGORIES),
            "stanza_index": stanza_index,
            "preserved": bool(entity.get("preserved")),
            "translation_note": require_string(entity.get("translation_note"), f"cultural_entities[{j}].translation_note", allow_empty=True),
        })

    return {
        "recitation_style": recitation_style,
        "emotional_arc": emotional_arc,
        "stanzas": normalized_stanzas,
        "cultural_entities": normalized_entities,
    }


def filter_non_cultural_entities(payload: dict[str, Any], language: str) -> dict[str, Any]:
    from .config import NON_CULTURAL_ENTITY_TERMS
    blocked = {normalize_term(t) for t in NON_CULTURAL_ENTITY_TERMS.get(language, set())}
    if not blocked:
        return payload
    kept = [e for e in payload["cultural_entities"] if normalize_term(e["term"]) not in blocked]
    return {**payload, "cultural_entities": kept}


# ── Source-term evidence gate (Requirement 3) ────────────────────────────────
def apply_source_term_gate(payload: dict[str, Any], poem: PreprocessedPoem) -> dict[str, Any]:
    """Drop hallucinated terms; record what was dropped and which review items to raise."""
    source = poem.original_poem
    dropped_metaphors = 0
    dropped_entities = 0
    review_items: list[dict[str, Any]] = []

    new_stanzas = []
    for st in payload["stanzas"]:
        kept_spans = []
        for span in st["metaphor_spans"]:
            if term_in_source(span["source_term"], source):
                kept_spans.append(span)
            else:
                dropped_metaphors += 1
                review_items.append({
                    "field_path": f"annotation.stanzas[{st['index']}].metaphor_spans",
                    "severity": "low",
                    "resolved_value": "dropped",
                    "model_value": span["source_term"],
                    "note": "metaphor_source_term_not_in_source",
                })
        new_stanzas.append({**st, "metaphor_spans": kept_spans})

    kept_entities = []
    for ent in payload["cultural_entities"]:
        if term_in_source(ent["term"], source):
            kept_entities.append(ent)
        else:
            dropped_entities += 1
            review_items.append({
                "field_path": "annotation.cultural_entities",
                "severity": "medium",
                "resolved_value": "dropped",
                "model_value": ent["term"],
                "note": "entity_term_not_in_source",
            })

    gated = {**payload, "stanzas": new_stanzas, "cultural_entities": kept_entities}
    stats = {
        "entities_total": len(payload["cultural_entities"]),
        "entities_dropped": dropped_entities,
        "metaphors_total": sum(len(s["metaphor_spans"]) for s in payload["stanzas"]),
        "metaphors_dropped": dropped_metaphors,
    }
    return {"payload": gated, "source_term_checks": stats, "review_items": review_items}


# ══════════════════════════════════════════════════════════════════════════
# Schema v1.1 validation (Stage 2)
#
# Everything above this line is the existing schema-version-5 validator and
# is completely unmodified — legacy validation continues to behave exactly
# as before. The functions below are new, additive, and never called by the
# existing schema-version-5 code paths (fetch_gemini_annotation, the async
# pipeline, apply_source_term_gate, etc.). A payload is only validated as
# v1.1 when a caller explicitly invokes validate_model_payload_v1_1() or
# validate_v1_1_envelope() — v1.1-ness is never inferred from payload shape.
#
# Design choice (documented in docs/SCHEMA_V1_1_DECISIONS.md): v1.1
# validation is fail-fast, exactly like legacy validation, but every raised
# error is a SchemaValidationError carrying one structured ValidationIssue
# (object type, field, item index, expected type/allowed values) rather than
# a bare string. This was chosen over a "collect every error and return a
# list" design because it is the least disruptive extension of the existing
# exception-based validation idiom already used throughout this module, and
# every acceptance case in the Stage 2 test suite only needs to assert that
# ONE specific malformed field is rejected — batch multi-error collection
# would add real complexity for no requirement it actually satisfies.
#
# Stage 3 (not this stage) will add span-vs-poem-text grounding checks for
# source_span_original/source_span_translation; here they are validated only
# as optional strings.
#
# CONTRACT (Stage 2.1 — Task 4): everything in this section is a TRANSITIONAL
# candidate/migration validator, not a lifecycle-stage gate. It checks that a
# payload is well-formed enough to be stored as a v1.1-shaped CANDIDATE (or to
# be migrated from a legacy-shaped record) — it says nothing about whether
# that candidate is complete, cross-model-agreed, or reviewer-approved. In
# particular:
#   - Every new v1.1 field being optional/nullable here is a statement about
#     what a *candidate* record may look like, not a completeness bar for
#     "silver" or "gold". A record that validates here (e.g. with every new
#     field null) is NOT thereby silver- or gold-eligible.
#   - preserved's tri-state (True/False/None) is likewise a candidate-level
#     "no assertion made yet" allowance, not a claim that None is an
#     acceptable final state for a reviewed/gold record.
#   - Stricter, separate validation for silver-eligibility and gold-eligibility
#     (e.g. requiring specific fields to be non-null, requiring cross-model
#     agreement, requiring human sign-off) is explicitly OUT OF SCOPE for this
#     stage and will be added later (see docs/ANNOTATION_LIFECYCLE.md for the
#     target lifecycle states and docs/IMPLEMENTATION_PLAN_V1_1.md Stage 9-10
#     for where that stricter validation is planned to live). Do not treat
#     "validates under validate_model_payload_v1_1" as "eligible for silver
#     or gold" anywhere in this codebase.
# ══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ValidationIssue:
    """A single structured v1.1 validation problem.

    object_type: the kind of object that failed, e.g. "payload", "stanza",
        "cultural_entity", "figurative_expression", "translation_loss",
        "metaphor_mapping", "envelope".
    field: the field name within that object ("" for whole-object problems
        such as "must be an object").
    message: human-readable explanation. Only ever references the single
        offending field/value, never surrounding payload content.
    index: the item's 0-based position within its containing list, when
        the object is a list item (None for poem-level/object-level fields).
    expected: a short description of the expected type or allowed values.
    """
    object_type: str
    field: str
    message: str
    index: int | None = None
    expected: str | None = None


class SchemaValidationError(ModelValidationError):
    """Raised by v1.1 validators. Behaves as an ordinary ModelValidationError
    (str(exc) gives a human-readable message, so existing `except
    ModelValidationError` handling continues to work unchanged) but also
    carries the structured issue in `.issue` for programmatic inspection."""

    def __init__(self, issue: ValidationIssue) -> None:
        self.issue = issue
        super().__init__(issue.message)


# ── V1.1 primitive validators (structured errors; never mutate their input) ──
def require_string_field(
    value: Any, field: str, object_type: str, index: int | None = None, *, allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be a string, not {type(value).__name__}.",
            expected="string",
        ))
    stripped = value.strip()
    if not allow_empty and not stripped:
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be a non-empty string.",
            expected="non-empty string",
        ))
    return stripped


def optional_string_field(
    value: Any, field: str, object_type: str, index: int | None = None, *, allow_empty: bool = True,
) -> str | None:
    """Absent/null -> None. Otherwise must be a string (see require_string_field)."""
    if value is None:
        return None
    return require_string_field(value, field, object_type, index, allow_empty=allow_empty)


def optional_line_ref_field(
    value: Any, field: str, object_type: str, index: int | None = None,
) -> str | None:
    """Absent/null -> None. Otherwise must be a string in the canonical
    line_ref SYNTAX (grounding.parse_line_ref) — "L<n>" or "L<n>-L<m>".

    Stage 3 addition (design principle 2: "keep the existing structural
    validator intact unless a small syntax validation addition is clearly
    necessary"): now that Stage 3 formalizes the line_ref grammar, a
    manifestly malformed value (e.g. "stanza 2", "3") is a plain syntax
    error independent of any specific poem's content, so it belongs at the
    schema layer exactly like an enum check does. This checks SYNTAX ONLY —
    it never has access to actual poem text and therefore cannot check
    whether the reference is in range for a specific poem; that bounds
    check is grounding.py's job (see validate_cultural_grounding_v1_1 /
    validate_figurative_grounding_v1_1), run as an explicit second step.
    """
    stripped = optional_string_field(value, field, object_type, index, allow_empty=False)
    if stripped is None:
        return None
    try:
        parse_line_ref(stripped)
    except LineRefError as exc:
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=str(exc),
            expected="'L<n>' or 'L<n>-L<m>' (e.g. 'L1', 'L2-L4')",
        )) from None
    return stripped


def require_enum_field(
    value: Any, field: str, allowed: tuple[str, ...], object_type: str, index: int | None = None,
) -> str:
    normalized = require_string_field(value, field, object_type, index)
    allowed_lookup = {item.casefold(): item for item in allowed}
    key = normalized.casefold()
    if key not in allowed_lookup:
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be one of {list(allowed)}.",
            expected=f"one of {list(allowed)}",
        ))
    return allowed_lookup[key]


def optional_enum_field(
    value: Any, field: str, allowed: tuple[str, ...], object_type: str, index: int | None = None,
) -> str | None:
    """Absent/null -> None. Otherwise must casefold-match one of `allowed`."""
    if value is None:
        return None
    normalized = require_string_field(value, field, object_type, index)
    allowed_lookup = {item.casefold(): item for item in allowed}
    key = normalized.casefold()
    if key not in allowed_lookup:
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be one of {list(allowed)} or null.",
            expected=f"one of {list(allowed)} or null",
        ))
    return allowed_lookup[key]


def optional_string_list_field(
    value: Any, field: str, object_type: str, index: int | None = None,
) -> list[str]:
    """Absent/null -> []. Otherwise must be a list whose every element is a string."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be an array of strings, not {type(value).__name__}.",
            expected="array of strings",
        ))
    result: list[str] = []
    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise SchemaValidationError(ValidationIssue(
                object_type=object_type, field=f"{field}[{item_index}]", index=index,
                message=f"{object_type}.{field}[{item_index}] must be a string, not {type(item).__name__}.",
                expected="string",
            ))
        result.append(item.strip())
    return result


def require_bool_field(
    value: Any, field: str, object_type: str, index: int | None = None,
) -> bool:
    """Strict JSON-boolean check. Unlike legacy `bool(entity.get('preserved'))`,
    this rejects any truthy-but-not-boolean value (e.g. the string "false",
    which Python's bool() would treat as True) — see docs/SCHEMA_V1_1_DECISIONS.md."""
    if not isinstance(value, bool):
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be a JSON boolean (true/false); got {value!r}.",
            expected="boolean",
        ))
    return value


def _normalize_preserved_v1_1(value: Any, object_type: str, index: int | None) -> bool | None:
    """preserved is a genuine optional tri-state field in v1.1: True, False,
    or None.

    Stage 2.1 correction: Stage 2 originally defaulted absent/null to False,
    matching the legacy `bool(entity.get('preserved'))` coercion. That default
    was wrong for v1.1 because it silently collapsed four distinct states
    ("explicitly not preserved", "missing", "not yet annotated", "unresolved")
    into a single False value, which is indistinguishable downstream from a
    model/human explicitly asserting "not preserved". Absent and explicit
    null are now BOTH treated as None ("no assertion made yet"), consistent
    with every other optional v1.1 scalar field (see optional_string_field /
    optional_enum_field, which use the same absent-or-null -> None
    convention). When a value IS present and non-null, it must still be a
    genuine JSON boolean — 0/1/"true"/"false"/etc. are rejected even though
    Python's bool is technically an int subclass. See
    docs/SCHEMA_V1_1_DECISIONS.md (Stage 2.1) for the full rationale and for
    other v1.1 fields that were reviewed for the same class of issue and
    found NOT to need the same fix.

    The legacy schema-version-5 validator (`bool(entity.get("preserved"))`
    in validate_model_payload) is untouched by this change."""
    if value is None:
        return None
    return require_bool_field(value, "preserved", object_type, index)


def ensure_only_keys_v1_1(
    obj: dict[str, Any], allowed_keys: frozenset[str] | set[str], object_type: str, index: int | None = None,
) -> None:
    """Reject unknown fields (Task 9: unknown fields are rejected, never
    silently accepted, and no compatibility extension area is introduced)."""
    extra_keys = sorted(set(obj) - set(allowed_keys))
    if extra_keys:
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=extra_keys[0], index=index,
            message=f"{object_type} contains unexpected key(s): {extra_keys}.",
            expected=f"only {sorted(allowed_keys)}",
        ))


# ── metaphor_mapping (Task 5) ─────────────────────────────────────────────────
def validate_metaphor_mapping_v1_1(
    value: Any, field: str, object_type: str, index: int | None = None,
) -> dict[str, Any] | None:
    """metaphor_mapping is optional/nullable at the containing figurative
    expression (not every expression type needs one), but when present it
    must itself be meaningful: vehicle_concept and tenor_concept are
    required non-empty strings; transferred_attributes is an optional list
    of strings (may be empty — the mapping may exist without an itemized
    attribute list yet). See docs/SCHEMA_V1_1_DECISIONS.md for why this
    {vehicle_concept, tenor_concept, transferred_attributes} shape was
    chosen over mechanically splitting the legacy abstract_meaning field."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type=object_type, field=field, index=index,
            message=f"{object_type}.{field} must be an object or null, not {type(value).__name__}.",
            expected="object with vehicle_concept, tenor_concept, transferred_attributes",
        ))
    extra_keys = sorted(set(value) - METAPHOR_MAPPING_KEYS)
    if extra_keys:
        raise SchemaValidationError(ValidationIssue(
            object_type="metaphor_mapping", field=extra_keys[0], index=index,
            message=f"metaphor_mapping contains unexpected key(s): {extra_keys}.",
            expected=f"only {sorted(METAPHOR_MAPPING_KEYS)}",
        ))
    vehicle_concept = require_string_field(value.get("vehicle_concept"), "vehicle_concept", "metaphor_mapping", index)
    tenor_concept = require_string_field(value.get("tenor_concept"), "tenor_concept", "metaphor_mapping", index)
    transferred_attributes = optional_string_list_field(
        value.get("transferred_attributes"), "transferred_attributes", "metaphor_mapping", index,
    )
    return {
        "vehicle_concept": vehicle_concept,
        "tenor_concept": tenor_concept,
        "transferred_attributes": transferred_attributes,
    }


# ── Figurative expression (metaphor_spans item), v1.1 ────────────────────────
def validate_figurative_expression_v1_1(item: Any, stanza_number: int, index: int) -> dict[str, Any]:
    """Validates one entry of stanzas[].metaphor_spans under v1.1. The stored
    key name `metaphor_spans` is unchanged (Task: do not rename stored keys
    in this stage) even though, conceptually, entries may now be any
    expression_type, not only metaphors."""
    if not isinstance(item, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="figurative_expression", field="", index=index,
            message=f"stanzas[{stanza_number - 1}].metaphor_spans[{index}] must be an object, not {type(item).__name__}.",
            expected="object",
        ))
    ensure_only_keys_v1_1(item, METAPHOR_KEYS_V1_1, "figurative_expression", index)

    source_term = require_string_field(item.get("source_term"), "source_term", "figurative_expression", index)
    abstract_meaning = require_string_field(item.get("abstract_meaning"), "abstract_meaning", "figurative_expression", index)

    return {
        # Legacy fields — unchanged in shape and meaning.
        "source_term": source_term,
        "abstract_meaning": abstract_meaning,
        # V1.1 additions — all optional/nullable; none require the others.
        "source_span_original": optional_string_field(item.get("source_span_original"), "source_span_original", "figurative_expression", index),
        "source_span_translation": optional_string_field(item.get("source_span_translation"), "source_span_translation", "figurative_expression", index),
        "expression_type": optional_enum_field(item.get("expression_type"), "expression_type", ALLOWED_EXPRESSION_TYPES, "figurative_expression", index),
        "literal_meaning": optional_string_field(item.get("literal_meaning"), "literal_meaning", "figurative_expression", index),
        "vehicle": optional_string_field(item.get("vehicle"), "vehicle", "figurative_expression", index),
        "tenor": optional_string_field(item.get("tenor"), "tenor", "figurative_expression", index),
        "metaphor_mapping": validate_metaphor_mapping_v1_1(item.get("metaphor_mapping"), "metaphor_mapping", "figurative_expression", index),
        "line_ref": optional_line_ref_field(item.get("line_ref"), "line_ref", "figurative_expression", index),
        # Pilot-only / not-yet-enumerated (Task 2): open string or null, no controlled vocabulary yet.
        "literalization_risk": optional_string_field(item.get("literalization_risk"), "literalization_risk", "figurative_expression", index),
        "visualization_strategy": optional_string_field(item.get("visualization_strategy"), "visualization_strategy", "figurative_expression", index),
        "acceptable_visual_variants": optional_string_list_field(item.get("acceptable_visual_variants"), "acceptable_visual_variants", "figurative_expression", index),
        "visualization_difficulty": optional_string_field(item.get("visualization_difficulty"), "visualization_difficulty", "figurative_expression", index),
    }


# ── Structured translation loss (Task 6) ──────────────────────────────────────
def validate_translation_loss_items(value: Any, stanza_number: int) -> list[dict[str, Any]]:
    """Optional stanza-level list, additive alongside (never replacing) the
    existing stanza-level loss_note. Placement rationale documented in
    docs/SCHEMA_V1_1_DECISIONS.md. what_was_lost is required content when an
    item is present at all (an item with nothing lost would be meaningless);
    where/severity are optional. No severity enum is invented (Task 6)."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise SchemaValidationError(ValidationIssue(
            object_type="translation_loss", field="translation_loss", index=None,
            message=f"stanzas[{stanza_number}].translation_loss must be an array, not {type(value).__name__}.",
            expected="array of objects",
        ))
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise SchemaValidationError(ValidationIssue(
                object_type="translation_loss", field="", index=idx,
                message=f"stanzas[{stanza_number}].translation_loss[{idx}] must be an object, not {type(item).__name__}.",
                expected="object",
            ))
        ensure_only_keys_v1_1(item, TRANSLATION_LOSS_KEYS, "translation_loss", idx)
        what_was_lost = require_string_field(item.get("what_was_lost"), "what_was_lost", "translation_loss", idx)
        # `where` uses the same L<n>/L<n>-L<m> convention as line_ref (Task 2
        # rule 8), against the original-language poem's line numbering.
        where = optional_line_ref_field(item.get("where"), "where", "translation_loss", idx)
        severity = optional_string_field(item.get("severity"), "severity", "translation_loss", idx, allow_empty=False)
        normalized.append({"what_was_lost": what_was_lost, "where": where, "severity": severity})
    return normalized


# ── Cultural entity, v1.1 ──────────────────────────────────────────────────────
def validate_cultural_entity_v1_1(entity: Any, index: int, poem: PreprocessedPoem) -> dict[str, Any]:
    if not isinstance(entity, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="cultural_entity", field="", index=index,
            message=f"cultural_entities[{index}] must be an object, not {type(entity).__name__}.",
            expected="object",
        ))
    ensure_only_keys_v1_1(entity, ENTITY_KEYS_V1_1, "cultural_entity", index)

    stanza_index_raw = entity.get("stanza_index", 1)
    try:
        stanza_index = int(stanza_index_raw)
    except (TypeError, ValueError):
        raise SchemaValidationError(ValidationIssue(
            object_type="cultural_entity", field="stanza_index", index=index,
            message=f"cultural_entities[{index}].stanza_index must be an integer.",
            expected="integer",
        )) from None
    stanza_index = min(max(stanza_index, 1), len(poem.stanzas))

    return {
        # Legacy fields — unchanged in shape and meaning.
        "term": require_string_field(entity.get("term"), "term", "cultural_entity", index),
        "romanization": require_string_field(entity.get("romanization"), "romanization", "cultural_entity", index, allow_empty=True),
        "category": require_enum_field(entity.get("category"), "category", ALLOWED_ENTITY_CATEGORIES, "cultural_entity", index),
        "stanza_index": stanza_index,
        "preserved": _normalize_preserved_v1_1(entity.get("preserved"), "cultural_entity", index),
        "translation_note": require_string_field(entity.get("translation_note"), "translation_note", "cultural_entity", index, allow_empty=True),
        # V1.1 additions — all optional/nullable.
        "gloss": optional_string_field(entity.get("gloss"), "gloss", "cultural_entity", index),
        "line_ref": optional_line_ref_field(entity.get("line_ref"), "line_ref", "cultural_entity", index),
        "source_span_original": optional_string_field(entity.get("source_span_original"), "source_span_original", "cultural_entity", index),
        "source_span_translation": optional_string_field(entity.get("source_span_translation"), "source_span_translation", "cultural_entity", index),
        "visual_features": optional_string_list_field(entity.get("visual_features"), "visual_features", "cultural_entity", index),
        "visual_priority": optional_enum_field(entity.get("visual_priority"), "visual_priority", ALLOWED_VISUAL_PRIORITIES, "cultural_entity", index),
        "acceptable_visual_variants": optional_string_list_field(entity.get("acceptable_visual_variants"), "acceptable_visual_variants", "cultural_entity", index),
        "negative_confusions": optional_string_list_field(entity.get("negative_confusions"), "negative_confusions", "cultural_entity", index),
        # translation_status: absent/null/non-empty string only (Task 4 — empty string rejected).
        "translation_status": optional_string_field(entity.get("translation_status"), "translation_status", "cultural_entity", index, allow_empty=False),
        # Pilot-only (Task 2): open string or null, no controlled vocabulary yet.
        "cultural_specificity_level": optional_string_field(entity.get("cultural_specificity_level"), "cultural_specificity_level", "cultural_entity", index),
    }


# ── Stanza, v1.1 ────────────────────────────────────────────────────────────────
def validate_stanza_v1_1(stanza: Any, position: int) -> dict[str, Any]:
    index = position - 1
    if not isinstance(stanza, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="stanza", field="", index=index,
            message=f"stanzas[{index}] must be an object, not {type(stanza).__name__}.",
            expected="object",
        ))
    ensure_only_keys_v1_1(stanza, STANZA_KEYS_V1_1, "stanza", index)
    if stanza.get("index") != position:
        raise SchemaValidationError(ValidationIssue(
            object_type="stanza", field="index", index=index,
            message=f"stanzas[{index}].index must be {position}.",
            expected=str(position),
        ))

    translation_quality = require_enum_field(
        stanza.get("translation_quality"), "translation_quality", ALLOWED_TRANSLATION_QUALITIES, "stanza", index,
    )
    loss_note = require_string_field(stanza.get("loss_note"), "loss_note", "stanza", index, allow_empty=True)
    if translation_quality == "faithful" and loss_note:
        raise SchemaValidationError(ValidationIssue(
            object_type="stanza", field="loss_note", index=index,
            message=f"stanzas[{index}].loss_note must be empty when translation_quality is faithful.",
            expected="empty string",
        ))

    metaphor_spans_raw = stanza.get("metaphor_spans")
    if metaphor_spans_raw is None:
        metaphor_spans_raw = []
    if not isinstance(metaphor_spans_raw, list):
        raise SchemaValidationError(ValidationIssue(
            object_type="stanza", field="metaphor_spans", index=index,
            message=f"stanzas[{index}].metaphor_spans must be an array, not {type(metaphor_spans_raw).__name__}.",
            expected="array",
        ))
    metaphor_spans = [
        validate_figurative_expression_v1_1(m, position, m_index) for m_index, m in enumerate(metaphor_spans_raw)
    ]

    translation_loss = validate_translation_loss_items(stanza.get("translation_loss"), index)

    return {
        "index": position,
        "emotion": require_enum_field(stanza.get("emotion"), "emotion", ALLOWED_EMOTIONS, "stanza", index),
        "tone": require_enum_field(stanza.get("tone"), "tone", ALLOWED_TONES, "stanza", index),
        "translation_quality": translation_quality,
        "loss_note": loss_note,
        "metaphor_spans": metaphor_spans,
        "translation_loss": translation_loss,
    }


# ── Full v1.1 payload validation ──────────────────────────────────────────────
def validate_model_payload_v1_1(payload: Any, poem: PreprocessedPoem) -> dict[str, Any]:
    """The v1.1 counterpart of validate_model_payload(). Same overall shape
    and stanza-count contract as the legacy validator (a payload is fundamentally
    poem-shaped: N stanzas must match), extended with the v1.1 fields. Does
    NOT require any poem to contain at least one cultural entity or
    figurative expression (Task 9) — empty lists are valid."""
    if not isinstance(payload, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="payload", field="", index=None,
            message=f"Model output must be a JSON object, not {type(payload).__name__}.",
            expected="object",
        ))
    ensure_only_keys_v1_1(payload, TOPLEVEL_KEYS_V1_1, "payload")

    recitation_style = require_enum_field(payload.get("recitation_style"), "recitation_style", ALLOWED_RECITATION_STYLES, "payload")
    emotional_arc = coerce_to_text(payload.get("emotional_arc"))  # reuses the legacy helper; never raises

    stanzas = payload.get("stanzas")
    if not isinstance(stanzas, list):
        raise SchemaValidationError(ValidationIssue(
            object_type="payload", field="stanzas", index=None,
            message=f"stanzas must be a JSON array, not {type(stanzas).__name__}.",
            expected="array",
        ))
    if len(stanzas) != len(poem.stanzas):
        raise SchemaValidationError(ValidationIssue(
            object_type="payload", field="stanzas", index=None,
            message=f"Expected {len(poem.stanzas)} stanzas but model returned {len(stanzas)}.",
            expected=str(len(poem.stanzas)),
        ))
    normalized_stanzas = [validate_stanza_v1_1(s, i) for i, s in enumerate(stanzas, start=1)]

    cultural_entities = payload.get("cultural_entities")
    if cultural_entities is None:
        cultural_entities = []
    if not isinstance(cultural_entities, list):
        raise SchemaValidationError(ValidationIssue(
            object_type="payload", field="cultural_entities", index=None,
            message=f"cultural_entities must be a JSON array, not {type(cultural_entities).__name__}.",
            expected="array",
        ))
    normalized_entities = [validate_cultural_entity_v1_1(e, j, poem) for j, e in enumerate(cultural_entities)]

    theme = optional_string_field(payload.get("theme"), "theme", "payload")

    return {
        "recitation_style": recitation_style,
        "emotional_arc": emotional_arc,
        "stanzas": normalized_stanzas,
        "cultural_entities": normalized_entities,
        "theme": theme,
    }


def validate_v1_1_envelope(record: Any, poem: PreprocessedPoem) -> dict[str, Any]:
    """Task 9: the minimum requirement for a record to be considered a
    schema-v1.1 record. A bare {"schema_version": "1.1"} is NOT sufficient —
    the record must also carry an "annotation" object that itself passes
    full v1.1 payload validation.

    Scope note: this checks only `schema_version` and `annotation`. It does
    not validate the surrounding output-file envelope (poem_id, status,
    preprocessing, review_items, _raw, ...) — that shape belongs to
    output.py, which is out of scope for Stage 2 (schema.py/models.py only).
    """
    if not isinstance(record, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="envelope", field="", index=None,
            message=f"A v1.1 record must be a JSON object, not {type(record).__name__}.",
            expected="object",
        ))
    version = record.get("schema_version")
    if version != MORPHOVERSE_SCHEMA_VERSION:
        raise SchemaValidationError(ValidationIssue(
            object_type="envelope", field="schema_version", index=None,
            message=f"schema_version must be exactly {MORPHOVERSE_SCHEMA_VERSION!r} for a v1.1 record; got {version!r}.",
            expected=repr(MORPHOVERSE_SCHEMA_VERSION),
        ))
    annotation = record.get("annotation")
    if not isinstance(annotation, dict):
        raise SchemaValidationError(ValidationIssue(
            object_type="envelope", field="annotation", index=None,
            message=f"A v1.1 record must contain an 'annotation' object, not {type(annotation).__name__}.",
            expected="object",
        ))
    return validate_model_payload_v1_1(annotation, poem)


# ── API call (Gemini only) ───────────────────────────────────────────────────
class EmptyModelResponse(RuntimeError):
    """Proxy returned HTTP 200 but with no text content (often throttling/quota)."""


def call_model_text(model: str, system_prompt: str, user_prompt: str, token: str,
                    base_url: str, *, max_tokens: int) -> str:
    client = LLMProxyClient(token=token, base_url=base_url)
    response = client.chat(
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=REQUEST_TEMPERATURE,
        max_tokens=max_tokens,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise EmptyModelResponse(f"malformed response shape: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise EmptyModelResponse("model returned empty content")
    return content


def _call_with_retries(request_func, model, system_prompt, prompt_text, token, base_url,
                       *, max_tokens, retries, base_delay):
    """Synchronous call wrapped in bounded exponential-backoff retries.
    Retries on empty responses and on transient proxy errors (429 / 5xx / network)."""
    import time

    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            text = request_func(model, system_prompt, prompt_text, token, base_url, max_tokens=max_tokens)
            if not isinstance(text, str) or not text.strip():
                raise EmptyModelResponse("model returned empty content")
            return text
        except EmptyModelResponse as exc:
            last_exc = exc
        except LLMProxyError as exc:
            # Only retry transient statuses; re-raise hard auth/validation errors immediately.
            if exc.status_code in (0, 408, 409, 425, 429, 500, 502, 503, 504):
                last_exc = exc
            else:
                raise
        if attempt < retries:
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc if last_exc else EmptyModelResponse("exhausted retries")


async def fetch_gemini_annotation(
    poem: PreprocessedPoem,
    system_prompt: str,
    primary_user_prompt: str,
    repair_user_prompt: str,
    token: str,
    base_url: str,
    request_fn: Any = None,
) -> dict[str, Any]:
    """Gemini-only: primary -> repair -> flash fallback. No voting."""
    import asyncio
    import inspect

    request_func = request_fn or call_model_text
    budget = max_tokens_for(len(poem.stanzas))
    last_raw = ""
    last_error = ""

    attempts = [
        (GEMINI_PRIMARY, primary_user_prompt, "primary"),
        (GEMINI_PRIMARY, repair_user_prompt, "repair"),
        (GEMINI_FALLBACK, repair_user_prompt, "flash_fallback"),
    ]

    for attempt_index, (model, prompt_text, kind) in enumerate(attempts):
        try:
            if request_fn is None:
                last_raw = await asyncio.to_thread(
                    _call_with_retries, request_func, model, system_prompt, prompt_text, token, base_url,
                    max_tokens=budget, retries=3, base_delay=2.0,
                )
            else:
                result = request_func(model, system_prompt, prompt_text, token, base_url, max_tokens=budget)
                last_raw = await result if inspect.isawaitable(result) else result
                if not isinstance(last_raw, str) or not last_raw.strip():
                    raise EmptyModelResponse("model returned empty content")
        except EmptyModelResponse as exc:
            last_error = f"empty_response: {exc}"
            if kind != "flash_fallback":
                continue
            return _fail("empty_response", attempt_index, last_raw or "", last_error, model, kind)
        except LLMProxyError as exc:
            last_error = f"{exc.status_code} {exc.code}: {exc.message}"
            if kind != "flash_fallback":
                continue
            return _fail("api_error", attempt_index, last_raw or "", last_error, model, kind)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if kind != "flash_fallback":
                continue
            return _fail("request_error", attempt_index, last_raw or "", last_error, model, kind)

        try:
            parsed = expand_abbreviated_keys(extract_json_payload(last_raw))
        except json.JSONDecodeError as exc:
            if kind != "flash_fallback":
                continue
            return _fail("parse_failed", attempt_index, last_raw, f"JSON parse failed: {exc}", model, kind)

        try:
            validated = validate_model_payload(parsed, poem)
            validated = filter_non_cultural_entities(validated, poem.language)
        except StanzaCountMismatch as exc:
            if kind != "flash_fallback":
                continue
            return _fail("stanza_mismatch", attempt_index, last_raw, str(exc), model, kind)
        except ModelValidationError as exc:
            if kind != "flash_fallback":
                continue
            return _fail("validation_failed", attempt_index, last_raw, str(exc), model, kind)

        return {"status": "valid", "model": model, "prompt_kind": kind,
                "retry_count": attempt_index, "raw_text": last_raw,
                "parsed": validated, "discard_reason": ""}

    return _fail("empty_response", len(attempts) - 1, last_raw or "",
                 last_error or "All Gemini attempts returned empty/invalid output.", GEMINI_FALLBACK, "flash_fallback")


def _fail(status: str, attempt: int, raw: str, reason: str, model: str, kind: str) -> dict[str, Any]:
    return {"status": status, "model": model, "prompt_kind": kind,
            "retry_count": attempt, "raw_text": raw, "parsed": None, "discard_reason": reason}
