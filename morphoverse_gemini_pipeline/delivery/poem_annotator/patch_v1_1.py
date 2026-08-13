"""Stage 5B — flat path-value patch format, safe path resolution, and patch
application for schema-v1.1 pre-backfill candidates.

Scope discipline: this module never calls a model, provider, or proxy, and
never reads/writes a credential. It only (a) parses and validates a strict
flat `{schema_version, poem_id, patches: [{path, value}, ...]}` document,
(b) resolves/sets dotted+indexed paths against an in-memory document without
`eval`/`exec` and without ever creating a container that doesn't already
exist, and (c) applies a validated patch to a DEEP COPY of a candidate,
running it back through Stage 2's structural validator
(`models.validate_model_payload_v1_1`) and Stage 3's transitional grounding
validators before accepting the transaction. It never fetches, generates,
or stores an actual provider response — see `ProviderResponseRecord` below,
which is a data shape for a *future* stage, not something this module
produces.

See docs/PATCH_BACKFILL_STAGE5B.md for the full contract.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .dataset import PreprocessedPoem, StanzaInput
from .models import (
    ModelValidationError,
    SchemaValidationError,
    validate_model_payload_v1_1,
    validate_metaphor_mapping_v1_1,
    validate_translation_loss_items,
)
from .grounding import (
    parse_line_ref,
    LineRefError,
    validate_cultural_grounding_v1_1,
    validate_figurative_grounding_v1_1,
    GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
)
from .schema import (
    MORPHOVERSE_SCHEMA_VERSION,
    ALLOWED_EXPRESSION_TYPES,
    ALLOWED_VISUAL_PRIORITIES,
)

# ══════════════════════════════════════════════════════════════════════════
# Task 1 — missing-path classification (shared with the backfill-plan module)
# ══════════════════════════════════════════════════════════════════════════
CLASS_MODEL_BACKFILL_ALLOWED = "model_backfill_allowed"
CLASS_DETERMINISTIC_RETRY = "deterministic_retry"
CLASS_HUMAN_REVIEW_REQUIRED = "human_review_required"
CLASS_INTENTIONALLY_NULLABLE = "intentionally_nullable"
CLASS_NOT_APPLICABLE = "not_applicable"

# Fields a model MAY be asked to backfill in Stage 5B, subject to the same
# type validation (Task 5) and, for grounding-sensitive fields, mandatory
# re-grounding (Task 6) before any patch touching them is accepted.
# `source_span_translation`/`visual_priority` are included per Task 5's own
# validation rules for them; nothing here bypasses grounding or enum checks.
MODEL_BACKFILL_FIELDS = frozenset({
    "gloss", "visual_features", "acceptable_visual_variants", "negative_confusions",
    "theme", "expression_type", "literal_meaning", "vehicle", "tenor",
    "metaphor_mapping", "literalization_risk", "visualization_strategy",
    "translation_loss", "source_span_translation", "visual_priority",
})
# Fields where Stage 5A already made its best mechanical attempt and failed
# (ambiguous occurrence, or the underlying value depends on a still-unresolved
# schema decision) — a model re-attempt over the same poem text has no new
# information to work with, so these are routed to human review instead of
# a model-backfill request (see Task 1's own examples).
HUMAN_REVIEW_FIELDS = frozenset({"line_ref", "source_span_original", "translation_status"})
# Pilot-only fields with no finalized enum (docs/SCHEMA_V1_1_DECISIONS.md) —
# remain intentionally null unless a future stage explicitly justifies them.
INTENTIONALLY_NULLABLE_FIELDS = frozenset({"cultural_specificity_level", "visualization_difficulty"})

# Fields whose value, once patched in, must be re-verified against the poem
# text via Stage 3 grounding before the transaction is accepted (Task 5/6).
GROUNDING_SENSITIVE_FIELDS = frozenset({"source_span_original", "source_span_translation", "line_ref"})


def leaf_field(path: str) -> str:
    """The final dotted component of a patch path, e.g. 'gloss' for
    'annotation.cultural_entities[2].gloss'. Pure string op, no parsing."""
    return path.rsplit(".", 1)[-1].split("[", 1)[0]


def classify_missing_path(path: str) -> str:
    """Classify a single Stage 5A missing-field path into exactly one of the
    five Task 1 categories, based on the field it targets. Never requests a
    field merely because it is null — the classification is keyed on WHICH
    field it is, not on the fact that it appeared in the missing report."""
    leaf = leaf_field(path)
    if leaf in INTENTIONALLY_NULLABLE_FIELDS:
        return CLASS_INTENTIONALLY_NULLABLE
    if leaf in HUMAN_REVIEW_FIELDS:
        return CLASS_HUMAN_REVIEW_REQUIRED
    if leaf in MODEL_BACKFILL_FIELDS:
        return CLASS_MODEL_BACKFILL_ALLOWED
    # Any field name this project doesn't yet recognize at all. None of the
    # current Stage 5A missing-field reports produce this outcome (verified
    # by classifying every path in pilot/reports/stage5a/missing_fields.json)
    # — it exists as a safe default for a future, not-yet-catalogued field.
    return CLASS_NOT_APPLICABLE


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — safe, structured path resolution (no eval/exec)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PathIssue:
    code: str
    path: str
    message: str


class PatchPathError(ValueError):
    """Raised for any path-syntax or path-resolution problem. Carries a
    single structured PathIssue — never a bare KeyError/IndexError/TypeError."""

    def __init__(self, issue: PathIssue):
        super().__init__(issue.message)
        self.issue = issue


@dataclass(frozen=True)
class _KeyStep:
    name: str


@dataclass(frozen=True)
class _IndexStep:
    index: int


# Each dot-separated segment is a bare identifier optionally followed by one
# or more [<digits>] index accessors, e.g. "cultural_entities[0]",
# "metaphor_spans[1]", "theme". No other character class is accepted —
# there is no way to smuggle attribute access, calls, or operators through
# this grammar, and no eval()/exec() is used anywhere in this module.
_SEGMENT_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)((?:\[[0-9]+\])*)$")
_INDEX_RE = re.compile(r"\[([0-9]+)\]")
_BAD_INDEX_RE = re.compile(r"\[-[0-9]+\]|\[[0-9]*[^0-9\]][0-9]*\]")


def parse_patch_path(path: str) -> tuple[Any, ...]:
    """Parse a dotted/indexed patch path into an ordered tuple of
    `_KeyStep`/`_IndexStep`. Pure syntax check only — does not touch any
    document. Raises PatchPathError for anything not matching the grammar,
    including negative indexes, non-numeric indexes, empty segments, leading/
    trailing dots, whitespace, or any character outside
    `[a-zA-Z0-9_.\\[\\]]`."""
    if not isinstance(path, str) or not path:
        raise PatchPathError(PathIssue("empty_or_non_string_path", str(path), "path must be a non-empty string."))
    if path != path.strip():
        raise PatchPathError(PathIssue("path_has_whitespace", path, "path must not have leading/trailing whitespace."))
    if not re.fullmatch(r"[a-zA-Z0-9_.\[\]-]+", path):
        raise PatchPathError(PathIssue("unsupported_path_syntax", path, "path contains unsupported characters."))
    if ".." in path or path.startswith(".") or path.endswith("."):
        raise PatchPathError(PathIssue("unsupported_path_syntax", path, "path has an empty segment."))

    steps: list[Any] = []
    for segment in path.split("."):
        if _BAD_INDEX_RE.search(segment):
            raise PatchPathError(PathIssue("negative_or_invalid_index", path, f"segment {segment!r} has a negative or non-numeric index."))
        match = _SEGMENT_RE.match(segment)
        if not match:
            raise PatchPathError(PathIssue("unsupported_path_syntax", path, f"segment {segment!r} is not a valid key[index]* form."))
        key, indexes = match.group(1), match.group(2)
        steps.append(_KeyStep(key))
        for idx_str in _INDEX_RE.findall(indexes):
            steps.append(_IndexStep(int(idx_str)))
    return tuple(steps)


def _require_annotation_scope(path: str, steps: tuple[Any, ...]) -> None:
    if not steps or not isinstance(steps[0], _KeyStep) or steps[0].name != "annotation":
        raise PatchPathError(PathIssue("path_outside_annotation", path, "path must start with 'annotation'."))


def get_value_at_path(document: dict, path: str) -> Any:
    """Resolve `path` against `document` (read-only — never mutates
    `document`). Raises PatchPathError with a structured PathIssue for any
    unknown key, out-of-range/negative index, or type mismatch — never a
    bare KeyError/IndexError/TypeError."""
    steps = parse_patch_path(path)
    _require_annotation_scope(path, steps)
    cur: Any = document
    for step in steps:
        if isinstance(step, _KeyStep):
            if not isinstance(cur, dict):
                raise PatchPathError(PathIssue("type_mismatch", path, f"expected an object at {step.name!r}, found {type(cur).__name__}."))
            if step.name not in cur:
                raise PatchPathError(PathIssue("unknown_key", path, f"key {step.name!r} does not exist."))
            cur = cur[step.name]
        else:
            if not isinstance(cur, list):
                raise PatchPathError(PathIssue("type_mismatch", path, f"expected an array at index {step.index}, found {type(cur).__name__}."))
            if step.index < 0:
                raise PatchPathError(PathIssue("negative_index", path, f"index {step.index} is negative."))
            if step.index >= len(cur):
                raise PatchPathError(PathIssue("index_out_of_range", path, f"index {step.index} is out of range for a list of length {len(cur)}."))
            cur = cur[step.index]
    return cur


def set_value_at_path(document: dict, path: str, value: Any) -> None:
    """Set `path` to `value` on `document` IN PLACE. The caller is
    responsible for deep-copying `document` first if the original must be
    preserved (Task 3: this function itself performs no copy — it is the
    low-level primitive `apply_patch_v1_1` calls only after deep-copying).
    Never creates a container that doesn't already exist: the parent object/
    array and the final key/index must already be present, exactly like
    `get_value_at_path`'s own resolution rules. Raises PatchPathError with a
    structured PathIssue on any failure, before any mutation occurs."""
    steps = parse_patch_path(path)
    _require_annotation_scope(path, steps)
    if not steps:
        raise PatchPathError(PathIssue("empty_path", path, "path resolves to no steps."))

    cur: Any = document
    for step in steps[:-1]:
        if isinstance(step, _KeyStep):
            if not isinstance(cur, dict) or step.name not in cur:
                raise PatchPathError(PathIssue("unknown_key", path, f"key {step.name!r} does not exist."))
            cur = cur[step.name]
        else:
            if not isinstance(cur, list):
                raise PatchPathError(PathIssue("type_mismatch", path, f"expected an array at index {step.index}."))
            if step.index < 0:
                raise PatchPathError(PathIssue("negative_index", path, f"index {step.index} is negative."))
            if step.index >= len(cur):
                raise PatchPathError(PathIssue("index_out_of_range", path, f"index {step.index} is out of range."))
            cur = cur[step.index]

    last = steps[-1]
    if isinstance(last, _KeyStep):
        if not isinstance(cur, dict) or last.name not in cur:
            raise PatchPathError(PathIssue("unknown_key", path, f"key {last.name!r} does not exist; set_value_at_path never creates new keys."))
        cur[last.name] = value
    else:
        if not isinstance(cur, list):
            raise PatchPathError(PathIssue("type_mismatch", path, f"expected an array at index {last.index}."))
        if last.index < 0:
            raise PatchPathError(PathIssue("negative_index", path, f"index {last.index} is negative."))
        if last.index >= len(cur):
            raise PatchPathError(PathIssue("index_out_of_range", path, f"index {last.index} is out of range; set_value_at_path never extends an array."))
        cur[last.index] = value


# ══════════════════════════════════════════════════════════════════════════
# Task 2 — flat patch document format
# ══════════════════════════════════════════════════════════════════════════
_TOP_LEVEL_KEYS = frozenset({"schema_version", "poem_id", "patches"})
_PATCH_ITEM_KEYS = frozenset({"path", "value"})


@dataclass(frozen=True)
class PatchFormatIssue:
    code: str
    message: str


class PatchFormatError(ValueError):
    """Raised for any violation of the flat patch-document format. Carries a
    single structured PatchFormatIssue."""

    def __init__(self, issue: PatchFormatIssue):
        super().__init__(issue.message)
        self.issue = issue


def validate_patch_document(
    raw: Any,
    *,
    expected_poem_id: str,
    approved_requested_paths: "set[str] | frozenset[str]",
) -> dict[str, Any]:
    """Validate the flat `{schema_version, poem_id, patches}` document.
    Returns a `{path: value}` mapping (patches, in document order) on
    success. Raises PatchFormatError on the first violation found — never
    partially accepts a malformed document. No explanatory-prose sibling
    field can ever pass this check, since only exactly {schema_version,
    poem_id, patches} is accepted at top level and only exactly {path,
    value} per patch item — there is no field a model could use to attach
    commentary."""
    if not isinstance(raw, dict):
        raise PatchFormatError(PatchFormatIssue("not_an_object", f"patch document must be an object, not {type(raw).__name__}."))

    extra_top = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if extra_top:
        raise PatchFormatError(PatchFormatIssue("unexpected_top_level_field", f"unexpected top-level field(s): {extra_top}."))
    missing_top = sorted(_TOP_LEVEL_KEYS - set(raw))
    if missing_top:
        raise PatchFormatError(PatchFormatIssue("missing_top_level_field", f"missing top-level field(s): {missing_top}."))

    if raw.get("schema_version") != MORPHOVERSE_SCHEMA_VERSION:
        raise PatchFormatError(PatchFormatIssue("wrong_schema_version", f"schema_version must be exactly {MORPHOVERSE_SCHEMA_VERSION!r}, got {raw.get('schema_version')!r}."))

    if raw.get("poem_id") != expected_poem_id:
        raise PatchFormatError(PatchFormatIssue("wrong_poem_id", f"poem_id must be {expected_poem_id!r}, got {raw.get('poem_id')!r}."))

    patches_raw = raw.get("patches")
    if not isinstance(patches_raw, list) or len(patches_raw) == 0:
        raise PatchFormatError(PatchFormatIssue("empty_or_invalid_patches", "patches must be a non-empty list."))

    seen_paths: set[str] = set()
    ordered: dict[str, Any] = {}
    for i, item in enumerate(patches_raw):
        if not isinstance(item, dict):
            raise PatchFormatError(PatchFormatIssue("invalid_patch_item", f"patches[{i}] must be an object, not {type(item).__name__}."))
        extra_item = sorted(set(item) - _PATCH_ITEM_KEYS)
        if extra_item:
            raise PatchFormatError(PatchFormatIssue("unexpected_patch_item_field", f"patches[{i}] has unexpected field(s): {extra_item}."))
        missing_item = sorted(_PATCH_ITEM_KEYS - set(item))
        if missing_item:
            raise PatchFormatError(PatchFormatIssue("missing_patch_item_field", f"patches[{i}] is missing field(s): {missing_item}."))

        path = item["path"]
        if not isinstance(path, str):
            raise PatchFormatError(PatchFormatIssue("invalid_path_type", f"patches[{i}].path must be a string."))
        if path in seen_paths:
            raise PatchFormatError(PatchFormatIssue("duplicate_path", f"path {path!r} appears more than once."))

        try:
            steps = parse_patch_path(path)
            _require_annotation_scope(path, steps)
        except PatchPathError as exc:
            raise PatchFormatError(PatchFormatIssue(exc.issue.code, f"patches[{i}].path {path!r}: {exc.issue.message}")) from exc

        if path not in approved_requested_paths:
            raise PatchFormatError(PatchFormatIssue("unrequested_path", f"path {path!r} is not in the approved requested-field list."))

        seen_paths.add(path)
        ordered[path] = item["value"]

    return ordered


# ══════════════════════════════════════════════════════════════════════════
# Task 4 — existing-value protection
# ══════════════════════════════════════════════════════════════════════════
def is_approved_migration_placeholder(current_value: Any) -> bool:
    """True only for the two Stage 5A migration-default placeholder shapes:
    `null`, or an empty list. Every other current value — non-empty string,
    non-empty list, True/False, a number, or an existing structured object —
    is protected and must never be overwritten by a patch."""
    if current_value is None:
        return True
    if isinstance(current_value, list) and len(current_value) == 0:
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# Task 5 — type validation by target field
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PatchValueIssue:
    code: str
    path: str
    message: str


class PatchValueError(ValueError):
    """Raised when a patch value fails type validation for its target
    field. Carries a single structured PatchValueIssue."""

    def __init__(self, issue: PatchValueIssue):
        super().__init__(issue.message)
        self.issue = issue


def _nonempty_string_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PatchValueError(PatchValueIssue("not_a_list", path, f"{path} must be a list of non-empty strings."))
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise PatchValueError(PatchValueIssue("non_string_list_element", path, f"{path} contains a non-string or empty-string element."))
    return list(value)


def _optional_nonempty_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PatchValueError(PatchValueIssue("invalid_string", path, f"{path} must be a non-empty string or null."))
    return value


def _enum_or_none(value: Any, path: str, allowed: tuple[str, ...]) -> str | None:
    if value is None:
        return None
    if value not in allowed:
        raise PatchValueError(PatchValueIssue("invalid_enum", path, f"{path} must be one of {sorted(allowed)} or null, got {value!r}."))
    return value


def _line_ref_or_none(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise PatchValueError(PatchValueIssue("invalid_line_ref", path, f"{path} must be a string or null."))
    try:
        parse_line_ref(value)
    except LineRefError as exc:
        raise PatchValueError(PatchValueIssue("invalid_line_ref", path, str(exc))) from exc
    return value


def _metaphor_mapping_or_none(value: Any, path: str) -> dict | None:
    if value is None:
        return None
    try:
        return validate_metaphor_mapping_v1_1(value, "metaphor_mapping", "figurative_expression")
    except SchemaValidationError as exc:
        raise PatchValueError(PatchValueIssue("invalid_metaphor_mapping", path, str(exc))) from exc


_TRANSLATION_LOSS_STANZA_RE = re.compile(r"^annotation\.stanzas\[(\d+)\]\.translation_loss$")


def _translation_loss_list(value: Any, path: str) -> list[dict]:
    match = _TRANSLATION_LOSS_STANZA_RE.match(path)
    stanza_number = int(match.group(1)) if match else 0
    try:
        return validate_translation_loss_items(value, stanza_number)
    except SchemaValidationError as exc:
        raise PatchValueError(PatchValueIssue("invalid_translation_loss", path, str(exc))) from exc


_SIMPLE_STRING_FIELDS = frozenset({
    "gloss", "theme", "vehicle", "tenor", "literal_meaning",
    "literalization_risk", "visualization_strategy",
    "translation_status", "cultural_specificity_level", "visualization_difficulty",
})
_STRING_LIST_FIELDS = frozenset({"visual_features", "acceptable_visual_variants", "negative_confusions"})


def validate_patch_value(path: str, value: Any) -> Any:
    """Validate `value` against the canonical schema rules for the field
    named by `path`'s final segment (Task 5). Returns the normalized value
    on success; raises PatchValueError otherwise. Grounding-sensitive fields
    (source_span_original/source_span_translation/line_ref) are only
    syntax/type-checked here — their textual grounding is verified
    separately, at the whole-transaction level, in `apply_patch_v1_1`."""
    leaf = leaf_field(path)

    if leaf in _SIMPLE_STRING_FIELDS:
        return _optional_nonempty_string(value, path)
    if leaf in _STRING_LIST_FIELDS:
        return _nonempty_string_list(value, path)
    if leaf == "expression_type":
        return _enum_or_none(value, path, ALLOWED_EXPRESSION_TYPES)
    if leaf == "visual_priority":
        return _enum_or_none(value, path, ALLOWED_VISUAL_PRIORITIES)
    if leaf == "metaphor_mapping":
        return _metaphor_mapping_or_none(value, path)
    if leaf == "translation_loss":
        return _translation_loss_list(value, path)
    if leaf in ("source_span_original", "source_span_translation"):
        return _optional_nonempty_string(value, path)
    if leaf == "line_ref":
        return _line_ref_or_none(value, path)

    raise PatchValueError(PatchValueIssue("unsupported_target_field", path, f"no type validator is defined for field {leaf!r}."))


# ══════════════════════════════════════════════════════════════════════════
# Task 6 — patch application transaction
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TransactionResult:
    accepted: bool
    patched_candidate: dict | None
    applied_paths: tuple[str, ...]
    rejected_paths: tuple[str, ...]
    validation_result: str
    grounding_result: dict
    rejection_reason: str | None = None


def _poem_from_candidate(candidate: dict) -> PreprocessedPoem:
    stanzas = [
        StanzaInput(s["stanza_index"], s["line_count"], s["source_lines"], s["translated_lines"])
        for s in candidate["preprocessing"]["stanzas"]
    ]
    return PreprocessedPoem(
        candidate["poem_id"], candidate.get("poem_title", ""), candidate.get("language", ""),
        candidate["original_poem"], candidate["translated_poem"], stanzas,
        len(stanzas), len(stanzas), "aligned", 1.0, "",
    )


def _run_grounding(annotation: dict, original_poem: str, translated_poem: str) -> tuple[int, int]:
    """Re-run Stage 3 transitional grounding over the whole patched
    annotation. Returns (error_count, review_count). Read-only."""
    errors = 0
    reviews = 0
    for index, entity in enumerate(annotation.get("cultural_entities", [])):
        for issue in validate_cultural_grounding_v1_1(
            entity, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
        ):
            if issue.severity == "error":
                errors += 1
            else:
                reviews += 1
    for stanza in annotation.get("stanzas", []):
        for index, expr in enumerate(stanza.get("metaphor_spans", [])):
            for issue in validate_figurative_grounding_v1_1(
                expr, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
            ):
                if issue.severity == "error":
                    errors += 1
                else:
                    reviews += 1
    return errors, reviews


def apply_patch_v1_1(
    candidate: dict,
    patch_document: dict,
    approved_requested_paths: "set[str] | frozenset[str]",
    original_poem: str,
    translated_poem: str,
) -> TransactionResult:
    """Apply a validated patch to a DEEP COPY of `candidate`. Never modifies
    `candidate` itself. Never partially applies: every check (format, path
    resolution, value protection, type validation, Stage 2 structural
    validation, Stage 3 grounding) must pass before any value is written to
    the copy; the first failure rejects the whole transaction and returns
    `patched_candidate=None`."""
    poem_id = candidate.get("poem_id")

    try:
        patches = validate_patch_document(
            patch_document, expected_poem_id=poem_id, approved_requested_paths=set(approved_requested_paths),
        )
    except PatchFormatError as exc:
        return TransactionResult(
            accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=(),
            validation_result="not_attempted", grounding_result={},
            rejection_reason=f"patch_format:{exc.issue.code}: {exc.issue.message}",
        )

    requested_paths = tuple(sorted(patches))
    candidate_copy = copy.deepcopy(candidate)
    normalized_values: dict[str, Any] = {}

    for path, raw_value in patches.items():
        try:
            current = get_value_at_path(candidate_copy, path)
        except PatchPathError as exc:
            return TransactionResult(
                accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=requested_paths,
                validation_result="not_attempted", grounding_result={},
                rejection_reason=f"path:{exc.issue.code}: {exc.issue.message}",
            )
        if not is_approved_migration_placeholder(current):
            return TransactionResult(
                accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=requested_paths,
                validation_result="not_attempted", grounding_result={},
                rejection_reason=f"value_protection: {path} is not an approved migration placeholder (current value is not null/[]).",
            )
        try:
            normalized_values[path] = validate_patch_value(path, raw_value)
        except PatchValueError as exc:
            return TransactionResult(
                accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=requested_paths,
                validation_result="not_attempted", grounding_result={},
                rejection_reason=f"type_validation:{exc.issue.code}: {exc.issue.message}",
            )

    # All checks passed for every path — now, and only now, mutate the copy.
    for path, value in normalized_values.items():
        set_value_at_path(candidate_copy, path, value)

    poem = _poem_from_candidate(candidate_copy)
    try:
        validated_annotation = validate_model_payload_v1_1(candidate_copy["annotation"], poem)
    except ModelValidationError as exc:
        return TransactionResult(
            accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=requested_paths,
            validation_result=f"invalid: {exc}", grounding_result={},
            rejection_reason="structural_validation_failed",
        )
    candidate_copy["annotation"] = validated_annotation

    grounding_errors, grounding_reviews = _run_grounding(validated_annotation, original_poem, translated_poem)
    if grounding_errors:
        return TransactionResult(
            accepted=False, patched_candidate=None, applied_paths=(), rejected_paths=requested_paths,
            validation_result="valid", grounding_result={"errors": grounding_errors, "reviews": grounding_reviews},
            rejection_reason="ungrounded_field",
        )

    return TransactionResult(
        accepted=True, patched_candidate=candidate_copy, applied_paths=requested_paths, rejected_paths=(),
        validation_result="valid", grounding_result={"errors": 0, "reviews": grounding_reviews},
        rejection_reason=None,
    )


# ══════════════════════════════════════════════════════════════════════════
# Task 7 — raw vs. parsed provider response records (data shape only)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ProviderResponseRecord:
    """Provider-neutral record of one (future) backfill request/response.
    Nothing in this module ever populates one of these from a real call —
    it exists so a later stage has a defined shape to fill in. A record's
    mere existence never implies silver or gold status; `patch_application_status`
    only ever reaches "applied" via `apply_patch_v1_1` succeeding."""
    poem_id: str
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    requested_paths: tuple[str, ...] = field(default_factory=tuple)
    raw_response_text: str | None = None
    parsed_patch: dict | None = None
    parse_status: str = "not_attempted"
    patch_validation_status: str = "not_attempted"
    patch_application_status: str = "not_attempted"
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    generated_at: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
