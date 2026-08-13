"""Stage 5D — production-safe Gemini (Vertex, via google-genai) execution
layer for exactly one approved Stage 5C smoke batch (MV++_1153_batch_01).

Scope discipline: this module is the ONLY place in the pipeline that imports
a Gemini/Vertex provider SDK. It never imports the deprecated
`vertexai.generative_models` module or `google-cloud-aiplatform`'s generative
namespaces. It never constructs a provider client at import time — every
client is created lazily, inside a function, and only ever via dependency
injection in tests (a fake client is passed in; no test in this project
makes a real network call). Credentials are read only through Application
Default Credentials (`google.auth.default()`); no API key path is
implemented here, and no credential value is ever printed, logged, or
serialized to an artifact file.

Only one batch is authorized in this stage: `AUTHORIZED_BATCH_ID`. Every
public entry point re-validates this — there is no code path by which a
caller can select a different Stage 5C batch.

See docs/GEMINI_SMOKE_EXECUTION_STAGE5D.md for the full contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from . import execution_batch_v1_1 as eb
from . import patch_v1_1 as pv
from .models import ModelValidationError, validate_model_payload_v1_1
from .grounding import (
    validate_cultural_grounding_v1_1,
    validate_figurative_grounding_v1_1,
    GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
)
from .patch_v1_1 import (
    PatchFormatError,
    apply_patch_v1_1,
    validate_patch_document,
)
from .prompt_v1_1 import PromptBundle
from .schema import (
    MORPHOVERSE_SCHEMA_VERSION,
    ALLOWED_EXPRESSION_TYPES,
    ALLOWED_VISUAL_PRIORITIES,
)

STAGE = "5D_smoke_execution"

# ══════════════════════════════════════════════════════════════════════════
# Authorization constants (Task: single-batch smoke authorization)
# ══════════════════════════════════════════════════════════════════════════
AUTHORIZED_BATCH_ID = "MV++_1153_batch_01"
MAX_LIVE_ATTEMPTS = 3
BACKOFF_SCHEDULE_SECONDS: tuple[int, ...] = (2, 4, 8)
RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
GENERATION_TEMPERATURE = 0
MAX_OUTPUT_TOKENS_SMOKE = 4096
API_VERSION = "v1"
PROVIDER_NAME = "google_genai_vertex"

STAGE5C_DIR = Path("pilot") / "backfill_requests" / "stage5c"
STAGE5D_RUNS_DIR = Path("pilot") / "provider_runs" / "stage5d"
SMOKE_CANDIDATE_DIR = Path("pilot") / "annotations_v1_1" / "backfill_smoke"
PRE_BACKFILL_DIR = Path("pilot") / "annotations_v1_1" / "pre_backfill"

STATUS_PLANNED_NOT_EXECUTED = "planned_not_executed"
SMOKE_CANDIDATE_STATUS = "partially_backfilled_smoke_candidate"

# ══════════════════════════════════════════════════════════════════════════
# Stage 5D.1 Task 1 — confirmed root cause of the Stage 5D live-call failure
# ══════════════════════════════════════════════════════════════════════════
# The original PATCH_RESPONSE_JSON_SCHEMA's `patches.items.properties` did not
# declare a "value" property at all (only "path" had a type) — "value" was
# merely named in `required`, with no schema node backing it. Vertex's
# response_json_schema validator rejects a required property with no declared
# type/anyOf/enum ("schema at properties.patches.items requires unspecified
# property 'value'"), a 400 INVALID_ARGUMENT returned before any generation
# occurred. This is a REQUEST-construction defect only: it is unrelated to
# patch parsing, Stage 5B local validation, Stage 3 grounding, ADC/credential
# resolution, or candidate application — none of that code ever ran, because
# the request itself was never accepted. See docs/GEMINI_SMOKE_EXECUTION_STAGE5D.md.
#
# Stage 5D.1 Task 2 — batch-derived response schema (every property typed)
# ══════════════════════════════════════════════════════════════════════════
# Each of these is a single, reusable JSON-Schema "shape" for one leaf field
# category (Task 2, options A-F). None is ever empty ({}), and none is ever
# used un-typed. `null` is intentionally NOT nested inside each shape — every
# field this batch format can request is optional/nullable at the Stage 5B
# validation layer (patch_v1_1.validate_patch_value), so a single shared
# "null" branch in the top-level value union (see build_patch_response_json_schema)
# covers every field's nullability without repeating {"anyOf": [X, null]} in
# every branch (avoids the "nesting redundant anyOf blocks" Task 2 warns
# against).
_NONEMPTY_STRING_SHAPE: dict[str, Any] = {"type": "string", "minLength": 1}
_STRING_LIST_SHAPE: dict[str, Any] = {
    "type": "array",
    "items": {"type": "string", "minLength": 1},
}
# Task 2.D — metaphor_mapping's exact shape, matching
# models.METAPHOR_MAPPING_KEYS / models.validate_metaphor_mapping_v1_1.
_METAPHOR_MAPPING_SHAPE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["vehicle_concept", "tenor_concept", "transferred_attributes"],
    "properties": {
        "vehicle_concept": {"type": "string", "minLength": 1},
        "tenor_concept": {"type": "string", "minLength": 1},
        "transferred_attributes": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}
# Task 2.E — translation_loss's exact item shape, matching
# schema.TRANSLATION_LOSS_KEYS / models.validate_translation_loss_items (only
# "what_was_lost" is required there; "where"/"severity" are optional/nullable
# strings — no second semantic definition is invented here).
_TRANSLATION_LOSS_ITEM_SHAPE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["what_was_lost"],
    "properties": {
        "what_was_lost": {"type": "string", "minLength": 1},
        "where": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
        "severity": {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]},
    },
}
_TRANSLATION_LOSS_SHAPE: dict[str, Any] = {
    "type": "array",
    "items": _TRANSLATION_LOSS_ITEM_SHAPE,
}
_NULL_SHAPE: dict[str, Any] = {"type": "null"}


def _expression_type_shape() -> dict[str, Any]:
    # Task 2.F — pulled directly from schema.py, never a hand-copied list.
    return {"type": "string", "enum": list(ALLOWED_EXPRESSION_TYPES)}


def _visual_priority_shape() -> dict[str, Any]:
    return {"type": "string", "enum": list(ALLOWED_VISUAL_PRIORITIES)}


# Leaf field name -> (shape key, shape builder). The shape key is used only to
# de-duplicate identical branches when several requested fields share one
# shape (e.g. every plain free-text field maps to "string"). This mirrors
# patch_v1_1's own field classification (_SIMPLE_STRING_FIELDS/_STRING_LIST_FIELDS)
# rather than re-deriving a second, possibly-drifting definition.
def _leaf_value_shape(leaf: str) -> tuple[str, dict[str, Any]]:
    if leaf in pv._SIMPLE_STRING_FIELDS or leaf in ("source_span_original", "source_span_translation"):
        return "string", _NONEMPTY_STRING_SHAPE
    if leaf in pv._STRING_LIST_FIELDS:
        return "string_list", _STRING_LIST_SHAPE
    if leaf == "expression_type":
        return "expression_type_enum", _expression_type_shape()
    if leaf == "visual_priority":
        return "visual_priority_enum", _visual_priority_shape()
    if leaf == "metaphor_mapping":
        return "metaphor_mapping", _METAPHOR_MAPPING_SHAPE
    if leaf == "translation_loss":
        return "translation_loss", _TRANSLATION_LOSS_SHAPE
    raise ValueError(f"no provider response schema shape is defined for field {leaf!r}.")


def build_patch_response_json_schema(
    poem_id: str,
    requested_field_paths: "list[str] | tuple[str, ...]",
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Pure helper (Task 2): every property below has an explicit type,
    anyOf, or enum — no empty `{}` schema, no unspecified-type property.
    `value`'s shape is a batch-level union over only the value shapes this
    batch's own requested fields actually need (deduplicated), because a
    provider JSON Schema cannot make one array item's "value" type depend on
    that same item's own "path" value (Stage 5B's `validate_patch_value`
    remains the authoritative, per-path type check after parsing — this
    schema is an output-shape aid only, never a replacement for it)."""
    if candidate.get("poem_id") != poem_id:
        raise ValueError(f"candidate poem_id {candidate.get('poem_id')!r} does not match {poem_id!r}.")
    paths = list(requested_field_paths)
    if not paths:
        raise ValueError("requested_field_paths must be non-empty.")
    if len(paths) != len(set(paths)):
        raise ValueError("requested_field_paths contains duplicate entries.")

    branches: dict[str, dict[str, Any]] = {}
    for path in paths:
        leaf = pv.leaf_field(path)
        shape_key, shape = _leaf_value_shape(leaf)
        branches.setdefault(shape_key, copy.deepcopy(shape))

    value_any_of = [copy.deepcopy(_NULL_SHAPE)] + [branches[key] for key in sorted(branches)]

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "poem_id", "patches"],
        "properties": {
            "schema_version": {"type": "string", "enum": [MORPHOVERSE_SCHEMA_VERSION]},
            "poem_id": {"type": "string", "enum": [poem_id]},
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["path", "value"],
                    "properties": {
                        "path": {"type": "string", "enum": sorted(paths)},
                        "value": {"anyOf": value_any_of},
                    },
                },
            },
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# Stage 5D.1 Task 3 — pure recursive schema self-audit (never network-facing)
# ══════════════════════════════════════════════════════════════════════════
def _audit_schema_node(node: Any, *, node_path: str, errors: list[str]) -> None:
    if not isinstance(node, dict):
        return
    if node == {}:
        errors.append(f"{node_path}: empty schema dict is not permitted.")
        return
    for key, value in node.items():
        if callable(value):
            errors.append(f"{node_path}.{key}: schema must not contain a callable/executable value.")
    has_type_marker = "type" in node or "anyOf" in node or "enum" in node
    if not has_type_marker:
        errors.append(f"{node_path}: property has no type, anyOf, or enum.")
    if "anyOf" in node:
        if not isinstance(node["anyOf"], list) or not node["anyOf"]:
            errors.append(f"{node_path}.anyOf: must be a non-empty list.")
        else:
            for i, branch in enumerate(node["anyOf"]):
                _audit_schema_node(branch, node_path=f"{node_path}.anyOf[{i}]", errors=errors)
    if node.get("type") == "object" and isinstance(node.get("properties"), dict):
        for prop_name, prop_schema in node["properties"].items():
            _audit_schema_node(prop_schema, node_path=f"{node_path}.properties.{prop_name}", errors=errors)
    if node.get("type") == "array" and "items" in node:
        _audit_schema_node(node["items"], node_path=f"{node_path}.items", errors=errors)


def audit_provider_response_schema(
    response_schema: dict[str, Any], *, expected_paths: "list[str] | tuple[str, ...] | frozenset[str]"
) -> tuple[bool, tuple[str, ...]]:
    """Task 3: pure, offline, recursive audit. Rejects empty schema dicts, any
    property without type/anyOf/enum, a missing `value`/`path` requirement, a
    callable/executable value anywhere in the schema, a duplicate path-enum
    entry, and a path enum that differs from the authorized batch's own
    requested paths. Returns (passed, error_messages) — never raises."""
    errors: list[str] = []
    _audit_schema_node(response_schema, node_path="$", errors=errors)

    try:
        item_schema = response_schema["properties"]["patches"]["items"]
    except (KeyError, TypeError):
        errors.append("$.properties.patches.items: missing.")
        item_schema = None

    if item_schema is not None:
        required = item_schema.get("required", [])
        if "value" not in required:
            errors.append("$.properties.patches.items.required: 'value' is missing.")
        if "path" not in required:
            errors.append("$.properties.patches.items.required: 'path' is missing.")

        path_enum = item_schema.get("properties", {}).get("path", {}).get("enum")
        if not path_enum:
            errors.append("$.properties.patches.items.properties.path.enum: missing or empty.")
        else:
            if len(path_enum) != len(set(path_enum)):
                errors.append("$.properties.patches.items.properties.path.enum: contains duplicate entries.")
            if set(path_enum) != set(expected_paths):
                errors.append(
                    "$.properties.patches.items.properties.path.enum: does not match the "
                    "authorized batch's requested paths exactly."
                )

        value_schema = item_schema.get("properties", {}).get("value")
        if not value_schema or not (value_schema.get("anyOf")):
            errors.append("$.properties.patches.items.properties.value: missing a non-empty anyOf.")

    return (len(errors) == 0, tuple(errors))


# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════
class ConfigError(RuntimeError):
    """Raised when required non-secret Gemini configuration is missing."""


class BatchAuthorizationError(ValueError):
    """Raised when a caller requests any batch other than AUTHORIZED_BATCH_ID,
    or requests the authorized batch when it is not planned_not_executed."""


# ══════════════════════════════════════════════════════════════════════════
# Task 2 — configuration contract (non-secret; ADC-only authentication)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class GeminiConfig:
    project: str
    location: str
    model: str


def load_gemini_config() -> GeminiConfig:
    """Read the three required non-secret settings from config.py (which
    itself only reads environment variables — see config.py's own Stage 1
    hardening). Raises ConfigError naming exactly which variable(s) are
    missing; never raises for a missing credential (that is ADC's concern,
    checked separately by check_adc_available())."""
    project = cfg.GOOGLE_CLOUD_PROJECT
    location = cfg.GOOGLE_CLOUD_LOCATION
    model = cfg.VERTEX_GEMINI_MODEL
    missing = [
        name
        for name, value in (
            ("GOOGLE_CLOUD_PROJECT", project),
            ("GOOGLE_CLOUD_LOCATION", location),
            ("VERTEX_GEMINI_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ConfigError(
            "Missing required Gemini configuration: "
            + ", ".join(missing)
            + ". Set these environment variables before running the Stage 5D executor."
        )
    return GeminiConfig(project=project, location=location, model=model)


def check_adc_available() -> bool:
    """True if Application Default Credentials can be loaded. Never raises,
    never returns or logs the credential itself — only a boolean."""
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    try:
        google.auth.default()
        return True
    except DefaultCredentialsError:
        return False


def gemini_safe_config_summary() -> dict[str, Any]:
    """Safe-to-print configuration summary (Task 2). Built from an explicit
    allowlist of fields, exactly like config.safe_config_summary() — never
    includes a credential path, token, or key material of any kind."""
    try:
        gemini_config = load_gemini_config()
        project, location, model = gemini_config.project, gemini_config.location, gemini_config.model
    except ConfigError:
        project = location = model = None
    return {
        "project": project,
        "region": location,
        "model": model,
        "authentication_mode": "adc",
        "credential_available": check_adc_available(),
    }


# ══════════════════════════════════════════════════════════════════════════
# Task 1 / Task 3 — client factory (dependency-injected; no import-time client)
# ══════════════════════════════════════════════════════════════════════════
def default_client_factory() -> Any:
    """Construct a real google-genai Client in Vertex/enterprise mode using
    ADC. Only ever called from the live-execution path — never at module
    import time, and never from any test in this suite."""
    from google import genai
    import google.auth

    gemini_config = load_gemini_config()
    credentials, _ = google.auth.default()
    return genai.Client(
        enterprise=True,
        project=gemini_config.project,
        location=gemini_config.location,
        credentials=credentials,
    )


ClientFactory = Callable[[], Any]


# ══════════════════════════════════════════════════════════════════════════
# Batch / candidate loading (re-validates authorization on every call)
# ══════════════════════════════════════════════════════════════════════════
def _require_authorized_batch_id(batch_id: str) -> None:
    if batch_id != AUTHORIZED_BATCH_ID:
        raise BatchAuthorizationError(
            f"batch_id must be exactly {AUTHORIZED_BATCH_ID!r} in Stage 5D; got {batch_id!r}."
        )


def load_batch(batch_id: str, repo_root: Path) -> dict[str, Any]:
    """Load and re-validate the one authorized Stage 5C batch file. Raises
    BatchAuthorizationError for any other batch id, or if the authorized
    batch is not (still) planned_not_executed."""
    _require_authorized_batch_id(batch_id)
    path = repo_root / STAGE5C_DIR / f"{batch_id}.json"
    with path.open("r", encoding="utf-8") as f:
        batch = json.load(f)
    if batch.get("poem_id") != "MV++_1153":
        raise BatchAuthorizationError(f"batch {batch_id!r} does not belong to MV++_1153.")
    if batch.get("execution_status") != STATUS_PLANNED_NOT_EXECUTED:
        raise BatchAuthorizationError(
            f"batch {batch_id!r} is not planned_not_executed (found {batch.get('execution_status')!r})."
        )
    return batch


def load_candidate(batch: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path = repo_root / batch["source_candidate_path"]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def output_dir_for(batch_id: str, repo_root: Path) -> Path:
    _require_authorized_batch_id(batch_id)
    return repo_root / STAGE5D_RUNS_DIR / batch_id


# ══════════════════════════════════════════════════════════════════════════
# Stage 5D.1 Task 5 — failed-attempt (evidence) immutability
# ══════════════════════════════════════════════════════════════════════════
# Stage 5D's one live call wrote flat files directly under output_dir_for(...)
# (request_metadata.json, prompt_request.json, run_summary.json). Those files
# are now evidence of attempt 1 and must never be overwritten by a future
# attempt — including a future FAILED attempt, not only a future success.
# Compatibility approach (documented, per Stage 5D.1 Task 5): the existing
# flat root files are treated as legacy "attempt_01"; every future execution
# — regardless of this run's own outcome — is written under a NEW,
# never-before-used "attempt_NN" subdirectory, computed but never created by
# these pure functions. No attempt_02 directory is created by this stage.
ATTEMPT_DIR_PREFIX = "attempt_"


def _has_legacy_root_attempt(out_dir: Path) -> bool:
    return any(
        (out_dir / name).exists()
        for name in ("run_summary.json", "request_metadata.json", "prompt_request.json")
    )


def _existing_attempt_numbers(out_dir: Path) -> list[int]:
    if not out_dir.exists():
        return []
    numbers = []
    for child in out_dir.iterdir():
        if child.is_dir() and child.name.startswith(ATTEMPT_DIR_PREFIX):
            suffix = child.name[len(ATTEMPT_DIR_PREFIX):]
            if suffix.isdigit():
                numbers.append(int(suffix))
    return sorted(numbers)


def next_attempt_dir(out_dir: Path) -> Path:
    """Pure path computation — never creates anything on disk. Always
    returns a NEW, not-yet-existing attempt_NN directory, so a future
    execution (success or failure) can never silently overwrite a prior
    attempt's evidence. The legacy flat root files (if present) count as
    attempt_01 without being moved or touched."""
    numbered = _existing_attempt_numbers(out_dir)
    baseline = 1 if _has_legacy_root_attempt(out_dir) else 0
    highest = max(numbered + [baseline])
    return out_dir / f"{ATTEMPT_DIR_PREFIX}{highest + 1:02d}"


def latest_attempt_run_summary(out_dir: Path) -> dict[str, Any] | None:
    """Read the most recent attempt's run_summary.json, whether that is the
    legacy flat root (attempt_01) or a numbered attempt_NN directory.
    Read-only; returns None if no attempt has ever been recorded."""
    numbered = _existing_attempt_numbers(out_dir)
    if numbered:
        candidate_path = out_dir / f"{ATTEMPT_DIR_PREFIX}{max(numbered):02d}" / "run_summary.json"
    elif _has_legacy_root_attempt(out_dir):
        candidate_path = out_dir / "run_summary.json"
    else:
        return None
    if not candidate_path.exists():
        return None
    with candidate_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Atomic artifact writes (Task 6) — never write credentials/auth headers
# ══════════════════════════════════════════════════════════════════════════
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def prompt_sha256(system_prompt: str, user_prompt: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(b"\n---\n")
    hasher.update(user_prompt.encode("utf-8"))
    return hasher.hexdigest()


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — request construction (pure; no network access)
# ══════════════════════════════════════════════════════════════════════════
def build_generation_config(
    system_prompt: str, response_json_schema: dict[str, Any], *, max_output_tokens: int = MAX_OUTPUT_TOKENS_SMOKE,
) -> Any:
    """Deterministic, JSON-only, tool-free generation config (Task 3/4).
    `response_json_schema` must come from build_patch_response_json_schema —
    every property in it is explicitly typed (Stage 5D.1's fix).
    `max_output_tokens` defaults to the original Stage 5D smoke value
    (unchanged behavior for any existing caller); Stage 5E.1 callers pass an
    adaptive value from `determine_output_token_budget` instead.

    Stage 5E.15: explicitly requests `thinking_level="minimal"` — never a
    numeric `thinking_budget`, and never both controls together (the SDK's
    own `ThinkingConfig` only exposes one `thinking_level` field here).
    These annotation/backfill requests are constrained extraction and
    structured-annotation tasks, not open-ended reasoning tasks: the
    model's own JSON output (bounded by `max_output_tokens`) is the
    complete, authoritative product, and every returned value is still
    independently re-validated by this project's own schema/path/type/
    grounding validators regardless of how much (or how little) the model
    "thought" before producing it. Minimal thinking is an operational
    execution policy chosen to preserve output-token capacity for complete
    JSON (Stage 5E.14's MV++_0073_batch_05 failure consumed 3929 of 4096
    tokens on thinking, leaving only 151 for visible output) — it is not a
    claim about annotation quality, and it applies uniformly to every
    structured annotation/backfill request this function builds, never
    keyed on poem ID or batch ID."""
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=GENERATION_TEMPERATURE,
        candidate_count=1,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=response_json_schema,
        tools=None,
        thinking_config=types.ThinkingConfig(thinking_level="minimal"),
    )


def build_request(
    bundle: PromptBundle,
    model: str,
    *,
    poem_id: str,
    requested_field_paths: "list[str] | tuple[str, ...]",
    candidate: dict[str, Any],
    max_output_tokens: int = MAX_OUTPUT_TOKENS_SMOKE,
) -> dict[str, Any]:
    """Build the exact keyword arguments for client.models.generate_content.
    Pure — makes no call itself. The response schema is derived fresh from
    this batch's own requested fields (Stage 5D.1 Task 2), not a static
    constant. `max_output_tokens` defaults to the Stage 5D smoke value;
    unchanged for any existing caller that doesn't pass it."""
    response_schema = build_patch_response_json_schema(poem_id, requested_field_paths, candidate)
    return {
        "model": model,
        "contents": bundle.user_prompt,
        "config": build_generation_config(bundle.system_prompt, response_schema, max_output_tokens=max_output_tokens),
    }


def generation_settings_summary(*, max_output_tokens: int = MAX_OUTPUT_TOKENS_SMOKE) -> dict[str, Any]:
    return {
        "temperature": GENERATION_TEMPERATURE,
        "candidate_count": 1,
        "max_output_tokens": max_output_tokens,
        "response_mime_type": "application/json",
        "response_json_schema_used": True,
        "tools": None,
        # Stage 5E.15 Task 3 — safe, plain-value record of the thinking
        # policy actually configured on the request; never the full SDK
        # ThinkingConfig object, never a credential, header, or token.
        "thinking_level": "minimal",
        "thinking_explicitly_configured": True,
    }


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.1 Task 3 — adaptive output-token budgeting
# (31-40 tier raised 12288 -> 24576 in Stage 5E.11)
# ══════════════════════════════════════════════════════════════════════════
# Conservative, provider-neutral OPERATIONAL SAFETY DEFAULTS — not a
# scientific measurement of any model's true capacity, and not a claim
# about how many tokens any given annotation "requires." They exist only to
# give the provider enough room to avoid an avoidable MAX_TOKENS truncation
# for a batch of a given size; a batch that still hits MAX_TOKENS at its
# tier's ceiling is a genuine provider-behavior signal to report and
# investigate (Stage 5E.11's MV++_0073_batch_04), not evidence the ceiling
# was miscalculated. Never derived from poem_id. A Stage 5C batch's own JSON
# output size scales with how many fields it requests, not with which poem
# it belongs to, so the budget is tiered purely by requested-path count.
# Ceilings intentionally align with Stage 5C's own packing policy
# (execution_batch_v1_1.MAX_PATHS_PER_BATCH = 40), so every batch Stage 5C
# could ever produce has a defined tier — and the tier is a hard ceiling,
# never an unbounded fallback: a path count above 40 is still rejected
# outright below, not silently granted more headroom.
TOKEN_BUDGET_TIERS: "tuple[tuple[int, int], ...]" = (
    (20, MAX_OUTPUT_TOKENS_SMOKE),  # 1-20 paths -> 4096 (unchanged from Stage 5D's smoke value)
    (30, 8192),                      # 21-30 paths -> 8192
    (40, 24576),                     # 31-40 paths -> 24576 (raised from 12288 in Stage 5E.11 after
                                      # MV++_0073_batch_04's confirmed MAX_TOKENS finish at the old ceiling)
)


def determine_output_token_budget(
    requested_field_paths: "list[str] | tuple[str, ...]",
    semantic_units: "list | tuple | dict | None" = None,
) -> int:
    """Pure, offline, provider-neutral output-token budget for one Stage 5C
    batch (Task 3). Derived only from how many fields the batch itself
    requests — never from poem_id, never from a network/token-count call.
    `semantic_units` (the batch's own semantic_unit_types, if available) is
    accepted for interface completeness with how a batch is actually
    described, but the tiering rule below depends only on path count: Stage
    5C's own unit-packing policy already keeps a batch's unit count and path
    count in the same ballpark for every real batch, so no second axis is
    needed. Rejects zero paths (nothing to budget for) and rejects a path
    count above the Stage 5C safety maximum — a batch that large should have
    been split further upstream, never silently granted an unbounded
    budget here."""
    count = len(requested_field_paths)
    if count == 0:
        raise ValueError("determine_output_token_budget: requested_field_paths must be non-empty.")
    if count > eb.MAX_PATHS_PER_BATCH:
        raise ValueError(
            f"determine_output_token_budget: {count} requested paths exceeds the Stage 5C safety "
            f"maximum of {eb.MAX_PATHS_PER_BATCH}; this batch should have been split upstream rather "
            "than assigned an unbounded token budget."
        )
    for ceiling, budget in TOKEN_BUDGET_TIERS:
        if count <= ceiling:
            return budget
    raise AssertionError("unreachable: count already validated <= MAX_PATHS_PER_BATCH")  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.1 Task 4 — safe, tolerant provider completion metadata
# ══════════════════════════════════════════════════════════════════════════
_SENSITIVE_MESSAGE_MARKERS = ("begin private key", "service_account", "refresh_token", "authorization:", "bearer ")


def _safe_str(value: Any) -> "str | None":
    """Converts an enum-like SDK value (e.g. a FinishReason member) to a
    plain string safely — prefers `.name` when present (the usual shape for
    a google-genai enum), falls back to `str()`, never raises, never
    returns anything but a plain str or None."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name) if name is not None else str(value)


def _sanitize_finish_message(message: Any) -> "str | None":
    """Never serializes a credential/auth-header-shaped string; truncates an
    otherwise-ordinary message to a safe length. The provider's own
    finish-message text is not expected to contain secrets, but this is
    defense in depth, matching the same markers this project's other
    artifact-safety tests already check for."""
    if message is None:
        return None
    text = str(message)
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MESSAGE_MARKERS):
        return "[redacted: message contained a sensitive-looking marker]"
    return text[:500]


# Stage 5E.11 Task 2/4: tolerant, multi-name attribute lookup for fields
# whose SDK attribute name has varied across google-genai versions/response
# shapes. Tries each name in order and returns the first value that is
# actually an int (bool excluded, since bool is an int subclass in Python)
# — never coerces a non-int value, never fabricates a count when the SDK
# genuinely didn't expose one.
_OUTPUT_TOKEN_COUNT_ATTRS = ("candidates_token_count", "output_token_count")
_THOUGHTS_TOKEN_COUNT_ATTRS = ("thoughts_token_count", "thought_token_count", "reasoning_token_count")


def _first_present_int(obj: Any, attr_names: "tuple[str, ...]") -> "int | None":
    if obj is None:
        return None
    for name in attr_names:
        value = getattr(obj, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def extract_completion_metadata(response: Any) -> dict[str, Any]:
    """Best-effort, tolerant extraction of provider completion metadata
    (Task 4; extended Stage 5E.11 Task 2/4 with a safe thoughts/reasoning
    token count). Never raises for a missing/renamed SDK attribute — every
    field defaults to None when unavailable, so a valid response is never
    failed merely because usage metadata wasn't exposed. Never serializes
    the full provider response object, a credential, an auth header, or
    account information. Thoughts-token usage is never estimated or
    inferred — it is either a genuine int the SDK exposed, or None."""
    finish_reason = None
    finish_message = None
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        first = candidates[0]
        finish_reason = _safe_str(getattr(first, "finish_reason", None))
        finish_message = _sanitize_finish_message(getattr(first, "finish_message", None))

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
    output_tokens = _first_present_int(usage, _OUTPUT_TOKEN_COUNT_ATTRS)
    total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None
    cached_tokens = getattr(usage, "cached_content_token_count", None) if usage is not None else None
    thoughts_tokens = _first_present_int(usage, _THOUGHTS_TOKEN_COUNT_ATTRS)

    response_id = _safe_str(getattr(response, "response_id", None))

    return {
        "finish_reason": finish_reason,
        "finish_message": finish_message,
        "prompt_token_count": prompt_tokens,
        "output_token_count": output_tokens,
        "total_token_count": total_tokens,
        "cached_token_count": cached_tokens,
        "thoughts_token_count": thoughts_tokens,
        "response_id": response_id,
    }


_SAFETY_FINISH_REASONS = frozenset({"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"})


def classify_completion(*, provider_call_succeeded: bool, finish_reason: "str | None") -> str:
    """Pure classification into completed/max_tokens/safety/malformed_or_unknown/
    provider_error (Task 4/5). NEVER infers 'max_tokens' (or any other
    classification) from a local JSON parse failure — only from a
    provider-reported finish_reason, and only ever 'provider_error' when the
    provider call itself did not succeed."""
    if not provider_call_succeeded:
        return "provider_error"
    if finish_reason is None:
        return "malformed_or_unknown"
    normalized = finish_reason.upper()
    if normalized == "STOP":
        return "completed"
    if normalized == "MAX_TOKENS":
        return "max_tokens"
    if normalized in _SAFETY_FINISH_REASONS:
        return "safety"
    return "malformed_or_unknown"


# ══════════════════════════════════════════════════════════════════════════
# Task 4 — retry and error classification policy
# ══════════════════════════════════════════════════════════════════════════
def classify_api_error(exc: BaseException) -> str:
    """Returns "retryable" or "non_retryable". Never retries an
    authentication/permission failure, invalid project/location/model,
    malformed request, or safety refusal — only a small, explicit set of
    transient HTTP/connection failures."""
    from google.genai import errors

    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        return "retryable" if code in RETRYABLE_HTTP_CODES else "non_retryable"
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "retryable"
    return "non_retryable"


@dataclass(frozen=True)
class AttemptRecord:
    attempt_number: int
    outcome: str  # "success" | "retryable_error" | "non_retryable_error"
    error_code: int | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class GenerationOutcome:
    success: bool
    response: Any | None
    attempts: tuple[AttemptRecord, ...]
    final_error: str | None


def generate_with_retry(
    client: Any,
    request: dict[str, Any],
    *,
    max_attempts: int = MAX_LIVE_ATTEMPTS,
    backoff_schedule: tuple[int, ...] = BACKOFF_SCHEDULE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> GenerationOutcome:
    """Call client.models.generate_content(**request), retrying only on a
    classified-retryable error, up to max_attempts total attempts, with the
    given backoff schedule injected (so tests never actually sleep)."""
    attempts: list[AttemptRecord] = []
    for attempt_number in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(**request)
        except Exception as exc:  # noqa: BLE001 - classified below, re-raised never
            classification = classify_api_error(exc)
            code = getattr(exc, "code", None)
            attempts.append(AttemptRecord(attempt_number, f"{classification}_error", code, str(exc)))
            if classification == "non_retryable" or attempt_number == max_attempts:
                return GenerationOutcome(False, None, tuple(attempts), str(exc))
            sleep_fn(backoff_schedule[min(attempt_number - 1, len(backoff_schedule) - 1)])
            continue
        attempts.append(AttemptRecord(attempt_number, "success"))
        return GenerationOutcome(True, response, tuple(attempts), None)
    return GenerationOutcome(False, None, tuple(attempts), "max attempts exhausted")  # pragma: no cover


# ══════════════════════════════════════════════════════════════════════════
# Task 7 — response parsing / validation (never repairs malformed JSON)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ParseResult:
    status: str  # "success" | "json_parse_failed" | "empty_response"
    parsed: dict[str, Any] | None
    error: str | None = None


def parse_raw_response(raw_text: str | None) -> ParseResult:
    if not raw_text or not raw_text.strip():
        return ParseResult(status="empty_response", parsed=None, error="response text was empty.")
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ParseResult(status="json_parse_failed", parsed=None, error=str(exc))
    if not isinstance(parsed, dict):
        return ParseResult(status="json_parse_failed", parsed=None, error="parsed JSON is not an object.")
    return ParseResult(status="success", parsed=parsed)


@dataclass(frozen=True)
class PatchValidationResult:
    status: str  # "accepted" | "rejected"
    reason: str | None
    missing_paths: tuple[str, ...] = field(default_factory=tuple)
    extra_paths: tuple[str, ...] = field(default_factory=tuple)


def validate_patch_matches_batch_exactly(
    parsed: dict[str, Any], *, expected_poem_id: str, approved_paths: frozenset[str]
) -> PatchValidationResult:
    """Task 7 step 4: patch paths must match the authorized batch exactly —
    no missing, no extra, no duplicate. validate_patch_document (Stage 5B)
    already rejects duplicates and any path outside approved_paths; this
    adds the "nothing missing" half of the exact-match requirement."""
    try:
        ordered = validate_patch_document(
            parsed, expected_poem_id=expected_poem_id, approved_requested_paths=set(approved_paths)
        )
    except PatchFormatError as exc:
        return PatchValidationResult(status="rejected", reason=f"patch_format:{exc.issue.code}: {exc.issue.message}")

    present = frozenset(ordered.keys())
    missing = tuple(sorted(approved_paths - present))
    extra = tuple(sorted(present - approved_paths))
    if missing or extra:
        return PatchValidationResult(
            status="rejected",
            reason=f"path_mismatch: missing={list(missing)} extra={list(extra)}",
            missing_paths=missing,
            extra_paths=extra,
        )
    return PatchValidationResult(status="accepted", reason=None)


# ══════════════════════════════════════════════════════════════════════════
# Task 8 — smoke backfilled candidate construction
# ══════════════════════════════════════════════════════════════════════════
def build_smoke_candidate(
    patched_candidate: dict[str, Any],
    *,
    batch_id: str,
    applied_paths: tuple[str, ...],
    run_timestamp: str,
) -> dict[str, Any]:
    """Deep-copy-safe: apply_patch_v1_1 already returned a fresh deep copy
    (patched_candidate), so this only sets envelope-level status/provenance
    fields, never re-touching `annotation` itself. Never calls this record
    silver, gold, human-reviewed, or complete."""
    smoke = copy.deepcopy(patched_candidate)
    smoke["status"] = SMOKE_CANDIDATE_STATUS
    smoke["stage5d_smoke_provenance"] = {
        "stage": STAGE,
        "batch_id": batch_id,
        "applied_paths": list(applied_paths),
        "provider": PROVIDER_NAME,
        "run_timestamp": run_timestamp,
        "note": (
            "Model-assisted smoke backfill candidate. Not silver, not gold, "
            "not human-reviewed. Reviewer availability for Kashmiri is not "
            "yet confirmed (pilot/PILOT_APPROVAL_CHECKLIST.md condition A)."
        ),
    }
    return smoke


def _poem_texts(candidate: dict[str, Any]) -> tuple[str, str]:
    return candidate["original_poem"], candidate["translated_poem"]


def revalidate_smoke_candidate_from_disk(path: Path) -> dict[str, Any]:
    """Task 8/13: reload the written smoke candidate and re-run Stage 2
    structural validation plus Stage 3 transitional grounding over it.
    Read-only; raises ModelValidationError on structural failure. Returns a
    small report dict (never raises for a grounding *review*-severity issue,
    only for an *error*-severity one, mirroring apply_patch_v1_1's own gate)."""
    with path.open("r", encoding="utf-8") as f:
        record = json.load(f)

    poem = pv._poem_from_candidate(record)  # reuses Stage 5B's own poem-shape helper
    validated_annotation = validate_model_payload_v1_1(record["annotation"], poem)

    original_poem, translated_poem = _poem_texts(record)
    errors = 0
    reviews = 0
    for index, entity in enumerate(validated_annotation.get("cultural_entities", [])):
        for issue in validate_cultural_grounding_v1_1(
            entity, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
        ):
            if issue.severity == "error":
                errors += 1
            else:
                reviews += 1
    for stanza in validated_annotation.get("stanzas", []):
        for index, expr in enumerate(stanza.get("metaphor_spans", [])):
            for issue in validate_figurative_grounding_v1_1(
                expr, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
            ):
                if issue.severity == "error":
                    errors += 1
                else:
                    reviews += 1

    return {"structural_validation": "valid", "grounding_errors": errors, "grounding_reviews": reviews}


# ══════════════════════════════════════════════════════════════════════════
# Task 5 — dry run (zero provider calls)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DryRunReport:
    batch_id: str
    poem_id: str
    requested_path_count: int
    system_prompt_chars: int
    user_prompt_chars: int
    output_dir: str
    next_attempt_dir: str
    response_schema_audit_passed: bool
    planned_calls: int
    actual_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dry_run(batch_id: str, repo_root: Path) -> DryRunReport:
    """Renders the prompt and reports safe size metadata. Makes ZERO
    provider/network calls — no client is even constructed. Also
    self-audits the corrected response schema (Stage 5D.1 Task 9) and
    reports where a future live attempt would write (never overwriting a
    prior attempt)."""
    batch = load_batch(batch_id, repo_root)
    candidate = load_candidate(batch, repo_root)
    bundle = eb.build_prompt_bundle_from_batch(batch, candidate)
    out_dir = output_dir_for(batch_id, repo_root)
    response_schema = build_patch_response_json_schema(
        batch["poem_id"], batch["requested_field_paths"], candidate,
    )
    audit_passed, _audit_errors = audit_provider_response_schema(
        response_schema, expected_paths=batch["requested_field_paths"],
    )
    return DryRunReport(
        batch_id=batch_id,
        poem_id=batch["poem_id"],
        requested_path_count=batch["requested_path_count"],
        system_prompt_chars=len(bundle.system_prompt),
        user_prompt_chars=len(bundle.user_prompt),
        output_dir=str(out_dir),
        next_attempt_dir=str(next_attempt_dir(out_dir)),
        response_schema_audit_passed=audit_passed,
        planned_calls=1,
        actual_calls=0,
    )


# ══════════════════════════════════════════════════════════════════════════
# Full execution orchestration (Tasks 3, 6, 7, 8 wired together)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SmokeExecutionResult:
    accepted: bool
    response_status: str
    run_summary: dict[str, Any]
    smoke_candidate_path: str | None


def execute_smoke_batch(
    batch_id: str,
    repo_root: Path,
    *,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    max_attempts: int = MAX_LIVE_ATTEMPTS,
) -> SmokeExecutionResult:
    """The one live-call orchestration path. Writes every artifact under a
    freshly computed attempt_NN directory (Stage 5D.1 Task 5) — never the
    flat legacy root, so a prior attempt's evidence (success or failure) is
    never overwritten. Never applies a patch or writes a smoke candidate
    except on full, unanimous success. `max_attempts` defaults to
    MAX_LIVE_ATTEMPTS (Stage 5D's retry-bounded policy); a caller under a
    stricter one-shot authorization (e.g. Stage 5D.2's "do not retry for any
    reason") passes max_attempts=1 to guarantee zero retries regardless of
    error classification."""
    batch = load_batch(batch_id, repo_root)
    candidate = load_candidate(batch, repo_root)
    gemini_config = load_gemini_config()
    out_dir = output_dir_for(batch_id, repo_root)
    attempt_dir = next_attempt_dir(out_dir)
    timestamp = now_fn().isoformat()
    approved_paths = frozenset(batch["requested_field_paths"])

    bundle = eb.build_prompt_bundle_from_batch(batch, candidate)
    phash = prompt_sha256(bundle.system_prompt, bundle.user_prompt)

    atomic_write_json(
        attempt_dir / "prompt_request.json",
        {"system_prompt": bundle.system_prompt, "user_prompt": bundle.user_prompt, "prompt_sha256": phash},
    )

    request = build_request(
        bundle, gemini_config.model,
        poem_id=batch["poem_id"], requested_field_paths=batch["requested_field_paths"], candidate=candidate,
    )
    client = client_factory()
    outcome = generate_with_retry(client, request, sleep_fn=sleep_fn, max_attempts=max_attempts)

    request_metadata = {
        "provider": PROVIDER_NAME,
        "model": gemini_config.model,
        "project": gemini_config.project,
        "region": gemini_config.location,
        "sdk_version": _installed_sdk_version(),
        "api_version": API_VERSION,
        "timestamp": timestamp,
        "batch_id": batch_id,
        "attempt_dir": attempt_dir.name,
        "requested_paths": sorted(approved_paths),
        "generation_settings": generation_settings_summary(),
        "attempt_count": len(outcome.attempts),
        "attempts": [asdict(a) for a in outcome.attempts],
        "prompt_sha256": phash,
    }

    if not outcome.success:
        request_metadata["response_status"] = "provider_call_failed"
        atomic_write_json(attempt_dir / "request_metadata.json", request_metadata)
        run_summary = _build_run_summary(
            batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
            parse_status="not_attempted", patch_validation_status="not_attempted",
            patch_application_status="not_attempted", response_status="provider_call_failed",
            rejection_reason=outcome.final_error, smoke_candidate_written=False, smoke_candidate_path=None,
        )
        atomic_write_json(attempt_dir / "run_summary.json", run_summary)
        return SmokeExecutionResult(False, "provider_call_failed", run_summary, None)

    request_metadata["response_status"] = "success"
    atomic_write_json(attempt_dir / "request_metadata.json", request_metadata)

    raw_text = getattr(outcome.response, "text", None)
    atomic_write_text(attempt_dir / "raw_response.txt", raw_text or "")

    parse_result = parse_raw_response(raw_text)
    if parse_result.status != "success":
        run_summary = _build_run_summary(
            batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
            parse_status=parse_result.status, patch_validation_status="not_attempted",
            patch_application_status="not_attempted", response_status=parse_result.status,
            rejection_reason=parse_result.error, smoke_candidate_written=False, smoke_candidate_path=None,
        )
        atomic_write_json(attempt_dir / "run_summary.json", run_summary)
        return SmokeExecutionResult(False, parse_result.status, run_summary, None)

    atomic_write_json(attempt_dir / "parsed_patch.json", parse_result.parsed)

    match_result = validate_patch_matches_batch_exactly(
        parse_result.parsed, expected_poem_id=batch["poem_id"], approved_paths=approved_paths,
    )
    if match_result.status != "accepted":
        atomic_write_json(attempt_dir / "patch_validation.json", {
            "status": match_result.status, "reason": match_result.reason,
            "missing_paths": list(match_result.missing_paths), "extra_paths": list(match_result.extra_paths),
        })
        run_summary = _build_run_summary(
            batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
            parse_status="success", patch_validation_status="rejected",
            patch_application_status="not_attempted", response_status="validation_failed",
            rejection_reason=match_result.reason, smoke_candidate_written=False, smoke_candidate_path=None,
        )
        atomic_write_json(attempt_dir / "run_summary.json", run_summary)
        return SmokeExecutionResult(False, "validation_failed", run_summary, None)

    original_poem, translated_poem = _poem_texts(candidate)
    try:
        transaction = apply_patch_v1_1(
            candidate, parse_result.parsed, approved_paths, original_poem, translated_poem,
        )
    except Exception as exc:  # candidate malformed beyond inspection — not expected in practice
        atomic_write_json(attempt_dir / "patch_validation.json", {"status": "rejected", "reason": str(exc)})
        run_summary = _build_run_summary(
            batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
            parse_status="success", patch_validation_status="rejected",
            patch_application_status="not_attempted", response_status="application_failed",
            rejection_reason=str(exc), smoke_candidate_written=False, smoke_candidate_path=None,
        )
        atomic_write_json(attempt_dir / "run_summary.json", run_summary)
        return SmokeExecutionResult(False, "application_failed", run_summary, None)

    atomic_write_json(attempt_dir / "patch_validation.json", {
        "status": "accepted" if transaction.accepted else "rejected",
        "reason": transaction.rejection_reason,
        "validation_result": transaction.validation_result,
        "grounding_result": transaction.grounding_result,
    })
    atomic_write_json(attempt_dir / "patch_application.json", {
        "accepted": transaction.accepted,
        "applied_paths": list(transaction.applied_paths),
        "rejected_paths": list(transaction.rejected_paths),
        "rejection_reason": transaction.rejection_reason,
    })

    if not transaction.accepted:
        run_summary = _build_run_summary(
            batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
            parse_status="success", patch_validation_status="rejected",
            patch_application_status="rejected", response_status="application_failed",
            rejection_reason=transaction.rejection_reason, smoke_candidate_written=False, smoke_candidate_path=None,
        )
        atomic_write_json(attempt_dir / "run_summary.json", run_summary)
        return SmokeExecutionResult(False, "application_failed", run_summary, None)

    smoke_candidate = build_smoke_candidate(
        transaction.patched_candidate, batch_id=batch_id, applied_paths=transaction.applied_paths,
        run_timestamp=timestamp,
    )
    smoke_path = repo_root / SMOKE_CANDIDATE_DIR / f"{batch['poem_id']}.json"
    atomic_write_json(smoke_path, smoke_candidate)
    revalidate_smoke_candidate_from_disk(smoke_path)  # raises if the written file itself is invalid

    run_summary = _build_run_summary(
        batch_id=batch_id, poem_id=batch["poem_id"], request_metadata=request_metadata,
        parse_status="success", patch_validation_status="accepted",
        patch_application_status="applied", response_status="success",
        rejection_reason=None, smoke_candidate_written=True, smoke_candidate_path=str(smoke_path),
    )
    atomic_write_json(attempt_dir / "run_summary.json", run_summary)
    return SmokeExecutionResult(True, "success", run_summary, str(smoke_path))


def _installed_sdk_version() -> str:
    import google.genai

    return getattr(google.genai, "__version__", "unknown")


# ══════════════════════════════════════════════════════════════════════════
# Task 5 — dry-run-default CLI
# ══════════════════════════════════════════════════════════════════════════
class CliRejection(RuntimeError):
    """Raised for any CLI-level safety rejection (Task 5). Always caught by
    main() and turned into a clean message + non-zero exit code — never an
    uncaught traceback."""


def build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gemini_backfill_executor_v1_1",
        description="Stage 5D — dry-run-by-default Gemini smoke-batch executor.",
    )
    parser.add_argument("--batch-id", default=AUTHORIZED_BATCH_ID)
    parser.add_argument("--execute", action="store_true", help="Make the one authorized live call. Default: dry run.")
    parser.add_argument("--max-calls", type=int, default=1)
    parser.add_argument("--acknowledge-billing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--repo-root", default=".")
    return parser


def _validate_cli_args_for_execute(args: Any) -> None:
    if args.batch_id != AUTHORIZED_BATCH_ID:
        raise CliRejection(f"only batch {AUTHORIZED_BATCH_ID!r} is authorized in Stage 5D; got {args.batch_id!r}.")
    if args.max_calls != 1:
        raise CliRejection(f"--max-calls must be exactly 1; got {args.max_calls}.")
    if not args.acknowledge_billing:
        raise CliRejection("--execute requires --acknowledge-billing.")


def main(
    argv: list[str] | None = None,
    *,
    client_factory: ClientFactory | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    stdout: Any = None,
) -> int:
    import sys

    out_stream = stdout if stdout is not None else sys.stdout
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    if args.batch_id != AUTHORIZED_BATCH_ID:
        print(f"REJECTED: only batch {AUTHORIZED_BATCH_ID!r} is authorized in Stage 5D.", file=out_stream)
        return 2
    if args.max_calls != 1:
        print("REJECTED: --max-calls must be exactly 1.", file=out_stream)
        return 2

    try:
        if not args.execute:
            report = dry_run(args.batch_id, repo_root)
            print(json.dumps(report.to_dict(), indent=2), file=out_stream)
            return 0

        _validate_cli_args_for_execute(args)

        out_dir = output_dir_for(args.batch_id, repo_root)
        existing = latest_attempt_run_summary(out_dir)
        if existing is not None and existing.get("response_status") == "success" and not args.overwrite:
            raise CliRejection("a successful run already exists for this batch; pass --overwrite to redo.")
        # Note: even with --overwrite, execution always writes to a NEW
        # attempt_NN directory (next_attempt_dir) — "--overwrite" only lifts
        # this business-logic gate, it never causes an existing attempt's
        # files to be overwritten (Stage 5D.1 Task 5).

        gemini_config_error: str | None = None
        try:
            load_gemini_config()
        except ConfigError as exc:
            gemini_config_error = str(exc)
        if gemini_config_error is not None:
            raise CliRejection(gemini_config_error)

        if not check_adc_available():
            raise CliRejection("Application Default Credentials are not available.")

        # Re-validates planned_not_executed status; raises BatchAuthorizationError otherwise.
        load_batch(args.batch_id, repo_root)

    except (CliRejection, BatchAuthorizationError) as exc:
        print(f"REJECTED: {exc}", file=out_stream)
        return 2

    factory = client_factory or default_client_factory
    result = execute_smoke_batch(
        args.batch_id, repo_root, client_factory=factory, sleep_fn=sleep_fn, now_fn=now_fn,
    )
    print(json.dumps(result.run_summary, indent=2), file=out_stream)
    return 0 if result.accepted else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())


def _build_run_summary(
    *, batch_id: str, poem_id: str, request_metadata: dict[str, Any], parse_status: str,
    patch_validation_status: str, patch_application_status: str, response_status: str,
    rejection_reason: str | None, smoke_candidate_written: bool, smoke_candidate_path: str | None,
) -> dict[str, Any]:
    attempts = request_metadata.get("attempts", [])
    successful = sum(1 for a in attempts if a.get("outcome") == "success")
    return {
        "stage": STAGE,
        "batch_id": batch_id,
        "poem_id": poem_id,
        "attempt_dir": request_metadata.get("attempt_dir"),
        "provider": request_metadata.get("provider"),
        "model": request_metadata.get("model"),
        "project": request_metadata.get("project"),
        "region": request_metadata.get("region"),
        "sdk_version": request_metadata.get("sdk_version"),
        "api_version": request_metadata.get("api_version"),
        "timestamp": request_metadata.get("timestamp"),
        "provider_calls_made": len(attempts),
        "provider_attempts": len(attempts),
        "successful_responses": successful,
        "retries": max(0, len(attempts) - 1),
        "parse_status": parse_status,
        "patch_validation_status": patch_validation_status,
        "patch_application_status": patch_application_status,
        "response_status": response_status,
        "rejection_reason": rejection_reason,
        "smoke_candidate_written": smoke_candidate_written,
        "smoke_candidate_path": smoke_candidate_path,
    }
