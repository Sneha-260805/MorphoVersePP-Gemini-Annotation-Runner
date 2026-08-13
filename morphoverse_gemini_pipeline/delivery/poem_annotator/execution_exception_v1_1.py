"""Stage 5E.7 — narrowly-scoped, machine-readable execution exceptions.

An execution exception records that ONE specific requested path in ONE
specific Stage 5C batch has been forensically adjudicated (Stage 5E.6) as
having failed model backfill twice despite clean, unambiguous translation
evidence, and is therefore deferred to human translation review rather than
receiving a third identical automated attempt. It lets a future,
separately-authorized run request the batch's REMAINING approved paths
without the excluded one — never a general mechanism for removing arbitrary
fields from a batch (see the deliberately hardcoded authorization check in
`validate_execution_exception` below).

This module never calls a model, provider, or proxy, and never reads or
writes a credential. It only loads, validates, and (elsewhere, in
`gemini_pilot_execution_v1_1.py`) is consulted by the orchestrator to decide
whether a reduced request is currently permitted — never to perform one on
its own. Writing/using an exception file never modifies Stage 5B or Stage
5C artifacts; both are only ever read here.

See docs/GEMINI_PILOT_EXECUTION_STAGE5E.md's "Stage 5E.7" section for the
full contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import patch_v1_1 as pv

EXCEPTION_SCHEMA_VERSION = "5E.7.1"
STAGE5C_DIR = Path("pilot") / "backfill_requests" / "stage5c"
STAGE5B_DIR = Path("pilot") / "backfill_requests" / "stage5b"
EXCEPTIONS_DIR = Path("pilot") / "execution_exceptions" / "stage5e"

EXCLUSION_REASON_NON_VERBATIM = "repeated_non_verbatim_translation_span"
DISPOSITION_DEFER_TO_HUMAN_REVIEW = "defer_specific_path_to_human_translation_review"

# The status a reduced (exception-covered) batch's run_summary.json carries
# once its 35-path subset succeeds — deliberately distinct from plain
# "success" everywhere this project reports on batch completion, so a
# poem's eventual final candidate can never be mistaken for fully
# model-completed while an adjudicated path remains deferred.
ADJUDICATED_COMPLETION_STATUS = "completed_with_adjudicated_human_review_exception"

# ══════════════════════════════════════════════════════════════════════════
# The ONE authorized exception (Stage 5E.6's adjudication). Deliberately
# hardcoded, not a lookup table or generic rule — this module implements no
# mechanism by which an arbitrary path could be excluded from an arbitrary
# batch. A future, genuinely different adjudicated exception would require
# its own reviewed code change here, not a new data file alone.
# ══════════════════════════════════════════════════════════════════════════
_AUTHORIZED_POEM_ID = "MV++_0073"
_AUTHORIZED_BATCH_ID = "MV++_0073_batch_01"
_AUTHORIZED_EXCLUDED_PATH = "annotation.cultural_entities[5].source_span_translation"

_FORBIDDEN_VALUE_KEYS = frozenset({
    "value", "values", "proposed_value", "annotation_value", "inserted_value",
    "source_span_translation_value", "replacement_value", "filled_value",
})


class ExecutionExceptionError(ValueError):
    """Raised for any execution-exception format or safety-validation
    failure. Carries a single, human-readable reason."""


def load_execution_exception(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _walk_values(node: Any):
    """Yields every leaf VALUE (never a dict key or list index) in a nested
    JSON-like structure — used so a safety scan for a leaked word can never
    be tripped by a field's own NAME (e.g. "silver_gold_implication" is a
    key, not a claim)."""
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_values(v)
    else:
        yield node


def validate_execution_exception(exception: dict[str, Any], repo_root: Path) -> None:
    """Raises ExecutionExceptionError on the first violation found; returns
    None (never a value) on success. Implements Stage 5E.7 Task 2's 15
    safety requirements. Never mutates `exception`, never writes to Stage
    5B/5C, never authorizes execution on its own — callers must separately
    require an explicit CLI acknowledgement before treating a validated
    exception as permission to make a reduced request."""
    if not isinstance(exception, dict):
        raise ExecutionExceptionError("execution exception must be a JSON object.")

    poem_id = exception.get("poem_id")
    batch_id = exception.get("original_batch_id")
    excluded_paths = exception.get("excluded_paths")
    remaining_paths = exception.get("remaining_model_paths")
    original_paths = exception.get("original_requested_paths")

    # Rules 6/15: this is a deliberately narrow, hardcoded authorization —
    # not a generic "any batch, any path" mechanism. Any other combination
    # is rejected outright, before any further structural check.
    if (poem_id, batch_id) != (_AUTHORIZED_POEM_ID, _AUTHORIZED_BATCH_ID):
        raise ExecutionExceptionError(
            f"unauthorized batch for an execution exception: poem_id={poem_id!r}, "
            f"original_batch_id={batch_id!r}. Only {_AUTHORIZED_POEM_ID!r}/"
            f"{_AUTHORIZED_BATCH_ID!r} is currently adjudicated."
        )

    # Rule 1: the original batch must exist in Stage 5C.
    batch_path = repo_root / STAGE5C_DIR / f"{batch_id}.json"
    if not batch_path.exists():
        raise ExecutionExceptionError(f"Stage 5C batch file not found: {batch_path}")
    stage5c_batch = _load_json(batch_path)
    if stage5c_batch.get("poem_id") != poem_id:
        raise ExecutionExceptionError("Stage 5C batch poem_id does not match the exception's poem_id.")

    # Rule 2: original_requested_paths must exactly equal the Stage 5C batch's own paths.
    stage5c_paths = set(stage5c_batch.get("requested_field_paths", []))
    if not isinstance(original_paths, list) or set(original_paths) != stage5c_paths:
        raise ExecutionExceptionError("original_requested_paths does not exactly equal the Stage 5C batch's own paths.")
    if exception.get("original_requested_path_count") != len(original_paths):
        raise ExecutionExceptionError("original_requested_path_count does not match len(original_requested_paths).")

    # Rule 5/6: exactly one excluded path, and it must be the adjudicated one.
    if not isinstance(excluded_paths, list) or len(excluded_paths) != 1:
        raise ExecutionExceptionError("exactly one excluded_paths entry is required for this exception.")
    if excluded_paths[0] != _AUTHORIZED_EXCLUDED_PATH:
        raise ExecutionExceptionError(f"excluded_paths[0] must be exactly {_AUTHORIZED_EXCLUDED_PATH!r}.")

    # Rules 3/4: disjoint, and their union equals the original 36 paths.
    if not isinstance(remaining_paths, list):
        raise ExecutionExceptionError("remaining_model_paths must be a list.")
    excluded_set, remaining_set, original_set = set(excluded_paths), set(remaining_paths), set(original_paths)
    if excluded_set & remaining_set:
        raise ExecutionExceptionError("excluded_paths and remaining_model_paths are not disjoint.")
    if (excluded_set | remaining_set) != original_set:
        raise ExecutionExceptionError("excluded_paths union remaining_model_paths does not equal the original batch paths.")

    # Rule 7: remaining path count must be exactly 35.
    if len(remaining_paths) != 35:
        raise ExecutionExceptionError(f"remaining_model_paths must contain exactly 35 paths; got {len(remaining_paths)}.")

    # Rule 8: every remaining path must still be Stage 5B model_backfill_allowed.
    plan_path = repo_root / STAGE5B_DIR / f"{poem_id}.json"
    if not plan_path.exists():
        raise ExecutionExceptionError(f"Stage 5B plan not found: {plan_path}")
    plan = _load_json(plan_path)
    approved = set(plan.get("requested_field_paths", []))
    for path in remaining_paths:
        if path not in approved:
            raise ExecutionExceptionError(f"remaining path {path!r} is not in the Stage 5B approved requested_field_paths.")
        leaf = pv.leaf_field(path)
        if leaf not in pv.MODEL_BACKFILL_FIELDS:
            raise ExecutionExceptionError(f"remaining path {path!r} is not classified model_backfill_allowed.")

    # Rule 9 (structural guarantee, not a JSON check): this function never
    # writes to Stage 5B/5C; both were only ever opened for reading above.

    # Rule 10/11/12: the exception must point to an existing adjudication
    # report recording two failed attempts and EXACT_UNAMBIGUOUS evidence.
    report_rel_path = exception.get("adjudication_report_path")
    if not report_rel_path:
        raise ExecutionExceptionError("adjudication_report_path is required.")
    report_path = repo_root / report_rel_path
    if not report_path.exists():
        raise ExecutionExceptionError(f"adjudication report not found: {report_path}")
    report = _load_json(report_path)
    if report.get("attempt_count") != 2:
        raise ExecutionExceptionError("adjudication report does not record exactly two attempts.")
    substr_result = report.get("exact_substring_result", {})
    if not substr_result or any(substr_result.get(k) is not False for k in substr_result):
        raise ExecutionExceptionError("adjudication report does not record two failed (non-exact-substring) attempts.")
    category = (report.get("evidence_classification") or {}).get("category", "")
    if "EXACT_UNAMBIGUOUS" not in category:
        raise ExecutionExceptionError(
            f"adjudication report's evidence classification {category!r} is not EXACT_UNAMBIGUOUS; "
            "this exception format is only valid for EXACT_UNAMBIGUOUS evidence."
        )

    # Rule 13: no replacement annotation value anywhere in the exception.
    for key in exception:
        if key.lower() in _FORBIDDEN_VALUE_KEYS:
            raise ExecutionExceptionError(f"execution exception must not contain a replacement value field: {key!r}.")
    # Scan only VALUES (never key names) for a leaked annotation value or an
    # illegitimate status claim — a field named e.g. "silver_gold_implication"
    # is exactly how this format correctly *disclaims* silver/gold, so the
    # key names themselves must never trip this check, only what a value says.
    values_blob = " ".join(str(v).lower() for v in _walk_values(exception))
    if "sari" in values_blob:
        raise ExecutionExceptionError("execution exception must not contain a proposed annotation value.")

    # Human-review/silver/gold claim checks (Task 7 items 10/11).
    if exception.get("human_review_required") is not True:
        raise ExecutionExceptionError("human_review_required must be true — this exception defers to human review, it does not complete it.")
    if exception.get("silver_gold_implication") != "none":
        raise ExecutionExceptionError("silver_gold_implication must be exactly 'none'.")
    for forbidden_marker in ("human_reviewed", "human_review_completed", "silver", "gold"):
        if forbidden_marker in values_blob:
            raise ExecutionExceptionError(f"execution exception must not claim {forbidden_marker!r}.")

    # Rule 14: the exception must never authorize execution by itself.
    if exception.get("reduced_model_request_authorized") is not False:
        raise ExecutionExceptionError("reduced_model_request_authorized must be false — an exception file alone never authorizes execution.")
    if exception.get("model_retry_for_excluded_path_authorized") is not False:
        raise ExecutionExceptionError("model_retry_for_excluded_path_authorized must be false — the excluded path is never retried automatically.")

    if exception.get("annotation_value_inserted") is not False:
        raise ExecutionExceptionError("annotation_value_inserted must be false.")

    if exception.get("exclusion_reason") != EXCLUSION_REASON_NON_VERBATIM:
        raise ExecutionExceptionError(f"exclusion_reason must be {EXCLUSION_REASON_NON_VERBATIM!r}.")

    if exception.get("schema_version") != EXCEPTION_SCHEMA_VERSION:
        raise ExecutionExceptionError(f"schema_version must be {EXCEPTION_SCHEMA_VERSION!r}.")
