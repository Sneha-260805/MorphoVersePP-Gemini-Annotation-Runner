"""Stage 5E — sequential, one-attempt-per-batch execution of the 16 Stage 5C
batches not already completed by Stage 5D.2 (MV++_1153_batch_01), producing
five Gemini(Vertex)-backfilled pilot candidates.

Scope discipline: this module reuses `gemini_backfill_executor_v1_1`'s
provider-facing primitives (schema construction, request building, retry/
error classification, response parsing, atomic writes, ADC/config handling)
rather than duplicating them. It adds only what Stage 5D didn't need:
multi-batch discovery/ordering, a durable on-disk execution ledger (each
batch's own `run_summary.json` IS the ledger — no separate mutable ledger
file to drift out of sync), per-poem candidate accumulation with
checkpointing, and final-candidate assembly. No provider client is
constructed at import time. Never imports a Claude/GPT/IndicBERT module.

See docs/GEMINI_PILOT_EXECUTION_STAGE5E.md for the full contract.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import execution_batch_v1_1 as eb
from . import execution_exception_v1_1 as eex
from . import execution_split_v1_1 as esplit
from . import gemini_backfill_executor_v1_1 as ex
from . import patch_v1_1 as pv
from .grounding import (
    GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
    validate_cultural_grounding_v1_1,
    validate_figurative_grounding_v1_1,
)
from .models import validate_model_payload_v1_1
from .patch_v1_1 import apply_patch_v1_1

STAGE = "5E_pilot_execution"

# ══════════════════════════════════════════════════════════════════════════
# Authorization constants
# ══════════════════════════════════════════════════════════════════════════
PRIOR_COMPLETED_BATCH_ID = "MV++_1153_batch_01"  # Stage 5D.2 — never call again
MAX_NEW_ATTEMPTS = 16
PILOT_POEM_IDS = ("MV++_0011", "MV++_0073", "MV++_1118", "MV++_1153", "MV++_1249")

STAGE5C_DIR = ex.STAGE5C_DIR
PRE_BACKFILL_DIR = ex.PRE_BACKFILL_DIR
SMOKE_CANDIDATE_DIR = ex.SMOKE_CANDIDATE_DIR
STAGE5D_RUNS_DIR = ex.STAGE5D_RUNS_DIR
STAGE5B_DIR = Path("pilot") / "backfill_requests" / "stage5b"

STAGE5E_RUNS_DIR = Path("pilot") / "provider_runs" / "stage5e"
CHECKPOINT_DIR = Path("pilot") / "annotations_v1_1" / "gemini_checkpoints"
FINAL_CANDIDATE_DIR = Path("pilot") / "annotations_v1_1" / "gemini_backfilled"
REPORTS_DIR = Path("pilot") / "reports" / "stage5e"

FINAL_CANDIDATE_STATUS = "gemini_backfilled_pilot_candidate"
FINAL_CANDIDATE_TIER = "gemini_vertex_backfilled_candidate"
PROVENANCE_KEY = "gemini_backfill_provenance"

atomic_write_json = ex.atomic_write_json
atomic_write_text = ex.atomic_write_text
prompt_sha256 = ex.prompt_sha256
ClientFactory = ex.ClientFactory


# ══════════════════════════════════════════════════════════════════════════
# Errors
# ══════════════════════════════════════════════════════════════════════════
class BatchDiscoveryError(ValueError):
    """Raised when the on-disk Stage 5C batch set doesn't match what this
    stage expects (wrong count, an unrecognized ID, a requested batch not
    found)."""


class StageBlockedError(RuntimeError):
    """Raised when execution cannot proceed: a prior batch failed, working
    tree was dirty, or configuration/ADC is unavailable."""


# ══════════════════════════════════════════════════════════════════════════
# Task 2 — deterministic batch discovery (poem_id asc, then batch_index asc)
# ══════════════════════════════════════════════════════════════════════════
def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_all_stage5c_batches(repo_root: Path) -> tuple[dict[str, Any], ...]:
    paths = sorted((repo_root / STAGE5C_DIR).glob("*.json"))
    return tuple(_load_json(p) for p in paths)


def discover_remaining_batches(repo_root: Path) -> tuple[dict[str, Any], ...]:
    """The 16 batches authorized for this stage: every Stage 5C batch except
    PRIOR_COMPLETED_BATCH_ID, sorted deterministically by (poem_id,
    batch_index). Never re-discovers or re-includes the completed batch."""
    all_batches = discover_all_stage5c_batches(repo_root)
    remaining = [b for b in all_batches if b["batch_id"] != PRIOR_COMPLETED_BATCH_ID]
    remaining.sort(key=lambda b: (b["poem_id"], b["batch_index"]))
    return tuple(remaining)


def validate_batch_selection(batch_ids: "list[str] | tuple[str, ...]", repo_root: Path) -> tuple[dict[str, Any], ...]:
    """Used by the CLI's optional --batch-ids override: every ID must be a
    member of the authorized remaining set, and PRIOR_COMPLETED_BATCH_ID or
    any unrecognized ID is rejected outright."""
    remaining_by_id = {b["batch_id"]: b for b in discover_remaining_batches(repo_root)}
    if PRIOR_COMPLETED_BATCH_ID in batch_ids:
        raise BatchDiscoveryError(f"{PRIOR_COMPLETED_BATCH_ID} already completed in Stage 5D.2 and must never be called again.")
    unknown = [bid for bid in batch_ids if bid not in remaining_by_id]
    if unknown:
        raise BatchDiscoveryError(f"batch ID(s) not present in the authorized Stage 5C remaining set: {unknown}")
    selected = [remaining_by_id[bid] for bid in batch_ids]
    selected.sort(key=lambda b: (b["poem_id"], b["batch_index"]))
    return tuple(selected)


# ══════════════════════════════════════════════════════════════════════════
# Durable ledger — each batch's own run_summary.json IS the ledger entry
# ══════════════════════════════════════════════════════════════════════════
def batch_attempt_dir(batch_id: str, repo_root: Path) -> Path:
    """The default first-attempt location for a batch that has never been
    attempted before. Only ever used as `execute_one_batch`'s default
    `out_dir` (a genuinely fresh batch always starts at attempt_01) — never
    used to read a batch's CURRENT status, since a Stage 5E.2-style forensic
    retry may have since written a later, authoritative attempt_NN. Use
    `latest_attempt_dir_for_batch` to read status."""
    return repo_root / STAGE5E_RUNS_DIR / batch_id / "attempt_01"


def latest_attempt_dir_for_batch(batch_id: str, repo_root: Path) -> Path:
    """The most recent attempt directory that actually exists on disk for
    this batch — attempt_01 for the common case, or a higher attempt_NN if
    an explicitly-authorized forensic retry (Stage 5E.2) has since run.
    Read-only; never creates anything. Falls back to the (possibly
    not-yet-existing) attempt_01 path when no attempt has been made at all,
    matching read_batch_run_summary's existing "return None" behavior for
    that case."""
    base_dir = repo_root / STAGE5E_RUNS_DIR / batch_id
    numbered = ex._existing_attempt_numbers(base_dir)
    if not numbered:
        return batch_attempt_dir(batch_id, repo_root)
    return base_dir / f"attempt_{max(numbered):02d}"


def read_batch_run_summary(batch_id: str, repo_root: Path) -> dict[str, Any] | None:
    """Reads the batch's CURRENT (most recent) run_summary.json — never
    stuck on attempt_01 once a later attempt exists. This is the single
    source every ledger/resume/audit function in this module reads from, so
    a Stage 5E.2 forensic retry's success or failure is always what the
    rest of the system sees."""
    path = latest_attempt_dir_for_batch(batch_id, repo_root) / "run_summary.json"
    if not path.exists():
        return None
    return _load_json(path)


@dataclass(frozen=True)
class LedgerEntry:
    batch_id: str
    poem_id: str
    batch_index: int
    status: str  # "not_attempted" | "success" | "failed"
    response_status: str | None


def read_execution_ledger(repo_root: Path) -> tuple[LedgerEntry, ...]:
    """Pure, read-only scan of every authorized remaining batch's on-disk
    run_summary.json (or absence thereof). This IS the durable ledger —
    there is no separate mutable ledger file to fall out of sync with it."""
    entries = []
    for batch in discover_remaining_batches(repo_root):
        summary = read_batch_run_summary(batch["batch_id"], repo_root)
        if summary is None:
            status, response_status = "not_attempted", None
        elif _is_batch_complete(summary.get("response_status")):
            # Stage 5E.7 (adjudicated execution exception) and Stage 5E.12
            # (semantic execution split) both resume exactly like a plain
            # success — never blocking later batches — but their raw
            # response_status remains distinguishable everywhere it's read
            # from directly, so neither is ever misreported as an ordinary
            # single-attempt success.
            status, response_status = "success", summary.get("response_status")
        else:
            status, response_status = "failed", summary.get("response_status")
        entries.append(LedgerEntry(batch["batch_id"], batch["poem_id"], batch["batch_index"], status, response_status))
    return tuple(entries)


@dataclass(frozen=True)
class NextAction:
    action: str  # "run" | "blocked" | "all_complete"
    batch: dict[str, Any] | None
    blocking_batch_id: str | None


def next_runnable_batch(repo_root: Path) -> NextAction:
    """Task 2 resume logic: skip every already-successful batch; the first
    batch with a recorded FAILURE permanently blocks all later batches
    (durably — even across process restarts) unless a forensic override is
    explicitly supplied by the caller (never used within this stage's own
    execution, since every batch here starts not_attempted)."""
    for entry in read_execution_ledger(repo_root):
        if entry.status == "success":
            continue
        batch = next(b for b in discover_remaining_batches(repo_root) if b["batch_id"] == entry.batch_id)
        if entry.status == "failed":
            # Stage 5E.1: still return the batch dict (not just its ID) so a
            # dry-run/audit caller can report what a future retry would
            # target (next attempt dir, derived token budget) even though
            # this batch is not runnable right now without a forensic
            # override — it remains "the first unfinished batch."
            return NextAction("blocked", batch, entry.batch_id)
        return NextAction("run", batch, None)
    return NextAction("all_complete", None, None)


# ══════════════════════════════════════════════════════════════════════════
# Per-poem candidate state (Stage 5A start, or resume from latest checkpoint)
# ══════════════════════════════════════════════════════════════════════════
def load_pre_backfill_candidate(poem_id: str, repo_root: Path) -> dict[str, Any]:
    path = repo_root / PRE_BACKFILL_DIR / f"{poem_id}.json"
    return _load_json(path)


def checkpoint_path(poem_id: str, batch_index: int, repo_root: Path) -> Path:
    return repo_root / CHECKPOINT_DIR / poem_id / f"after_batch_{batch_index:02d}.json"


def latest_checkpoint(poem_id: str, repo_root: Path) -> dict[str, Any] | None:
    checkpoint_dir = repo_root / CHECKPOINT_DIR / poem_id
    if not checkpoint_dir.exists():
        return None
    numbered = sorted(
        int(p.stem.split("_")[-1])
        for p in checkpoint_dir.glob("after_batch_*.json")
        if p.stem.split("_")[-1].isdigit()
    )
    if not numbered:
        return None
    return _load_json(checkpoint_path(poem_id, numbered[-1], repo_root))


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.4 Task 5 — safe, read-only grounding-failure diagnostics
# ══════════════════════════════════════════════════════════════════════════
def diagnose_grounding_rejection(
    parsed_patch: dict[str, Any], approved_paths: "frozenset[str]", original_poem: str, translated_poem: str,
) -> "dict[str, Any] | None":
    """Best-effort, read-only diagnostic for an ALREADY-REJECTED patch whose
    rejection_reason was "ungrounded_field" (Task 5). Independently checks
    each requested grounding-sensitive path's returned value with a direct
    substring search against the relevant poem text — never touches
    patch_v1_1's own grounding validator or changes what gets
    accepted/rejected; this only explains a rejection that already
    happened. Returns None when no non-verbatim grounding-sensitive value is
    found (the rejection came from some other grounding condition this
    simple check doesn't independently model, e.g. a bad line_ref)."""
    patched_values = {
        item["path"]: item["value"]
        for item in parsed_patch.get("patches", [])
        if isinstance(item, dict) and "path" in item
    }
    for path in sorted(approved_paths):
        leaf = pv.leaf_field(path)
        if leaf not in ("source_span_original", "source_span_translation"):
            continue
        value = patched_values.get(path)
        if not isinstance(value, str):
            continue
        haystack = original_poem if leaf == "source_span_original" else translated_poem
        if value not in haystack:
            return {
                "failing_field_path": path,
                "returned_candidate_value": value,
                "is_exact_substring": False,
                "grounding_error_category": "non_verbatim_span",
            }
    return None


def starting_working_copy(poem_id: str, repo_root: Path) -> dict[str, Any]:
    """The poem's current accumulated candidate: its latest checkpoint if
    one exists (resume case), otherwise a deep copy of the immutable Stage
    5A pre-backfill candidate. Never mutates or re-reads the Stage 5A file
    after this initial load."""
    checkpoint = latest_checkpoint(poem_id, repo_root)
    if checkpoint is not None:
        return checkpoint
    return copy.deepcopy(load_pre_backfill_candidate(poem_id, repo_root))


def verify_checkpoint_continuity(
    previous_working_copy: dict[str, Any], new_working_copy: dict[str, Any], previously_applied_paths: "set[str]",
) -> None:
    """Task 5 step 4: defense-in-depth re-check that every path applied by
    an earlier batch is still present, unchanged, in the new working copy.
    Raises AssertionError (never silently ignored) if apply_patch_v1_1
    somehow disturbed something outside its own patch's paths."""
    for path in previously_applied_paths:
        before = pv.get_value_at_path(previous_working_copy, path)
        after = pv.get_value_at_path(new_working_copy, path)
        if before != after:
            raise AssertionError(f"checkpoint continuity violated: {path} changed outside its own batch.")


# ══════════════════════════════════════════════════════════════════════════
# Task 4/6 — one batch, one attempt, full validation chain (reuses Stage 5D)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class BatchExecutionResult:
    accepted: bool
    batch_id: str
    poem_id: str
    response_status: str
    run_summary: dict[str, Any]
    patched_candidate: dict[str, Any] | None
    applied_paths: tuple[str, ...]


def execute_one_batch(
    batch: dict[str, Any],
    poem_working_copy: dict[str, Any],
    repo_root: Path,
    *,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    out_dir: "Path | None" = None,
    prompt_bundle_builder: "Callable[[dict[str, Any], dict[str, Any]], Any] | None" = None,
) -> BatchExecutionResult:
    """Exactly one provider attempt (max_attempts=1 — no retry for any
    reason), full raw-preserve/parse/validate/apply chain, atomic writes
    under `out_dir` (defaults to
    pilot/provider_runs/stage5e/<batch_id>/attempt_01/ — the normal,
    never-before-attempted-batch case). A caller retrying a batch that
    already has a recorded failure (Stage 5E.2's `execute_forensic_retry`)
    passes an explicit, freshly computed `out_dir` — this function itself
    never decides which attempt number it is; it only ever writes to the
    directory it is given, and never touches any other directory.

    `prompt_bundle_builder` defaults to `execution_batch_v1_1.build_prompt_bundle_from_batch`
    (identical to every prior stage's behavior — every existing caller that
    doesn't pass this argument is completely unaffected). Stage 5E.12's
    split-part execution is the one caller that passes
    `execution_split_v1_1.build_split_part_prompt_bundle` instead, to append
    the generic anti-repetition contract to a split part's prompt — every
    other batch's prompt shape is untouched by this parameter's existence."""
    batch_id, poem_id = batch["batch_id"], batch["poem_id"]
    if out_dir is None:
        out_dir = batch_attempt_dir(batch_id, repo_root)
    if prompt_bundle_builder is None:
        prompt_bundle_builder = eb.build_prompt_bundle_from_batch
    gemini_config = ex.load_gemini_config()
    timestamp = now_fn().isoformat()
    approved_paths = frozenset(batch["requested_field_paths"])

    bundle = prompt_bundle_builder(batch, poem_working_copy)
    phash = prompt_sha256(bundle.system_prompt, bundle.user_prompt)
    atomic_write_json(
        out_dir / "prompt_request.json",
        {"system_prompt": bundle.system_prompt, "user_prompt": bundle.user_prompt, "prompt_sha256": phash},
    )

    # Stage 5E.1 Task 3: the output-token budget is derived from THIS batch's
    # own requested-path count, never from poem_id or a fixed smoke-sized
    # constant — a 36-path batch needs materially more room than a 19-path one.
    output_token_budget = ex.determine_output_token_budget(
        batch["requested_field_paths"], semantic_units=batch.get("semantic_unit_types", ()),
    )
    request = ex.build_request(
        bundle, gemini_config.model,
        poem_id=poem_id, requested_field_paths=batch["requested_field_paths"], candidate=poem_working_copy,
        max_output_tokens=output_token_budget,
    )
    client = client_factory()
    outcome = ex.generate_with_retry(client, request, sleep_fn=sleep_fn, max_attempts=1)

    request_metadata = {
        "provider": ex.PROVIDER_NAME, "model": gemini_config.model, "project": gemini_config.project,
        "region": gemini_config.location, "sdk_version": ex._installed_sdk_version(), "api_version": ex.API_VERSION,
        "timestamp": timestamp, "batch_id": batch_id, "poem_id": poem_id,
        "requested_paths": sorted(approved_paths),
        "generation_settings": ex.generation_settings_summary(max_output_tokens=output_token_budget),
        "output_token_budget": output_token_budget,
        "attempt_count": len(outcome.attempts), "attempts": [asdict(a) for a in outcome.attempts],
        "prompt_sha256": phash,
    }

    def _fail(response_status: str, parse_status: str, patch_validation_status: str,
               patch_application_status: str, rejection_reason: str | None) -> BatchExecutionResult:
        run_summary = _build_run_summary(
            batch_id, poem_id, request_metadata, parse_status, patch_validation_status,
            patch_application_status, response_status, rejection_reason, repo_root=repo_root,
        )
        atomic_write_json(out_dir / "run_summary.json", run_summary)
        return BatchExecutionResult(False, batch_id, poem_id, response_status, run_summary, None, ())

    if not outcome.success:
        request_metadata["response_status"] = "provider_call_failed"
        request_metadata["completion_classification"] = ex.classify_completion(
            provider_call_succeeded=False, finish_reason=None,
        )
        atomic_write_json(out_dir / "request_metadata.json", request_metadata)
        return _fail("provider_call_failed", "not_attempted", "not_attempted", "not_attempted", outcome.final_error)

    # Stage 5E.1 Task 4: capture safe, tolerant provider completion metadata
    # for every successful call, BEFORE parsing — so a parse failure (e.g.
    # truncated JSON) still gets an honest finish_reason/token-usage record
    # rather than none at all. Never fabricated: every field defaults to
    # None when the installed SDK's response object doesn't expose it.
    completion_metadata = ex.extract_completion_metadata(outcome.response)
    completion_classification = ex.classify_completion(
        provider_call_succeeded=True, finish_reason=completion_metadata["finish_reason"],
    )
    request_metadata["completion_metadata"] = completion_metadata
    request_metadata["completion_classification"] = completion_classification
    request_metadata["response_status"] = "success"
    atomic_write_json(out_dir / "request_metadata.json", request_metadata)

    raw_text = getattr(outcome.response, "text", None)
    atomic_write_text(out_dir / "raw_response.txt", raw_text or "")

    parse_result = ex.parse_raw_response(raw_text)
    if parse_result.status != "success":
        return _fail(parse_result.status, parse_result.status, "not_attempted", "not_attempted", parse_result.error)

    atomic_write_json(out_dir / "parsed_patch.json", parse_result.parsed)

    match_result = ex.validate_patch_matches_batch_exactly(
        parse_result.parsed, expected_poem_id=poem_id, approved_paths=approved_paths,
    )
    if match_result.status != "accepted":
        atomic_write_json(out_dir / "patch_validation.json", {
            "status": match_result.status, "reason": match_result.reason,
            "missing_paths": list(match_result.missing_paths), "extra_paths": list(match_result.extra_paths),
        })
        return _fail("validation_failed", "success", "rejected", "not_attempted", match_result.reason)

    original_poem, translated_poem = poem_working_copy["original_poem"], poem_working_copy["translated_poem"]
    transaction = apply_patch_v1_1(
        poem_working_copy, parse_result.parsed, approved_paths, original_poem, translated_poem,
    )
    atomic_write_json(out_dir / "patch_validation.json", {
        "status": "accepted" if transaction.accepted else "rejected",
        "reason": transaction.rejection_reason,
        "validation_result": transaction.validation_result,
        "grounding_result": transaction.grounding_result,
    })
    atomic_write_json(out_dir / "patch_application.json", {
        "accepted": transaction.accepted,
        "applied_paths": list(transaction.applied_paths),
        "rejected_paths": list(transaction.rejected_paths),
        "rejection_reason": transaction.rejection_reason,
    })
    if not transaction.accepted:
        if transaction.rejection_reason == "ungrounded_field":
            request_metadata["grounding_diagnostics"] = diagnose_grounding_rejection(
                parse_result.parsed, approved_paths, original_poem, translated_poem,
            )
        return _fail("application_failed", "success", "accepted", "rejected", transaction.rejection_reason)

    run_summary = _build_run_summary(
        batch_id, poem_id, request_metadata, "success", "accepted", "applied", "success", None, repo_root=repo_root,
    )
    atomic_write_json(out_dir / "run_summary.json", run_summary)
    return BatchExecutionResult(
        True, batch_id, poem_id, "success", run_summary, transaction.patched_candidate, transaction.applied_paths,
    )


def _build_run_summary(
    batch_id: str, poem_id: str, request_metadata: dict[str, Any], parse_status: str,
    patch_validation_status: str, patch_application_status: str, response_status: str, rejection_reason: str | None,
    *, repo_root: "Path | None" = None,
) -> dict[str, Any]:
    attempts = request_metadata.get("attempts", [])
    completion_metadata = request_metadata.get("completion_metadata") or {}
    finish_reason_recorded = completion_metadata.get("finish_reason") is not None
    completion_classification = request_metadata.get("completion_classification")
    max_tokens_diagnosis = _max_tokens_diagnosis(completion_classification, finish_reason_recorded)
    max_tokens_confirmed = max_tokens_diagnosis == "confirmed"

    # Stage 5E.11 Task 4/5 — safe derived diagnostic fields. Every value here
    # is either a plain int/float/bool/str already present elsewhere in
    # request_metadata/completion_metadata, or None when not computable —
    # never a fabricated estimate, never the raw provider response object.
    requested_paths = request_metadata.get("requested_paths") or []
    configured_budget = request_metadata.get("output_token_budget")
    provider_output_tokens = completion_metadata.get("output_token_count")
    provider_thoughts_tokens = completion_metadata.get("thoughts_token_count")

    observed_output_budget_utilization = None
    if (
        isinstance(provider_output_tokens, int) and not isinstance(provider_output_tokens, bool)
        and isinstance(configured_budget, int) and not isinstance(configured_budget, bool)
        and configured_budget > 0
    ):
        observed_output_budget_utilization = round(provider_output_tokens / configured_budget, 4)

    # Only meaningful for an unfinished batch — a proposed next attempt
    # directory is never computed (and never created) for a batch that just
    # succeeded. Pure, read-only path computation via the same numbered-
    # attempt-directory scan every other resume/ledger function in this
    # module already uses; never creates the directory itself.
    proposed_next_attempt_dir = None
    recommended_corrected_output_token_budget = None
    if response_status != "success" and repo_root is not None:
        proposed_next_attempt_dir = str(next_attempt_dir_for_batch(batch_id, repo_root))
        if max_tokens_confirmed and requested_paths:
            recommended_corrected_output_token_budget = ex.determine_output_token_budget(requested_paths)

    return {
        "stage": STAGE, "batch_id": batch_id, "poem_id": poem_id,
        "provider": request_metadata.get("provider"), "model": request_metadata.get("model"),
        "project": request_metadata.get("project"), "region": request_metadata.get("region"),
        "sdk_version": request_metadata.get("sdk_version"), "api_version": request_metadata.get("api_version"),
        "timestamp": request_metadata.get("timestamp"),
        "provider_calls_made": len(attempts), "provider_attempts": len(attempts),
        "successful_responses": sum(1 for a in attempts if a.get("outcome") == "success"),
        "retries": 0,
        "requested_path_count": len(requested_paths),
        "parse_status": parse_status, "patch_validation_status": patch_validation_status,
        "patch_application_status": patch_application_status, "response_status": response_status,
        "rejection_reason": rejection_reason,
        # Stage 5E.1 Task 3/4/5 — adaptive budgeting and honest, tolerant
        # completion-metadata reporting. Never inferred from parse_status:
        # max_tokens_diagnosis depends only on provider-reported finish_reason.
        "output_token_budget": request_metadata.get("output_token_budget"),
        "configured_output_token_budget": configured_budget,
        "completion_classification": completion_classification,
        "provider_finish_reason": completion_metadata.get("finish_reason"),
        "provider_finish_reason_recorded": finish_reason_recorded,
        "provider_output_token_count": provider_output_tokens,
        "provider_thoughts_token_count": provider_thoughts_tokens,
        "observed_output_budget_utilization": observed_output_budget_utilization,
        "max_tokens_diagnosis": max_tokens_diagnosis,
        "max_tokens_confirmed": max_tokens_confirmed,
        # Stage 5E.4 Task 5 — safe grounding-failure diagnostics. Only ever
        # populated when this batch was actually rejected for
        # "ungrounded_field"; None in every other case (success or any other
        # rejection reason) — never fabricated, never inferred from
        # parse_status alone.
        "grounding_diagnostics": request_metadata.get("grounding_diagnostics"),
        "patch_fully_rejected": {"applied": False, "rejected": True}.get(patch_application_status),
        "checkpoint_written": response_status == "success",
        "retry_recommendation": (
            "manual_authorization_required" if response_status != "success" else "not_applicable"
        ),
        # Stage 5E.11 Task 5 — max-token failure diagnostics. Both remain
        # None for a successful batch, and recommended_corrected_output_token_budget
        # remains None unless this specific failure was a confirmed
        # MAX_TOKENS finish (never suggested for any other failure reason).
        "proposed_next_attempt_dir": proposed_next_attempt_dir,
        "recommended_corrected_output_token_budget": recommended_corrected_output_token_budget,
    }


def _max_tokens_diagnosis(completion_classification: "str | None", finish_reason_recorded: bool) -> str:
    """'confirmed' only when the provider itself reported MAX_TOKENS;
    'contradicted' when a finish reason WAS recorded and it was something
    else; 'unknown' whenever no finish reason was captured at all (as with
    Stage 5E's original MV++_0011_batch_01 attempt) — never inferred from a
    local JSON parse failure alone."""
    if not finish_reason_recorded:
        return "unknown"
    return "confirmed" if completion_classification == "max_tokens" else "contradicted"


# ══════════════════════════════════════════════════════════════════════════
# Task 7 — final Gemini-backfilled candidate assembly
# ══════════════════════════════════════════════════════════════════════════
def final_candidate_path(poem_id: str, repo_root: Path) -> Path:
    return repo_root / FINAL_CANDIDATE_DIR / f"{poem_id}.json"


def _finalize_envelope(
    candidate: dict[str, Any], *, poem_id: str, batch_ids: "list[str]", applied_paths: "list[str]",
    run_timestamp: str, derived_from: str,
) -> dict[str, Any]:
    final = copy.deepcopy(candidate)
    final["status"] = FINAL_CANDIDATE_STATUS
    final["candidate_tier"] = FINAL_CANDIDATE_TIER
    final[PROVENANCE_KEY] = {
        "stage": STAGE,
        "poem_id": poem_id,
        "provider": ex.PROVIDER_NAME,
        "batches_applied": list(batch_ids),
        "applied_paths": sorted(applied_paths),
        "run_timestamp": run_timestamp,
        "derived_from": derived_from,
        "note": (
            "Gemini (Vertex AI) model-assisted candidate. Not silver, not gold, "
            "not final, not native-speaker-reviewed, not human-approved. "
            "Reviewer availability for this pilot poem's language remains "
            "pending (pilot/PILOT_APPROVAL_CHECKLIST.md condition A)."
        ),
    }
    return final


def write_final_candidate(
    poem_id: str, working_copy: dict[str, Any], repo_root: Path, *,
    batch_ids: "list[str]", applied_paths: "list[str]", run_timestamp: str, derived_from: str,
) -> Path:
    final = _finalize_envelope(
        working_copy, poem_id=poem_id, batch_ids=batch_ids, applied_paths=applied_paths,
        run_timestamp=run_timestamp, derived_from=derived_from,
    )
    path = final_candidate_path(poem_id, repo_root)
    atomic_write_json(path, final)
    revalidate_final_candidate(path)
    return path


def ensure_mv1153_final_candidate(repo_root: Path, *, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> Path:
    """MV++_1153 has no batches in this stage's remaining set — its Gemini
    backfill was already completed and accepted in Stage 5D.2. This derives
    the Stage 5E final candidate directly from that accepted smoke
    candidate, WITHOUT calling Gemini again. Idempotent: safe to call
    repeatedly."""
    smoke_path = repo_root / SMOKE_CANDIDATE_DIR / "MV++_1153.json"
    smoke = _load_json(smoke_path)
    batch = _load_json(repo_root / STAGE5C_DIR / f"{PRIOR_COMPLETED_BATCH_ID}.json")
    return write_final_candidate(
        "MV++_1153", smoke, repo_root,
        batch_ids=[PRIOR_COMPLETED_BATCH_ID], applied_paths=list(batch["requested_field_paths"]),
        run_timestamp=now_fn().isoformat(), derived_from="stage_5d.2_smoke_candidate",
    )


def revalidate_final_candidate(path: Path) -> dict[str, Any]:
    record = _load_json(path)
    poem = pv._poem_from_candidate(record)
    validated_annotation = validate_model_payload_v1_1(record["annotation"], poem)
    original_poem, translated_poem = record["original_poem"], record["translated_poem"]
    errors = reviews = 0
    for index, entity in enumerate(validated_annotation.get("cultural_entities", [])):
        for issue in validate_cultural_grounding_v1_1(
            entity, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
        ):
            errors += issue.severity == "error"
            reviews += issue.severity != "error"
    for stanza in validated_annotation.get("stanzas", []):
        for index, expr in enumerate(stanza.get("metaphor_spans", [])):
            for issue in validate_figurative_grounding_v1_1(
                expr, original_poem, translated_poem, mode=GROUNDING_MODE_TRANSITIONAL_CANDIDATE, index=index,
            ):
                errors += issue.severity == "error"
                reviews += issue.severity != "error"
    return {"structural_validation": "valid", "grounding_errors": errors, "grounding_reviews": reviews}


# ══════════════════════════════════════════════════════════════════════════
# Task 2 — top-level sequential orchestration
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class StageExecutionResult:
    stopped_early: bool
    blocking_batch_id: str | None
    new_attempts_made: int
    batch_results: tuple[BatchExecutionResult, ...]
    finalized_poem_ids: tuple[str, ...]


def _poem_remaining_batch_ids(poem_id: str, repo_root: Path) -> tuple[str, ...]:
    """Every batch this poem has in the full authorized remaining set (all
    Stage 5C batches for this poem except the already-completed
    MV++_1153_batch_01) — independent of whatever subset an individual
    execute_remaining_batches call was scoped to via --batch-ids. Used to
    decide true poem completeness, never a possibly-partial local list."""
    return tuple(
        b["batch_id"] for b in discover_all_stage5c_batches(repo_root)
        if b["poem_id"] == poem_id and b["batch_id"] != PRIOR_COMPLETED_BATCH_ID
    )


def _poem_full_applied_paths(poem_id: str, repo_root: Path) -> "list[str]":
    """Every path ACTUALLY populated across ALL of this poem's batches (its
    full set, not a --batch-ids-scoped subset) — used only once
    _poem_fully_succeeded has confirmed every one of them is on disk. Uses
    `_effective_applied_paths_for_batch` (Stage 5E.7) so a batch completed
    via an adjudicated execution exception never has its one deferred
    human-review path silently counted as model-populated here."""
    paths: set[str] = set()
    for batch in discover_all_stage5c_batches(repo_root):
        if batch["poem_id"] == poem_id and batch["batch_id"] != PRIOR_COMPLETED_BATCH_ID:
            paths.update(_effective_applied_paths_for_batch(batch, repo_root))
    return sorted(paths)


def _poem_fully_succeeded(poem_id: str, repo_root: Path) -> bool:
    """True only if EVERY one of this poem's batches (its full set, not a
    --batch-ids subset) has a successful on-disk run_summary. A poem must
    never be finalized off of a partial, explicitly-scoped selection —
    otherwise a caller could request only a poem's last batch_id and get a
    final candidate that silently skipped its earlier batches. A batch
    completed via an adjudicated execution exception (Stage 5E.7) counts as
    done here too — the poem can still eventually be finalized, carrying its
    one deferred human-review path forward (Task 5 item 9)."""
    for batch_id in _poem_remaining_batch_ids(poem_id, repo_root):
        summary = read_batch_run_summary(batch_id, repo_root)
        if summary is None or not _is_batch_complete(summary.get("response_status")):
            return False
    return True


def _is_batch_complete(response_status: "str | None") -> bool:
    """True for a plain success, a Stage 5E.7 adjudicated-exception
    completion, or a Stage 5E.12 semantic-split completion — all three mean
    "resume past this batch, never re-attempt it." False for anything else
    (failure, or no attempt at all)."""
    return response_status in ("success", eex.ADJUDICATED_COMPLETION_STATUS, esplit.SPLIT_COMPLETION_STATUS)


def _effective_applied_paths_for_batch(batch: dict[str, Any], repo_root: Path) -> "tuple[str, ...]":
    """The paths actually populated by this batch, honestly — the full
    Stage 5C `requested_field_paths` for an ordinary successful batch or a
    fully-succeeded semantic split (Stage 5E.12 — every one of the original
    batch's paths was genuinely populated across its parts, nothing
    deferred), or (Stage 5E.7) only the exception's `remaining_model_paths`
    when the batch instead completed via an adjudicated execution exception.
    Never silently reports a deferred human-review path as model-populated."""
    summary = read_batch_run_summary(batch["batch_id"], repo_root)
    if summary is not None and summary.get("response_status") == eex.ADJUDICATED_COMPLETION_STATUS:
        exception_path = repo_root / eex.EXCEPTIONS_DIR / f"{batch['batch_id']}.json"
        if exception_path.exists():
            exception = eex.load_execution_exception(exception_path)
            return tuple(exception["remaining_model_paths"])
    return tuple(batch["requested_field_paths"])


def execute_remaining_batches(
    repo_root: Path,
    *,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    max_new_attempts: int = MAX_NEW_ATTEMPTS,
    batch_ids: "list[str] | None" = None,
) -> StageExecutionResult:
    """Sequential, single-pass execution of the authorized remaining batches
    (or an explicit --batch-ids subset, still validated against the
    authorized set). Stops immediately — returns without attempting any
    further batch — on the first failure or once max_new_attempts is
    reached. Resume-safe: batches already successful on disk are skipped
    without a provider call; a batch already recorded as FAILED blocks
    everything without a provider call.

    MV++_1153 has zero batches in the authorized remaining set (its one
    batch already succeeded in Stage 5D.2), so it never passes through the
    per-batch loop below. Its final candidate is ensured here instead,
    idempotently and with no provider call, so every run that reaches this
    point leaves all five final-candidate slots consistent."""
    if max_new_attempts > MAX_NEW_ATTEMPTS:
        raise ValueError(f"max_new_attempts must be <= {MAX_NEW_ATTEMPTS}; got {max_new_attempts}.")

    all_remaining = discover_remaining_batches(repo_root) if batch_ids is None else validate_batch_selection(batch_ids, repo_root)

    working_copies: dict[str, dict[str, Any]] = {}
    poem_applied_paths: dict[str, set] = {}
    new_attempts_made = 0
    results: list[BatchExecutionResult] = []
    finalized: list[str] = []

    if not final_candidate_path("MV++_1153", repo_root).exists():
        ensure_mv1153_final_candidate(repo_root, now_fn=now_fn)
        finalized.append("MV++_1153")

    for batch in all_remaining:
        batch_id, poem_id, batch_index = batch["batch_id"], batch["poem_id"], batch["batch_index"]
        existing = read_batch_run_summary(batch_id, repo_root)

        if existing is not None:
            if _is_batch_complete(existing.get("response_status")):
                working_copies[poem_id] = starting_working_copy(poem_id, repo_root)
                poem_applied_paths.setdefault(poem_id, set()).update(
                    _effective_applied_paths_for_batch(batch, repo_root)
                )
                if _poem_fully_succeeded(poem_id, repo_root) and not final_candidate_path(poem_id, repo_root).exists():
                    write_final_candidate(
                        poem_id, working_copies[poem_id], repo_root,
                        batch_ids=list(_poem_remaining_batch_ids(poem_id, repo_root)),
                        applied_paths=_poem_full_applied_paths(poem_id, repo_root),
                        run_timestamp=now_fn().isoformat(), derived_from="stage_5e_batches",
                    )
                    finalized.append(poem_id)
                continue
            return StageExecutionResult(True, batch_id, new_attempts_made, tuple(results), tuple(finalized))

        if new_attempts_made >= max_new_attempts:
            return StageExecutionResult(True, batch_id, new_attempts_made, tuple(results), tuple(finalized))

        working_copy = working_copies.get(poem_id)
        if working_copy is None:
            working_copy = starting_working_copy(poem_id, repo_root)
            working_copies[poem_id] = working_copy
        previous_snapshot = copy.deepcopy(working_copy)

        result = execute_one_batch(batch, working_copy, repo_root, client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn)
        new_attempts_made += 1
        results.append(result)

        if not result.accepted:
            return StageExecutionResult(True, batch_id, new_attempts_made, tuple(results), tuple(finalized))

        verify_checkpoint_continuity(previous_snapshot, result.patched_candidate, poem_applied_paths.get(poem_id, set()))
        working_copies[poem_id] = result.patched_candidate
        poem_applied_paths.setdefault(poem_id, set()).update(result.applied_paths)
        atomic_write_json(checkpoint_path(poem_id, batch_index, repo_root), result.patched_candidate)

        if _poem_fully_succeeded(poem_id, repo_root):
            write_final_candidate(
                poem_id, result.patched_candidate, repo_root,
                batch_ids=list(_poem_remaining_batch_ids(poem_id, repo_root)),
                applied_paths=_poem_full_applied_paths(poem_id, repo_root),
                run_timestamp=now_fn().isoformat(), derived_from="stage_5e_batches",
            )
            finalized.append(poem_id)

    return StageExecutionResult(False, None, new_attempts_made, tuple(results), tuple(finalized))


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.2 — a single, explicitly-authorized new attempt for the one
# batch execute_remaining_batches's normal resume path permanently blocks
# ══════════════════════════════════════════════════════════════════════════
def execute_forensic_retry(
    repo_root: Path,
    *,
    expected_batch_id: str,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BatchExecutionResult:
    """A single, explicitly-scoped new attempt for a batch that already has
    a recorded FAILED run. This is NOT a general retry mechanism — Stage 5E's
    own resume rule ("a failed batch blocks everything") deliberately has no
    code path in `execute_remaining_batches` that would ever call the
    provider again for it on its own; that omission is intentional there.
    This function exists only because a project supervisor can explicitly,
    separately authorize exactly one new attempt for exactly one named
    batch (Stage 5E.2) — `expected_batch_id` is required and checked so a
    stale or misconfigured caller can never silently retry a DIFFERENT
    batch than the one it was actually authorized for.

    Writes only to a freshly computed next-attempt directory (via
    `next_attempt_dir_for_batch` — never attempt_01, never overwriting any
    prior attempt). Calls `execute_one_batch` exactly once (one provider
    attempt, no retry within that call). Never proceeds to any other batch.
    Writes a checkpoint only on success — never a final candidate, since a
    single batch succeeding never means every batch for its poem has."""
    next_action = next_runnable_batch(repo_root)
    if next_action.action != "blocked":
        raise StageBlockedError(
            f"execute_forensic_retry: no batch is currently blocked (next_action={next_action.action!r}); "
            "there is nothing to retry."
        )
    if next_action.blocking_batch_id != expected_batch_id:
        raise StageBlockedError(
            f"execute_forensic_retry: the currently blocked batch is "
            f"{next_action.blocking_batch_id!r}, not the expected {expected_batch_id!r}."
        )
    batch = next_action.batch
    poem_id = batch["poem_id"]
    out_dir = next_attempt_dir_for_batch(batch["batch_id"], repo_root)

    working_copy = starting_working_copy(poem_id, repo_root)
    previously_applied_paths: set = set()
    for other_batch_id in _poem_remaining_batch_ids(poem_id, repo_root):
        if other_batch_id == batch["batch_id"]:
            continue
        summary = read_batch_run_summary(other_batch_id, repo_root)
        if summary is not None and _is_batch_complete(summary.get("response_status")):
            other_batch = next(b for b in discover_all_stage5c_batches(repo_root) if b["batch_id"] == other_batch_id)
            previously_applied_paths.update(_effective_applied_paths_for_batch(other_batch, repo_root))
    previous_snapshot = copy.deepcopy(working_copy)

    result = execute_one_batch(
        batch, working_copy, repo_root,
        client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn, out_dir=out_dir,
    )
    if not result.accepted:
        return result

    verify_checkpoint_continuity(previous_snapshot, result.patched_candidate, previously_applied_paths)
    atomic_write_json(checkpoint_path(poem_id, batch["batch_index"], repo_root), result.patched_candidate)
    return result


# ══════════════════════════════════════════════════════════════════════════
# Task 11 — dry run (zero provider calls)
# ══════════════════════════════════════════════════════════════════════════
def next_attempt_dir_for_batch(batch_id: str, repo_root: Path) -> Path:
    """Pure path computation (Stage 5E.1 Task 2/8) — never creates anything
    on disk. Reuses gemini_backfill_executor_v1_1.next_attempt_dir's general
    numbered-attempt-directory logic (already exercised by Stage 5D) against
    this batch's own directory: since Stage 5E always writes into an
    `attempt_01/` SUBdirectory (never Stage 5D's flat legacy-root layout),
    `_has_legacy_root_attempt` is always False here and the numbered-subdir
    scan alone determines the answer — a batch with an existing `attempt_01`
    always yields `attempt_02` next, so a future retry could never overwrite
    it. No attempt_02 directory is created by calling this."""
    return ex.next_attempt_dir(repo_root / STAGE5E_RUNS_DIR / batch_id)


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.7 — adjudicated execution exceptions (Tasks 3-6). A single,
# narrowly-scoped path (never a generic "drop any field" mechanism) for
# requesting a batch's REMAINING approved paths after one specific path has
# been forensically adjudicated (Stage 5E.6) and deferred to human
# translation review. Nothing in this section is ever invoked with a real
# provider client by this stage — it exists so a future, separately
# authorized run can use it, gated by explicit CLI flags (Task 7).
# ══════════════════════════════════════════════════════════════════════════
class AdjudicatedExceptionBlockedError(RuntimeError):
    """Raised when an adjudicated-exception reduced request cannot proceed
    — an invalid/missing exception, an already-completed original batch, or
    the excluded path no longer matching its expected untouched state."""


def build_reduced_batch_for_exception(original_batch: dict[str, Any], exception: dict[str, Any]) -> dict[str, Any]:
    """A copy of `original_batch` with `requested_field_paths` replaced by
    the exception's 35 `remaining_model_paths` (Task 4) — same `batch_id`/
    `poem_id` (provenance is the original batch, per Task 4), so every
    existing, unmodified validation stage in `execute_one_batch` (strict
    JSON, exact-path-coverage, type validation, transactional application,
    structural + grounding validation) runs unchanged against the reduced
    path set. The excluded path is structurally absent from the result —
    never in the prompt's requested-path list, never in the response
    schema's path enum, never eligible for patch application. Pure; never
    mutates `original_batch` or `exception`."""
    reduced = dict(original_batch)
    reduced["requested_field_paths"] = list(exception["remaining_model_paths"])
    reduced["requested_path_count"] = len(exception["remaining_model_paths"])
    reduced["adjudicated_exception_applied"] = True
    reduced["adjudicated_excluded_paths"] = list(exception["excluded_paths"])
    return reduced


def _adjudicated_exception_preflight(
    repo_root: Path, exception_path: Path,
) -> "tuple[dict[str, Any], dict[str, Any], dict[str, Any]]":
    """Shared read-only preflight for both the dry-run and live paths (Task
    3's five required-true conditions, minus the CLI acknowledgement flag
    itself, which only main() can check). Returns
    (exception, stage5c_batch, working_copy) on success. Raises
    AdjudicatedExceptionBlockedError or eex.ExecutionExceptionError
    otherwise. Never writes anything."""
    if not exception_path.exists():
        raise AdjudicatedExceptionBlockedError(f"exception file not found: {exception_path}")
    exception = eex.load_execution_exception(exception_path)
    eex.validate_execution_exception(exception, repo_root)  # raises on any of the 15 safety rules

    batch_id, poem_id = exception["original_batch_id"], exception["poem_id"]
    existing = read_batch_run_summary(batch_id, repo_root)
    if existing is not None and _is_batch_complete(existing.get("response_status")):
        raise AdjudicatedExceptionBlockedError(
            f"{batch_id} already has a completed attempt ({existing.get('response_status')!r}); "
            "an adjudicated reduced request is only for a batch with no successful accepted attempt."
        )

    stage5c_batch = _load_json(repo_root / STAGE5C_DIR / f"{batch_id}.json")

    working_copy = starting_working_copy(poem_id, repo_root)
    excluded_path = exception["excluded_paths"][0]
    pre_backfill = load_pre_backfill_candidate(poem_id, repo_root)
    current_value = pv.get_value_at_path(working_copy, excluded_path)
    original_value = pv.get_value_at_path(pre_backfill, excluded_path)
    if current_value != original_value:
        raise AdjudicatedExceptionBlockedError(
            f"excluded path {excluded_path!r} no longer matches its Stage 5A value; "
            "an adjudicated reduced request requires it to remain untouched."
        )

    return exception, stage5c_batch, working_copy


@dataclass(frozen=True)
class AdjudicatedExceptionDryRunReport:
    poem_id: str
    original_batch_id: str
    original_path_count: int
    reduced_model_request_paths: int
    deferred_human_review_paths: int
    excluded_path: str
    next_attempt_dir: str
    actual_provider_attempts: int
    provider_client_created: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dry_run_adjudicated_exception(repo_root: Path, exception_path: Path) -> AdjudicatedExceptionDryRunReport:
    """Zero provider calls, zero writes, no client constructed (Task 7's
    dry-run requirements). Raises the same errors `_adjudicated_exception_preflight`
    would on an invalid/blocked exception — a caller (the CLI) is expected
    to catch and report those as a clean rejection, never a traceback."""
    exception, _stage5c_batch, _working_copy = _adjudicated_exception_preflight(repo_root, exception_path)
    batch_id = exception["original_batch_id"]
    return AdjudicatedExceptionDryRunReport(
        poem_id=exception["poem_id"],
        original_batch_id=batch_id,
        original_path_count=exception["original_requested_path_count"],
        reduced_model_request_paths=len(exception["remaining_model_paths"]),
        deferred_human_review_paths=len(exception["excluded_paths"]),
        excluded_path=exception["excluded_paths"][0],
        next_attempt_dir=str(next_attempt_dir_for_batch(batch_id, repo_root)),
        actual_provider_attempts=0,
        provider_client_created=False,
    )


def execute_adjudicated_reduced_retry(
    repo_root: Path,
    *,
    exception_path: Path,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BatchExecutionResult:
    """The actual reduced (35-path) live attempt (Task 4/5) — never called
    by this stage against a real provider; exists for a future, separately
    authorized run, gated by the CLI's explicit
    --allow-adjudicated-exception/--exception-file flags (Task 3/7). Exactly
    one provider attempt (via the unmodified execute_one_batch), written to
    a freshly computed attempt_03 (never overwriting attempt_01/attempt_02).
    On success: relabels ONLY this freshly-written attempt's own
    run_summary.json response_status from "success" to
    eex.ADJUDICATED_COMPLETION_STATUS (never touches attempt_01/attempt_02's
    preserved evidence) and writes a checkpoint differing from Stage 5A at
    exactly the 35 reduced paths — never a final candidate (four more
    batches remain for this poem)."""
    exception, stage5c_batch, working_copy = _adjudicated_exception_preflight(repo_root, exception_path)
    batch_id, poem_id, batch_index = exception["original_batch_id"], exception["poem_id"], stage5c_batch["batch_index"]
    reduced_batch = build_reduced_batch_for_exception(stage5c_batch, exception)
    out_dir = next_attempt_dir_for_batch(batch_id, repo_root)

    previous_snapshot = copy.deepcopy(working_copy)
    result = execute_one_batch(
        reduced_batch, working_copy, repo_root,
        client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn, out_dir=out_dir,
    )
    if not result.accepted:
        return result

    verify_checkpoint_continuity(previous_snapshot, result.patched_candidate, set())

    # Relabel this SAME freshly-written attempt (never a later mutation of
    # already-preserved evidence) so the ledger and every reporting surface
    # can never mistake this batch for fully model-completed.
    run_summary_path = out_dir / "run_summary.json"
    run_summary = _load_json(run_summary_path)
    run_summary["response_status"] = eex.ADJUDICATED_COMPLETION_STATUS
    run_summary["adjudicated_exception_file"] = str(exception_path)
    run_summary["model_populated_path_count"] = len(exception["remaining_model_paths"])
    run_summary["deferred_human_review_path_count"] = len(exception["excluded_paths"])
    atomic_write_json(run_summary_path, run_summary)

    atomic_write_json(checkpoint_path(poem_id, batch_index, repo_root), result.patched_candidate)
    return BatchExecutionResult(
        True, batch_id, poem_id, eex.ADJUDICATED_COMPLETION_STATUS, run_summary,
        result.patched_candidate, result.applied_paths,
    )


# ══════════════════════════════════════════════════════════════════════════
# Stage 5E.12 — semantic execution splits (Tasks 4-6). A general, poem/
# batch-ID-agnostic mechanism for executing ANY 31-40-path batch as several
# smaller, semantically-complete parts instead of one large request. Nothing
# in this section is ever invoked with a real provider client by this
# stage — it exists so a future, separately authorized run can use it.
# ══════════════════════════════════════════════════════════════════════════
class ExecutionSplitBlockedError(RuntimeError):
    """Raised when a split part cannot proceed — an invalid/missing overlay,
    an already-completed original batch, a not-yet-accepted earlier part, or
    every part already succeeded (use finalize_execution_split instead)."""


def split_part_checkpoint_path(poem_id: str, batch_index: int, part_index: int, repo_root: Path) -> Path:
    return repo_root / CHECKPOINT_DIR / poem_id / f"after_batch_{batch_index:02d}_part_{part_index:02d}.json"


def _split_preflight(repo_root: Path, overlay_path: Path) -> "tuple[dict[str, Any], dict[str, Any]]":
    """Shared read-only preflight (Task 4/6). Returns (overlay, stage5c_batch)
    on success. Raises ExecutionSplitBlockedError or
    execution_split_v1_1.ExecutionSplitError otherwise. Never writes
    anything."""
    if not overlay_path.exists():
        raise ExecutionSplitBlockedError(f"execution split overlay not found: {overlay_path}")
    overlay = esplit.load_execution_split(overlay_path)
    esplit.validate_execution_split(overlay, repo_root)  # raises on any of the 15 rules

    batch_id = overlay["original_batch_id"]
    existing = read_batch_run_summary(batch_id, repo_root)
    if existing is not None and _is_batch_complete(existing.get("response_status")):
        raise ExecutionSplitBlockedError(
            f"{batch_id} already has a completed attempt ({existing.get('response_status')!r}); "
            "a semantic split execution is only for a batch with no successful accepted attempt."
        )

    stage5c_batch = _load_json(repo_root / STAGE5C_DIR / f"{batch_id}.json")
    return overlay, stage5c_batch


def _next_unstarted_or_blocking_part_index(overlay: dict[str, Any], repo_root: Path) -> int:
    """The lowest part_index that has not yet succeeded (Task 6) — 1 if no
    part has ever been attempted. Parts are always resolved strictly in
    order; a value greater than the last part's index means every part
    already succeeded."""
    for part in overlay["parts"]:
        summary = read_batch_run_summary(part["part_id"], repo_root)
        if summary is None or summary.get("response_status") != "success":
            return part["part_index"]
    return len(overlay["parts"]) + 1


def split_starting_working_copy(overlay: dict[str, Any], part_index: int, repo_root: Path) -> dict[str, Any]:
    """Part 1 starts from the poem's normal latest checkpoint (the same
    resolution `starting_working_copy` already uses — the official
    after_batch_<N-1>.json, or Stage 5A if none exists yet). Every later
    part starts from the PRECEDING PART's own intermediate checkpoint, never
    from an official after_batch_<N>.json (which does not exist until the
    whole split succeeds) and never from any other part's checkpoint."""
    poem_id, batch_index = overlay["poem_id"], overlay["batch_index"]
    if part_index == 1:
        return starting_working_copy(poem_id, repo_root)
    prev_checkpoint = split_part_checkpoint_path(poem_id, batch_index, part_index - 1, repo_root)
    if not prev_checkpoint.exists():
        raise ExecutionSplitBlockedError(
            f"part {part_index} cannot start: preceding part's checkpoint {prev_checkpoint} does not exist."
        )
    return _load_json(prev_checkpoint)


def _cumulative_applied_paths_before_part(overlay: dict[str, Any], part_index: int) -> "set[str]":
    """Every path already applied by all EARLIER parts, derived directly
    from the overlay's own deterministic part definitions (never by
    re-reading provider artifacts) — used only for the continuity
    defense-in-depth check."""
    applied: set[str] = set()
    for part in overlay["parts"]:
        if part["part_index"] < part_index:
            applied.update(part["paths"])
    return applied


def execute_split_part(
    repo_root: Path,
    *,
    overlay_path: Path,
    client_factory: ClientFactory,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BatchExecutionResult:
    """Executes exactly the next unstarted (or currently blocking) part of a
    semantic execution split (Task 5/6) — never called by this stage
    against a real provider; exists for a future, separately authorized run.
    Exactly one provider attempt per part (via the unmodified
    execute_one_batch, using execution_split_v1_1's anti-repetition prompt
    builder), written to that part's own attempt_01 (never overwriting
    another part's or the original batch's evidence). On a part's success:
    writes ONLY that part's own intermediate checkpoint
    (after_batch_<N>_part_<M>.json) — never the official
    after_batch_<N>.json, and never any other part's checkpoint. Only once
    every part has succeeded does a caller's subsequent
    finalize_execution_split call write the official checkpoint and mark
    the original batch complete."""
    overlay, stage5c_batch = _split_preflight(repo_root, overlay_path)
    part_index = _next_unstarted_or_blocking_part_index(overlay, repo_root)
    if part_index > len(overlay["parts"]):
        raise ExecutionSplitBlockedError(
            f"every part of {overlay['original_batch_id']}'s split already succeeded; "
            "call finalize_execution_split instead of executing another part."
        )

    part_batch = esplit.part_batch_for(overlay, stage5c_batch, part_index)
    working_copy = split_starting_working_copy(overlay, part_index, repo_root)
    previous_snapshot = copy.deepcopy(working_copy)
    out_dir = batch_attempt_dir(part_batch["batch_id"], repo_root)

    result = execute_one_batch(
        part_batch, working_copy, repo_root,
        client_factory=client_factory, sleep_fn=sleep_fn, now_fn=now_fn, out_dir=out_dir,
        prompt_bundle_builder=esplit.build_split_part_prompt_bundle,
    )
    if not result.accepted:
        return result

    previously_applied = _cumulative_applied_paths_before_part(overlay, part_index)
    verify_checkpoint_continuity(previous_snapshot, result.patched_candidate, previously_applied)

    part_checkpoint = split_part_checkpoint_path(overlay["poem_id"], overlay["batch_index"], part_index, repo_root)
    atomic_write_json(part_checkpoint, result.patched_candidate)
    return result


def all_split_parts_succeeded(overlay: dict[str, Any], repo_root: Path) -> bool:
    for part in overlay["parts"]:
        summary = read_batch_run_summary(part["part_id"], repo_root)
        if summary is None or summary.get("response_status") != "success":
            return False
    return True


def finalize_execution_split(
    repo_root: Path, *, overlay_path: Path, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> BatchExecutionResult:
    """Writes the OFFICIAL after_batch_<N>.json and marks the original batch
    complete (Task 6) — ONLY once every one of the split's parts has
    independently succeeded. Never called by this stage. Writes a new,
    freshly numbered attempt for the ORIGINAL batch_id (never attempt_01,
    which remains the preserved failed evidence) recording
    execution_split_v1_1.SPLIT_COMPLETION_STATUS and referencing every
    part's own attempt directory — never a copy or mutation of any part's
    or the original failed attempt's own preserved run_summary.json."""
    overlay, _stage5c_batch = _split_preflight(repo_root, overlay_path)
    if not all_split_parts_succeeded(overlay, repo_root):
        raise ExecutionSplitBlockedError(
            f"not every part of {overlay['original_batch_id']}'s split has succeeded yet; "
            "the official checkpoint is never written until all parts pass."
        )

    batch_id, poem_id, batch_index = overlay["original_batch_id"], overlay["poem_id"], overlay["batch_index"]
    last_part_index = len(overlay["parts"])
    final_checkpoint = _load_json(split_part_checkpoint_path(poem_id, batch_index, last_part_index, repo_root))

    out_dir = next_attempt_dir_for_batch(batch_id, repo_root)
    run_summary = {
        "stage": STAGE, "batch_id": batch_id, "poem_id": poem_id,
        "response_status": esplit.SPLIT_COMPLETION_STATUS,
        "patch_application_status": "applied",
        "parse_status": "success",
        "provider_attempts": len(overlay["parts"]),
        "retries": 0,
        "checkpoint_written": True,
        "requested_path_count": overlay["original_path_count"],
        "split_overlay_file": str(overlay_path),
        "split_part_ids": [part["part_id"] for part in overlay["parts"]],
        "timestamp": now_fn().isoformat(),
        "retry_recommendation": "not_applicable",
    }
    atomic_write_json(out_dir / "run_summary.json", run_summary)
    atomic_write_json(checkpoint_path(poem_id, batch_index, repo_root), final_checkpoint)

    applied_paths = tuple(sorted(p for part in overlay["parts"] for p in part["paths"]))
    return BatchExecutionResult(
        True, batch_id, poem_id, esplit.SPLIT_COMPLETION_STATUS, run_summary, final_checkpoint, applied_paths,
    )


@dataclass(frozen=True)
class ExecutionSplitDryRunReport:
    poem_id: str
    original_batch_id: str
    original_path_count: int
    part_count: int
    part_path_counts: "tuple[int, ...]"
    part_semantic_unit_counts: "tuple[int, ...]"
    part_token_budgets: "tuple[int, ...]"
    starting_checkpoint: str
    next_part_index: int
    next_part_attempt_dir: str
    actual_provider_attempts: int
    provider_client_created: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dry_run_execution_split(repo_root: Path, overlay_path: Path) -> ExecutionSplitDryRunReport:
    """Zero provider calls, zero writes, no client constructed (Task 8).
    Raises the same errors `_split_preflight` would on an invalid/blocked
    overlay — a caller (a future CLI) is expected to catch and report those
    as a clean rejection, never a traceback."""
    overlay, _stage5c_batch = _split_preflight(repo_root, overlay_path)
    part_index = _next_unstarted_or_blocking_part_index(overlay, repo_root)
    part_path_counts = tuple(p["path_count"] for p in overlay["parts"])
    part_semantic_unit_counts = tuple(len(p["semantic_units"]) for p in overlay["parts"])
    part_token_budgets = tuple(ex.determine_output_token_budget(p["paths"]) for p in overlay["parts"])

    poem_id, batch_index = overlay["poem_id"], overlay["batch_index"]
    if part_index == 1:
        starting = f"after_batch_{batch_index - 1:02d}.json (or the Stage 5A pre-backfill candidate if absent)"
    else:
        starting = str(split_part_checkpoint_path(poem_id, batch_index, part_index - 1, repo_root))

    if part_index <= len(overlay["parts"]):
        next_part_id = overlay["parts"][part_index - 1]["part_id"]
        next_attempt_dir = str(batch_attempt_dir(next_part_id, repo_root))
    else:
        next_attempt_dir = "all parts already succeeded — call finalize_execution_split"

    return ExecutionSplitDryRunReport(
        poem_id=poem_id,
        original_batch_id=overlay["original_batch_id"],
        original_path_count=overlay["original_path_count"],
        part_count=len(overlay["parts"]),
        part_path_counts=part_path_counts,
        part_semantic_unit_counts=part_semantic_unit_counts,
        part_token_budgets=part_token_budgets,
        starting_checkpoint=starting,
        next_part_index=part_index,
        next_part_attempt_dir=next_attempt_dir,
        actual_provider_attempts=0,
        provider_client_created=False,
    )


@dataclass(frozen=True)
class DryRunReport:
    remaining_batch_count: int
    execution_order: tuple[str, ...]
    prior_completed_batch_excluded: bool
    planned_new_attempts: int
    actual_attempts: int
    next_action: str
    blocking_batch_id: str | None
    output_dirs: tuple[str, ...]
    first_unfinished_batch_id: str | None
    next_attempt_dir_for_first_unfinished: str | None
    derived_output_token_budget_for_first_unfinished: int | None
    thinking_level_for_first_unfinished: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def dry_run(repo_root: Path) -> DryRunReport:
    remaining = discover_remaining_batches(repo_root)
    next_action = next_runnable_batch(repo_root)
    ledger = read_execution_ledger(repo_root)
    pending = sum(1 for e in ledger if e.status == "not_attempted")

    first_unfinished_batch_id: str | None = None
    next_attempt_dir: str | None = None
    derived_budget: int | None = None
    thinking_level: str | None = None
    if next_action.action in ("run", "blocked") and next_action.batch is not None:
        first_unfinished_batch_id = next_action.batch["batch_id"]
        next_attempt_dir = str(next_attempt_dir_for_batch(first_unfinished_batch_id, repo_root))
        derived_budget = ex.determine_output_token_budget(
            next_action.batch["requested_field_paths"], semantic_units=next_action.batch.get("semantic_unit_types", ()),
        )
        # Stage 5E.15: same generation_settings_summary() every real request
        # is built from -- never a second, independently-maintained value.
        thinking_level = ex.generation_settings_summary(max_output_tokens=derived_budget)["thinking_level"]

    return DryRunReport(
        remaining_batch_count=len(remaining),
        execution_order=tuple(b["batch_id"] for b in remaining),
        prior_completed_batch_excluded=all(b["batch_id"] != PRIOR_COMPLETED_BATCH_ID for b in remaining),
        planned_new_attempts=pending,
        actual_attempts=0,
        next_action=next_action.action,
        blocking_batch_id=next_action.blocking_batch_id,
        output_dirs=tuple(str(batch_attempt_dir(b["batch_id"], repo_root)) for b in remaining),
        first_unfinished_batch_id=first_unfinished_batch_id,
        next_attempt_dir_for_first_unfinished=next_attempt_dir,
        derived_output_token_budget_for_first_unfinished=derived_budget,
        thinking_level_for_first_unfinished=thinking_level,
    )


# ══════════════════════════════════════════════════════════════════════════
# Working-tree cleanliness check (CLI Task 3)
# ══════════════════════════════════════════════════════════════════════════
def working_tree_is_clean(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(repo_root), capture_output=True, text=True, check=True,
    )
    return result.stdout.strip() == ""


# ══════════════════════════════════════════════════════════════════════════
# Task 8 — completeness audit
# ══════════════════════════════════════════════════════════════════════════
def _deferred_paths_for_poem(poem_id: str, repo_root: Path) -> "set[str]":
    """Every path deferred to human translation review via a Stage 5E.7
    execution exception for this poem — read-only; an empty set for a poem
    with no active exception. Used so completeness accounting can always
    distinguish "deferred" from "Gemini returned null" and never silently
    reports a deferred path as model-populated (Task 6)."""
    exceptions_dir = repo_root / eex.EXCEPTIONS_DIR
    if not exceptions_dir.exists():
        return set()
    deferred: set[str] = set()
    for exception_file in exceptions_dir.glob("*.json"):
        exception = eex.load_execution_exception(exception_file)
        if exception.get("poem_id") == poem_id:
            deferred.update(exception.get("excluded_paths", []))
    return deferred


def completeness_audit_for_poem(poem_id: str, repo_root: Path) -> dict[str, Any]:
    plan = _load_json(repo_root / STAGE5B_DIR / f"{poem_id}.json")
    final_path = final_candidate_path(poem_id, repo_root)
    deferred_paths = _deferred_paths_for_poem(poem_id, repo_root)
    result: dict[str, Any] = {
        "poem_id": poem_id,
        "stage5b_requested_field_count": len(plan["requested_field_paths"]),
        "human_review_only_field_count": len(plan["human_review_only_paths"]),
        "intentionally_nullable_field_count": len(plan["intentionally_nullable_paths"]),
        "deferred_via_adjudicated_exception_count": len(deferred_paths),
        "final_candidate_exists": final_path.exists(),
    }
    if not final_path.exists():
        result.update({
            "fields_populated_by_gemini": 0, "null_values_returned": 0, "empty_lists_returned": 0,
            "successful_batch_count": 0, "provider_attempt_count": 0,
            "schema_validation_result": "not_attempted", "grounding_result": "not_attempted",
            "unchanged_reliable_field_count": None,
            "unresolved_review_items": len(plan["human_review_only_paths"]) + len(deferred_paths),
        })
        return result

    final = _load_json(final_path)
    populated = nulls = empties = 0
    for path in plan["requested_field_paths"]:
        if path in deferred_paths:
            # Never counted as populated/null/empty here — reported
            # separately via deferred_via_adjudicated_exception_count, so a
            # human-review deferral can never be mistaken for a genuine
            # Gemini-returned null or an actual model-populated value.
            continue
        value = pv.get_value_at_path(final, path)
        if value is None:
            nulls += 1
        elif isinstance(value, list) and len(value) == 0:
            empties += 1
        else:
            populated += 1

    if poem_id == "MV++_1153":
        batch_ids = [PRIOR_COMPLETED_BATCH_ID]
    else:
        batch_ids = [b["batch_id"] for b in discover_remaining_batches(repo_root) if b["poem_id"] == poem_id]
    # Each successful batch used exactly one provider attempt (Stage 5E's
    # one-attempt-per-batch rule); MV++_1153's single successful attempt was
    # Stage 5D.2's attempt_02 (attempt_01 failed and is not counted as a
    # "successful response" here — this figure reports successful attempts
    # that actually contributed to the final candidate's content).
    attempt_count = len(batch_ids)

    validation = revalidate_final_candidate(final_path)
    result.update({
        "fields_populated_by_gemini": populated,
        "null_values_returned": nulls,
        "empty_lists_returned": empties,
        "successful_batch_count": len(batch_ids),
        "provider_attempt_count": attempt_count,
        "schema_validation_result": validation["structural_validation"],
        "grounding_result": {"errors": validation["grounding_errors"], "reviews": validation["grounding_reviews"]},
        "unresolved_review_items": len(plan["human_review_only_paths"]) + len(deferred_paths),
    })
    return result


def completeness_audit(repo_root: Path) -> dict[str, Any]:
    per_poem = {poem_id: completeness_audit_for_poem(poem_id, repo_root) for poem_id in PILOT_POEM_IDS}
    total_stage5b_paths = sum(p["stage5b_requested_field_count"] for p in per_poem.values())
    represented = sum(
        p["fields_populated_by_gemini"] + p["null_values_returned"] + p["empty_lists_returned"]
        for p in per_poem.values() if p["final_candidate_exists"]
    )
    return {
        "per_poem": per_poem,
        "total_stage5b_model_backfill_paths": total_stage5b_paths,
        "total_represented_across_final_candidates": represented,
        "all_paths_represented": total_stage5b_paths == represented,
        "mv1153_path_count": per_poem["MV++_1153"]["stage5b_requested_field_count"],
    }


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — sequential-only, dry-run-default CLI
# ══════════════════════════════════════════════════════════════════════════
class CliRejection(RuntimeError):
    """Raised for any CLI-level safety rejection. Always caught by main()
    and turned into a clean message + non-zero exit code."""


def build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="gemini_pilot_execution_v1_1",
        description="Stage 5E — sequential, one-attempt-per-batch execution of the 16 remaining Stage 5C batches.",
    )
    parser.add_argument("--execute", action="store_true", help="Make live provider calls. Default: dry run.")
    parser.add_argument("--remaining-batches", action="store_true", help="Confirm bulk execution of the authorized remaining-batch set.")
    parser.add_argument("--batch-ids", default=None, help="Optional comma-separated subset of the authorized remaining batch IDs.")
    parser.add_argument("--max-new-attempts", type=int, default=None)
    parser.add_argument("--acknowledge-billing", action="store_true")
    parser.add_argument("--stop-on-first-failure", action="store_true")
    parser.add_argument("--forensic-override-batch-id", default=None)
    parser.add_argument(
        "--allow-adjudicated-exception", action="store_true",
        help="Explicit acknowledgement that a validated Stage 5E.7 execution exception should be used "
             "for the batch it names, requesting only its remaining approved paths. Must be paired with "
             "--exception-file; neither alone has any effect.",
    )
    parser.add_argument(
        "--exception-file", default=None,
        help="Path to a Stage 5E.7 execution-exception JSON file. Must be paired with "
             "--allow-adjudicated-exception; neither alone has any effect.",
    )
    parser.add_argument("--repo-root", default=".")
    return parser


def _validate_cli_args_for_adjudicated_execute(args: Any) -> None:
    if not args.acknowledge_billing:
        raise CliRejection("--execute with --exception-file requires --acknowledge-billing.")
    if not args.stop_on_first_failure:
        raise CliRejection("--execute with --exception-file requires --stop-on-first-failure.")


def _validate_cli_args_for_execute(args: Any) -> "list[str] | None":
    if not args.remaining_batches:
        raise CliRejection("--execute requires --remaining-batches.")
    if args.max_new_attempts is None:
        raise CliRejection("--execute requires --max-new-attempts.")
    if args.max_new_attempts > MAX_NEW_ATTEMPTS:
        raise CliRejection(f"--max-new-attempts must be <= {MAX_NEW_ATTEMPTS}; got {args.max_new_attempts}.")
    if not args.acknowledge_billing:
        raise CliRejection("--execute requires --acknowledge-billing.")
    if not args.stop_on_first_failure:
        raise CliRejection("--execute requires --stop-on-first-failure.")

    batch_ids = None
    if args.batch_ids:
        batch_ids = [b.strip() for b in args.batch_ids.split(",") if b.strip()]
        if PRIOR_COMPLETED_BATCH_ID in batch_ids:
            raise CliRejection(f"{PRIOR_COMPLETED_BATCH_ID} already completed in Stage 5D.2 and must never be called again.")
    return batch_ids


def main(
    argv: "list[str] | None" = None,
    *,
    client_factory: "ClientFactory | None" = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    stdout: Any = None,
) -> int:
    import sys

    out_stream = stdout if stdout is not None else sys.stdout
    args = build_arg_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()

    # Stage 5E.7 Task 7 items 1/2: the two exception flags are only ever
    # meaningful together — one without the other is always rejected,
    # whether this is a dry run or a live --execute.
    if bool(args.exception_file) != bool(args.allow_adjudicated_exception):
        print(
            "REJECTED: --exception-file and --allow-adjudicated-exception must be supplied together.",
            file=out_stream,
        )
        return 2

    if args.exception_file and args.allow_adjudicated_exception:
        exception_path = Path(args.exception_file)
        if not args.execute:
            try:
                report = dry_run_adjudicated_exception(repo_root, exception_path)
            except (eex.ExecutionExceptionError, AdjudicatedExceptionBlockedError) as exc:
                print(f"REJECTED: {exc}", file=out_stream)
                return 2
            print(json.dumps(report.to_dict(), indent=2), file=out_stream)
            return 0

        try:
            _validate_cli_args_for_adjudicated_execute(args)
            try:
                ex.load_gemini_config()
            except ex.ConfigError as exc:
                raise CliRejection(str(exc)) from exc
            if not ex.check_adc_available():
                raise CliRejection("Application Default Credentials are not available.")
            if not working_tree_is_clean(repo_root):
                raise CliRejection("working tree was dirty before Stage 5E started; commit or stash first.")
            _adjudicated_exception_preflight(repo_root, exception_path)  # raises on any invalid/blocked exception
        except (CliRejection, eex.ExecutionExceptionError, AdjudicatedExceptionBlockedError) as exc:
            print(f"REJECTED: {exc}", file=out_stream)
            return 2

        factory = client_factory or ex.default_client_factory
        result = execute_adjudicated_reduced_retry(
            repo_root, exception_path=exception_path, client_factory=factory, sleep_fn=sleep_fn, now_fn=now_fn,
        )
        print(json.dumps({
            "accepted": result.accepted,
            "response_status": result.response_status,
            "batch_id": result.batch_id,
        }, indent=2), file=out_stream)
        # Task 7 item 14: never automatically continue to any later batch,
        # whether this reduced request succeeded or failed — a caller who
        # wants to resume the rest of Stage 5E must issue its own,
        # separate, freshly-authorized invocation.
        return 0 if result.accepted else 1

    if not args.execute:
        report = dry_run(repo_root)
        print(json.dumps(report.to_dict(), indent=2), file=out_stream)
        return 0

    try:
        batch_ids = _validate_cli_args_for_execute(args)

        if batch_ids is not None:
            validate_batch_selection(batch_ids, repo_root)  # raises BatchDiscoveryError on bad IDs

        try:
            ex.load_gemini_config()
        except ex.ConfigError as exc:
            raise CliRejection(str(exc)) from exc
        if not ex.check_adc_available():
            raise CliRejection("Application Default Credentials are not available.")

        if not working_tree_is_clean(repo_root):
            raise CliRejection("working tree was dirty before Stage 5E started; commit or stash first.")

        next_action = next_runnable_batch(repo_root)
        if next_action.action == "blocked" and args.forensic_override_batch_id != next_action.blocking_batch_id:
            raise CliRejection(
                f"batch {next_action.blocking_batch_id!r} previously failed; execution is blocked. "
                "A forensic override is required to proceed, and none was given for this batch."
            )
        if next_action.action == "run" and args.forensic_override_batch_id is None:
            # normal path: nothing to override, just proceed
            pass

    except (CliRejection, BatchDiscoveryError) as exc:
        print(f"REJECTED: {exc}", file=out_stream)
        return 2

    factory = client_factory or ex.default_client_factory
    result = execute_remaining_batches(
        repo_root, client_factory=factory, sleep_fn=sleep_fn, now_fn=now_fn,
        max_new_attempts=args.max_new_attempts, batch_ids=batch_ids,
    )
    print(json.dumps({
        "stopped_early": result.stopped_early,
        "blocking_batch_id": result.blocking_batch_id,
        "new_attempts_made": result.new_attempts_made,
        "finalized_poem_ids": list(result.finalized_poem_ids),
    }, indent=2), file=out_stream)
    return 1 if result.stopped_early else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
