"""Stage 5E.12 — reusable semantic execution-split overlays.

A Stage 5C batch with 31-40 requested paths occasionally risks a MAX_TOKENS
truncation even at the raised 24576-token tier (Stage 5E.11) — most
concretely, `MV++_0073_batch_04`'s confirmed failure, where the model
covered only 3 of 36 requested paths before entering a 255-entry
repetition loop. A blind larger-budget retry does not address a
generation-time repetition failure, so this module provides a general,
poem/batch-ID-agnostic mechanism to split ANY batch's requested paths into
smaller, semantically complete sub-batches ("parts") that can each be
executed as their own, fully independent provider request — reusing the
exact same request/response/validation pipeline every ordinary batch
already uses, never a parallel one.

This module never calls a model, provider, or proxy, and never reads or
writes a credential. It only:
  - reads an EXISTING Stage 5C batch (never modifies it);
  - groups that batch's own requested paths into the SAME semantic units
    Stage 5C itself already defines (`execution_batch_v1_1.semantic_unit_key`
    / `group_paths_by_unit`) — a unit is never split across parts;
  - derives a deterministic part plan and writes it to
    `pilot/execution_splits/stage5e/<batch_id>.json` as PLANNING DATA ONLY —
    the overlay never authorizes, triggers, or performs a provider call by
    itself;
  - validates an overlay's internal consistency against the real Stage 5C
    batch and Stage 5B plan it was derived from.

The overlay MECHANISM (this module, plus the split-execution orchestration
in `gemini_pilot_execution_v1_1.py`) is fully generic — no poem ID or batch
ID is hardcoded anywhere in this module's logic. Only the concrete overlay
DATA FILE for `MV++_0073_batch_04` is batch-specific, exactly the same
relationship Stage 5C's own generic batching logic has to any one of its
individual batch JSON files.

See docs/GEMINI_PILOT_EXECUTION_STAGE5E.md's "Stage 5E.12" section for the
full contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import execution_batch_v1_1 as eb
from . import patch_v1_1 as pv
from .prompt_v1_1 import PromptBundle

OVERLAY_SCHEMA_VERSION = "5E.12.1"
OVERLAY_TYPE = "semantic_execution_split"
STAGE5C_DIR = Path("pilot") / "backfill_requests" / "stage5c"
STAGE5B_DIR = Path("pilot") / "backfill_requests" / "stage5b"
SPLITS_DIR = Path("pilot") / "execution_splits" / "stage5e"

PREFERRED_PART_COUNT = 3
MAX_PATHS_PER_PART = 20

# The status a batch's ledger-visible completion carries once every one of
# its split parts has independently succeeded — deliberately distinct from
# plain "success" everywhere this project reports on batch completion, so a
# split-completed batch's provenance (three independent provider calls, not
# one) always remains visible to anyone reading its run summary directly.
SPLIT_COMPLETION_STATUS = "completed_via_semantic_split"


class ExecutionSplitError(Exception):
    """Raised by `validate_execution_split` for any structural, coverage, or
    authorization-boundary violation. Never raised for a merely undesirable
    (but internally consistent) split shape."""


# ══════════════════════════════════════════════════════════════════════════
# Task 1/3 — generic semantic-unit-preserving split planning
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SplitPlan:
    parts: "tuple[tuple[str, ...], ...]"  # each element: that part's sorted paths
    target_part_count: int
    fallback_reason: "str | None"


def _minimum_feasible_parts(unit_items: "list[tuple[tuple, list[str]]]", max_paths_per_part: int) -> int:
    """Greedy-optimal minimum number of CONTIGUOUS parts (in the given,
    already-canonical unit order) such that no part exceeds
    `max_paths_per_part` — optimal because, for a fixed order and a
    never-split-a-unit constraint, greedily filling each part as full as
    possible before starting a new one is the standard, provably minimal
    contiguous bin-packing result. Raises if a single unit alone exceeds the
    cap (that unit can never be placed in any part without splitting it)."""
    parts = 0
    current = 0
    for unit_key, paths in unit_items:
        if len(paths) > max_paths_per_part:
            raise ExecutionSplitError(
                f"semantic unit {unit_key!r} alone has {len(paths)} paths, exceeding the "
                f"{max_paths_per_part}-path-per-part cap; it cannot be split further without "
                "breaking unit integrity, so no valid split exists at this cap."
            )
        if current == 0 or current + len(paths) > max_paths_per_part:
            parts += 1
            current = len(paths)
        else:
            current += len(paths)
    return parts


def _chunk_units_into_parts(
    unit_items: "list[tuple[tuple, list[str]]]", target_parts: int, max_paths_per_part: int,
) -> "tuple[tuple[str, ...], ...]":
    """Balanced, deterministic, order-preserving chunking of `unit_items`
    into exactly `target_parts` non-empty contiguous groups (the first
    `n_units % target_parts` groups absorb one extra unit each), each
    returned as a sorted tuple of paths. Raises if any resulting part would
    exceed `max_paths_per_part` (defensive — the caller is expected to have
    already confirmed `target_parts >= _minimum_feasible_parts(...)`)."""
    n_units = len(unit_items)
    base, remainder = divmod(n_units, target_parts)
    groups: list[list[tuple[tuple, list[str]]]] = []
    idx = 0
    for i in range(target_parts):
        size = base + (1 if i < remainder else 0)
        groups.append(unit_items[idx: idx + size])
        idx += size
    parts: list[tuple[str, ...]] = []
    for group in groups:
        if not group:
            raise ExecutionSplitError(f"balanced chunking into {target_parts} parts produced an empty part.")
        part_paths = sorted(p for _unit, paths in group for p in paths)
        if len(part_paths) > max_paths_per_part:
            raise ExecutionSplitError(
                f"part would contain {len(part_paths)} paths, exceeding the {max_paths_per_part}-path cap "
                f"at target_parts={target_parts}."
            )
        parts.append(tuple(part_paths))
    return tuple(parts)


def plan_semantic_split(
    paths: "list[str] | tuple[str, ...]",
    *, preferred_parts: int = PREFERRED_PART_COUNT, max_paths_per_part: int = MAX_PATHS_PER_PART,
) -> SplitPlan:
    """Pure, offline, poem/batch-ID-agnostic split planner (Task 1/3). Groups
    `paths` into Stage 5C's own semantic units (never split), then targets
    exactly `preferred_parts` contiguous, balanced parts — falling back to
    the smallest safe part count only when 3 parts genuinely cannot satisfy
    `max_paths_per_part` without splitting a unit, or when fewer than
    `preferred_parts` units exist at all. Never targets MORE than
    `preferred_parts` merely for smaller parts — only ever forced upward by
    the hard cap, or downward by too few units to fill 3 non-empty parts."""
    grouped = eb.group_paths_by_unit(paths)
    unit_items = list(grouped.items())
    n_units = len(unit_items)
    if n_units == 0:
        raise ExecutionSplitError("cannot split an empty path list.")

    min_parts = _minimum_feasible_parts(unit_items, max_paths_per_part)

    if n_units < preferred_parts:
        target = n_units
        reason = (
            f"only {n_units} semantic unit(s) present; {preferred_parts} non-empty parts cannot be "
            "formed without splitting a unit, so the smallest safe part count (one unit per part) is used."
        )
    elif min_parts > preferred_parts:
        target = min_parts
        reason = (
            f"{preferred_parts} parts cannot satisfy the {max_paths_per_part}-path-per-part cap without "
            f"splitting a semantic unit; {min_parts} is the minimum part count that respects both the cap "
            "and unit integrity."
        )
    else:
        target = preferred_parts
        reason = None

    parts = _chunk_units_into_parts(unit_items, target, max_paths_per_part)
    return SplitPlan(parts=parts, target_part_count=target, fallback_reason=reason)


# ══════════════════════════════════════════════════════════════════════════
# Task 3 — overlay construction (planning data only; never self-authorizing)
# ══════════════════════════════════════════════════════════════════════════
def _semantic_unit_descriptor(unit_key: tuple) -> dict[str, Any]:
    unit_type = unit_key[0]
    if unit_type == eb.UNIT_FIGURATIVE_EXPRESSION:
        return {"unit_type": unit_type, "stanza_index": unit_key[1], "metaphor_span_index": unit_key[2]}
    if unit_type == eb.UNIT_CULTURAL_ENTITY:
        return {"unit_type": unit_type, "cultural_entity_index": unit_key[1]}
    if unit_type == eb.UNIT_TRANSLATION_LOSS:
        return {"unit_type": unit_type, "stanza_index": unit_key[1]}
    return {"unit_type": unit_type}


def build_execution_split_overlay(
    batch: dict[str, Any],
    *, preferred_parts: int = PREFERRED_PART_COUNT, max_paths_per_part: int = MAX_PATHS_PER_PART,
) -> dict[str, Any]:
    """Builds the JSON-serializable overlay document for ANY Stage 5C batch
    dict (Task 3). Generic — reads only `batch`'s own fields, never a
    hardcoded poem or batch ID. Pure; makes no filesystem write itself (the
    caller decides where/whether to persist it) and never modifies the
    `batch` dict it was given."""
    original_paths = tuple(batch["requested_field_paths"])
    plan = plan_semantic_split(original_paths, preferred_parts=preferred_parts, max_paths_per_part=max_paths_per_part)
    grouped = eb.group_paths_by_unit(original_paths)

    parts_doc = []
    for part_index, part_paths in enumerate(plan.parts, start=1):
        part_path_set = set(part_paths)
        part_units = [
            {**_semantic_unit_descriptor(unit_key), "path_count": len(unit_paths)}
            for unit_key, unit_paths in grouped.items()
            if part_path_set.issuperset(unit_paths)
        ]
        parts_doc.append({
            "part_id": f"{batch['batch_id']}_part_{part_index:02d}",
            "part_index": part_index,
            "paths": list(part_paths),
            "path_count": len(part_paths),
            "semantic_units": part_units,
        })

    return {
        "overlay_schema_version": OVERLAY_SCHEMA_VERSION,
        "overlay_type": OVERLAY_TYPE,
        "original_batch_id": batch["batch_id"],
        "poem_id": batch["poem_id"],
        "batch_index": batch["batch_index"],
        "original_path_count": len(original_paths),
        "original_requested_paths": sorted(original_paths),
        "target_part_count": plan.target_part_count,
        "part_count_fallback_reason": plan.fallback_reason,
        "parts": parts_doc,
        "modifies_stage5b": False,
        "modifies_stage5c": False,
        "self_authorizes_execution": False,
        "authorization_statement": (
            "This overlay is execution PLANNING data only. It does not authorize, trigger, or "
            "perform any provider call. A separate, explicit authorization is required before any "
            "part may be executed."
        ),
        "created_by_stage": "5E.12",
    }


def split_overlay_path(batch_id: str, repo_root: Path) -> Path:
    return repo_root / SPLITS_DIR / f"{batch_id}.json"


def load_execution_split(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Task 4 — split-overlay validation (15 rules; generic, no hardcoded IDs)
# ══════════════════════════════════════════════════════════════════════════
def validate_execution_split(overlay: dict[str, Any], repo_root: Path) -> None:
    """Raises ExecutionSplitError on the first violated rule; returns None
    when every rule passes. Every check re-derives ground truth from the
    real, on-disk Stage 5C batch and Stage 5B plan — never trusts the
    overlay's own cached counts/claims without independently recomputing
    them. Contains no poem ID or batch ID of its own; operates entirely on
    whatever `overlay["original_batch_id"]`/`overlay["poem_id"]` name."""
    original_batch_id = overlay.get("original_batch_id")
    poem_id = overlay.get("poem_id")
    if not original_batch_id or not poem_id:
        raise ExecutionSplitError("overlay is missing original_batch_id or poem_id.")

    # 1. The original Stage 5C batch exists.
    batch_path = repo_root / STAGE5C_DIR / f"{original_batch_id}.json"
    if not batch_path.exists():
        raise ExecutionSplitError(f"original Stage 5C batch {original_batch_id!r} does not exist at {batch_path}.")
    with batch_path.open("r", encoding="utf-8") as f:
        real_batch = json.load(f)
    real_paths = list(real_batch["requested_field_paths"])

    # 2. Path count matches (independently recomputed, not merely the
    #    overlay's own cached original_path_count).
    if overlay.get("original_path_count") != len(real_paths):
        raise ExecutionSplitError(
            f"overlay original_path_count {overlay.get('original_path_count')!r} does not match the real "
            f"Stage 5C batch's {len(real_paths)} requested paths."
        )
    if sorted(overlay.get("original_requested_paths") or []) != sorted(real_paths):
        raise ExecutionSplitError("overlay original_requested_paths does not match the real Stage 5C batch.")

    parts = overlay.get("parts") or []
    if not parts:
        raise ExecutionSplitError("overlay contains no parts.")

    all_split_paths: list[str] = []
    seen_paths: set[str] = set()
    for part in parts:
        part_paths = part.get("paths") or []
        # 7. Each part contains at least one path.
        if not part_paths:
            raise ExecutionSplitError(f"part {part.get('part_id')!r} contains zero paths.")
        # 8. No part exceeds the max-paths-per-part cap.
        if len(part_paths) > MAX_PATHS_PER_PART:
            raise ExecutionSplitError(
                f"part {part.get('part_id')!r} has {len(part_paths)} paths, exceeding the "
                f"{MAX_PATHS_PER_PART}-path cap."
            )
        for path in part_paths:
            # 4. No path duplicated across parts.
            if path in seen_paths:
                raise ExecutionSplitError(f"path {path!r} appears in more than one part.")
            seen_paths.add(path)
            all_split_paths.append(path)

    real_path_set = set(real_paths)
    split_path_set = set(all_split_paths)
    # 3. Every split path belongs to the original batch.
    unknown = split_path_set - real_path_set
    if unknown:
        raise ExecutionSplitError(f"split contains path(s) not in the original batch: {sorted(unknown)}")
    # 5. No path omitted.
    missing = real_path_set - split_path_set
    if missing:
        raise ExecutionSplitError(f"split omits path(s) from the original batch: {sorted(missing)}")
    # 6. The union is exactly the original path set (implied by 3+5, asserted
    #    explicitly for a clear, self-contained failure message).
    if split_path_set != real_path_set:
        raise ExecutionSplitError("split path union does not exactly equal the original batch's path set.")

    # 9. Semantic units remain intact — every unit's full path list must
    #    fall entirely within exactly one part.
    grouped = eb.group_paths_by_unit(real_paths)
    part_path_sets = [set(part.get("paths") or []) for part in parts]
    for unit_key, unit_paths in grouped.items():
        unit_path_set = set(unit_paths)
        owning_parts = [i for i, pps in enumerate(part_path_sets) if unit_path_set & pps]
        if len(owning_parts) != 1 or not unit_path_set.issubset(part_path_sets[owning_parts[0]]):
            raise ExecutionSplitError(f"semantic unit {unit_key!r} is split across more than one part.")

    # 10. Part ordering is deterministic: part_index values are exactly
    #     1..N, in ascending order, matching the parts list's own order.
    expected_indices = list(range(1, len(parts) + 1))
    actual_indices = [part.get("part_index") for part in parts]
    if actual_indices != expected_indices:
        raise ExecutionSplitError(f"part_index values {actual_indices} are not the deterministic sequence {expected_indices}.")

    # 11/12/13. No human-review-only, intentionally-nullable, or
    #           non-model-backfill (i.e. reliable/already-populated) leaf
    #           field is present — re-derived from patch_v1_1's own
    #           canonical field classification, never from a poem-specific
    #           list, so this check is fully generic.
    for path in split_path_set:
        leaf = pv.leaf_field(path)
        if leaf in pv.HUMAN_REVIEW_FIELDS:
            raise ExecutionSplitError(f"path {path!r} targets a human-review-only field ({leaf!r}).")
        if leaf in pv.INTENTIONALLY_NULLABLE_FIELDS:
            raise ExecutionSplitError(f"path {path!r} targets an intentionally-nullable field ({leaf!r}).")
        if leaf not in pv.MODEL_BACKFILL_FIELDS:
            raise ExecutionSplitError(f"path {path!r} targets a field ({leaf!r}) that is not model-backfill-allowed.")

    # 14. The overlay does not modify Stage 5B or Stage 5C.
    if overlay.get("modifies_stage5b") is not False:
        raise ExecutionSplitError("overlay must declare modifies_stage5b: false.")
    if overlay.get("modifies_stage5c") is not False:
        raise ExecutionSplitError("overlay must declare modifies_stage5c: false.")

    # 15. The overlay does not self-authorize execution.
    if overlay.get("self_authorizes_execution") is not False:
        raise ExecutionSplitError("overlay must declare self_authorizes_execution: false.")


# ══════════════════════════════════════════════════════════════════════════
# Task 5 — per-part batch dict + anti-repetition prompt addendum (generic;
# reused by ANY future split, never wired into the default, unsplit prompt
# path every ordinary/completed batch already uses).
# ══════════════════════════════════════════════════════════════════════════
def part_batch_for(overlay: dict[str, Any], real_batch: dict[str, Any], part_index: int) -> dict[str, Any]:
    """Constructs a Stage-5C-batch-SHAPED dict for exactly one part, reusing
    every field from the real, on-disk original batch except `batch_id`
    (replaced with the part's own ID) and `requested_field_paths`/
    `requested_path_count` (replaced with that part's own subset). This
    lets the part be executed through the EXACT SAME `execute_one_batch`
    pipeline every ordinary batch already uses, unchanged."""
    part = next(p for p in overlay["parts"] if p["part_index"] == part_index)
    part_batch = dict(real_batch)
    part_batch["batch_id"] = part["part_id"]
    part_batch["requested_field_paths"] = list(part["paths"])
    part_batch["requested_path_count"] = len(part["paths"])
    part_batch["batching_reason"] = (
        f"semantic execution split of {overlay['original_batch_id']} "
        f"(part {part_index} of {len(overlay['parts'])})."
    )
    return part_batch


_ANTI_REPETITION_INSTRUCTION = """
SPLIT-PART RESPONSE CONTRACT (generic; applies to every split-part request):
- Return exactly one patch object per requested field path listed above — never more, never fewer.
- Never repeat a path. Each requested path must appear in the "patches" array exactly once.
- The number of patch objects in "patches" must equal the number of requested field paths.
- Before returning the JSON, verify internally that every "path" value is unique and that none is missing.
""".strip()


def build_split_part_prompt_bundle(part_batch: dict[str, Any], candidate: dict[str, Any]) -> PromptBundle:
    """Generic, reusable prompt builder for exactly one split part (Task 5).
    Reuses `execution_batch_v1_1.build_prompt_bundle_from_batch` UNCHANGED to
    get the normal, already-tested batch-derived prompt/schema shape, then
    appends the anti-repetition contract to the user prompt only. Never
    modifies `prompt_v1_1.py`'s core builder, so no already-completed or
    unrelated batch's prompt shape is affected by this function's
    existence."""
    from dataclasses import replace

    bundle = eb.build_prompt_bundle_from_batch(part_batch, candidate)
    return replace(bundle, user_prompt=bundle.user_prompt + "\n\n" + _ANTI_REPETITION_INSTRUCTION)
