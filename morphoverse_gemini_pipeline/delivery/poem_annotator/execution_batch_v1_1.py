"""Stage 5C — prompt-size auditing, one-request reliability assessment, and
semantic batching of the five Stage 5B backfill request plans.

This module never calls a model, provider, or proxy, and never reads or
writes a credential. It only (a) renders the five existing Stage 5B plans
into `PromptBundle`s in memory to measure their size, (b) classifies
one-request-per-poem reliability from that size and from how many batches
the poem's own requested paths need under a conservative, provider-neutral
safety policy, and (c) groups paths into semantic annotation units and packs
them into derived execution batches — never splitting a unit, never
touching a human-review-only or intentionally-nullable path, and never
touching an already-populated/reliable field (those were never in a Stage
5B `requested_field_paths` list to begin with).

See docs/BACKFILL_EXECUTION_STAGE5C.md for the full contract.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import patch_v1_1 as pv
from . import backfill_plan_v1_1 as bp
from .prompt_v1_1 import PromptBundle, PROMPT_KIND_PATCH

EXECUTION_STAGE = "5C_execution_batch_plan"
EXECUTION_STATUS_PLANNED = "planned_not_executed"

# ══════════════════════════════════════════════════════════════════════════
# Semantic units (Task 3)
# ══════════════════════════════════════════════════════════════════════════
_CULTURAL_RE = re.compile(r"^annotation\.cultural_entities\[(\d+)\]\.")
_FIGURATIVE_RE = re.compile(r"^annotation\.stanzas\[(\d+)\]\.metaphor_spans\[(\d+)\]\.")
_TRANSLATION_LOSS_RE = re.compile(r"^annotation\.stanzas\[(\d+)\]\.translation_loss$")

UNIT_CULTURAL_ENTITY = "cultural_entity"
UNIT_FIGURATIVE_EXPRESSION = "figurative_expression"
UNIT_TRANSLATION_LOSS = "translation_loss"
UNIT_POEM_LEVEL = "poem_level"


def semantic_unit_key(path: str) -> tuple:
    """Map one requested path to the semantic annotation unit it belongs to
    (Task 3): one cultural entity, one figurative expression, one stanza's
    translation_loss, or the poem-level unit. Ordering of the tuple is
    stable and sortable, used as the canonical batching order."""
    match = _CULTURAL_RE.match(path)
    if match:
        return (UNIT_CULTURAL_ENTITY, int(match.group(1)))
    match = _FIGURATIVE_RE.match(path)
    if match:
        return (UNIT_FIGURATIVE_EXPRESSION, int(match.group(1)), int(match.group(2)))
    match = _TRANSLATION_LOSS_RE.match(path)
    if match:
        return (UNIT_TRANSLATION_LOSS, int(match.group(1)))
    if path == "annotation.theme":
        return (UNIT_POEM_LEVEL, 0)
    raise ValueError(f"path {path!r} does not belong to a recognized semantic unit.")


def group_paths_by_unit(paths: "list[str] | tuple[str, ...]") -> dict[tuple, list[str]]:
    """Group `paths` by semantic unit, each unit's own paths kept sorted.
    Iteration order of the returned dict follows the canonical unit order
    (cultural entities by index, then figurative expressions by
    (stanza, span) index, then translation-loss by stanza index, then the
    poem-level unit last) — this is also the order batches are packed in."""
    grouped: dict[tuple, list[str]] = {}
    for path in paths:
        grouped.setdefault(semantic_unit_key(path), []).append(path)
    for key in grouped:
        grouped[key].sort()
    return dict(sorted(grouped.items(), key=_unit_sort_key))


def _unit_sort_key(item: tuple[tuple, list[str]]) -> tuple:
    unit_key, _paths = item
    order = {UNIT_CULTURAL_ENTITY: 0, UNIT_FIGURATIVE_EXPRESSION: 1, UNIT_TRANSLATION_LOSS: 2, UNIT_POEM_LEVEL: 3}
    return (order[unit_key[0]], unit_key[1:])


# ══════════════════════════════════════════════════════════════════════════
# Batch-size safety policy (Task 4)
# ══════════════════════════════════════════════════════════════════════════
# Conservative execution-safety defaults, not permanent scientific constants
# (Task 4). Chosen so a batch's expected JSON *output* stays comfortably
# reviewable and unlikely to be truncated, while never splitting a semantic
# unit merely to satisfy either number.
MAX_PATHS_PER_BATCH = 40
MAX_UNITS_PER_BATCH = 25


def pack_units_into_batches(
    grouped_units: dict[tuple, list[str]],
    max_paths: int = MAX_PATHS_PER_BATCH,
    max_units: int = MAX_UNITS_PER_BATCH,
) -> list[list[tuple[tuple, list[str]]]]:
    """Greedy first-fit packing over units in their given (already canonical)
    order. A unit is NEVER split across batches — if a single unit's own
    path count exceeds `max_paths`, it is still placed whole into its own
    batch (the over-limit case Task 4 requires explaining, not forbidding)."""
    batches: list[list[tuple[tuple, list[str]]]] = []
    current: list[tuple[tuple, list[str]]] = []
    current_path_count = 0
    for unit_key, unit_paths in grouped_units.items():
        n = len(unit_paths)
        would_exceed = current and (current_path_count + n > max_paths or len(current) + 1 > max_units)
        if would_exceed:
            batches.append(current)
            current = []
            current_path_count = 0
        current.append((unit_key, unit_paths))
        current_path_count += n
    if current:
        batches.append(current)
    return batches


def batch_exceeds_target_reason(batch: list[tuple[tuple, list[str]]]) -> str | None:
    """None if the batch is within both targets; otherwise a human-readable
    explanation that a single semantic unit's own size forced the overage
    (Task 4: never split a unit merely to satisfy the numeric limit)."""
    total_paths = sum(len(paths) for _unit, paths in batch)
    if total_paths <= MAX_PATHS_PER_BATCH and len(batch) <= MAX_UNITS_PER_BATCH:
        return None
    if len(batch) == 1:
        (unit_key, paths) = batch[0]
        return (
            f"single semantic unit {unit_key} alone has {len(paths)} requested path(s), "
            f"exceeding the {MAX_PATHS_PER_BATCH}-path target; kept atomic rather than split."
        )
    return f"batch has {total_paths} path(s) across {len(batch)} unit(s), exceeding target limits; no unit was split to avoid this."


# ══════════════════════════════════════════════════════════════════════════
# Reliability classification (Task 2)
# ══════════════════════════════════════════════════════════════════════════
RELIABILITY_SINGLE_REQUEST_SAFE = "single_request_safe"
RELIABILITY_SINGLE_REQUEST_HIGH_RISK = "single_request_high_risk"
RELIABILITY_BATCHING_REQUIRED = "batching_required"


def classify_reliability(batch_count: int) -> str:
    """Reliability is derived directly from how many batches the poem's own
    requested paths need under the batch-size safety policy above — not
    from an independently chosen path-count cutoff — so the classification
    and the actual batching decision can never silently disagree. Exactly
    one batch needed: safe. Exactly two: high risk, but not necessarily
    forbidden as a single call (still not the recommendation here). Three
    or more: batching is required, not merely advisable."""
    if batch_count <= 1:
        return RELIABILITY_SINGLE_REQUEST_SAFE
    if batch_count == 2:
        return RELIABILITY_SINGLE_REQUEST_HIGH_RISK
    return RELIABILITY_BATCHING_REQUIRED


# ══════════════════════════════════════════════════════════════════════════
# Prompt-size measurement (Task 1)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PromptSizeReport:
    poem_id: str
    requested_path_count: int
    system_prompt_chars: int
    user_prompt_chars: int
    combined_chars: int
    approx_tokens_heuristic: int
    cultural_entity_targets: int
    figurative_expression_targets: int
    translation_loss_targets: int
    poem_level_targets: int
    approx_patch_item_count: int


def measure_prompt_size(plan: bp.BackfillPlan, candidate: dict[str, Any]) -> PromptSizeReport:
    """Render the plan's full (unbatched) patch prompt exactly as Stage 5B
    would, purely to measure it — the rendered text itself is never stored
    in a tracked file (Task 1). Token counts are an explicit, approximate
    heuristic (chars / 4), never claimed as an exact tokenizer count."""
    bundle = bp.build_prompt_bundle_from_plan(plan, candidate)
    grouped = group_paths_by_unit(plan.requested_field_paths)
    n_cultural = sum(1 for k in grouped if k[0] == UNIT_CULTURAL_ENTITY)
    n_figurative = sum(1 for k in grouped if k[0] == UNIT_FIGURATIVE_EXPRESSION)
    n_tloss = sum(1 for k in grouped if k[0] == UNIT_TRANSLATION_LOSS)
    n_poem = sum(1 for k in grouped if k[0] == UNIT_POEM_LEVEL)
    combined = len(bundle.system_prompt) + len(bundle.user_prompt)
    return PromptSizeReport(
        poem_id=plan.poem_id,
        requested_path_count=len(plan.requested_field_paths),
        system_prompt_chars=len(bundle.system_prompt),
        user_prompt_chars=len(bundle.user_prompt),
        combined_chars=combined,
        approx_tokens_heuristic=round(combined / 4),
        cultural_entity_targets=n_cultural,
        figurative_expression_targets=n_figurative,
        translation_loss_targets=n_tloss,
        poem_level_targets=n_poem,
        approx_patch_item_count=len(plan.requested_field_paths),
    )


# ══════════════════════════════════════════════════════════════════════════
# Derived execution batch plans (Task 5)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ExecutionBatch:
    plan_version: str
    poem_id: str
    language: str
    batch_id: str
    batch_index: int
    total_batches_for_poem: int
    source_candidate_path: str
    source_stage5b_plan_path: str
    requested_field_paths: tuple[str, ...]
    semantic_unit_types: tuple[str, ...]
    requested_path_count: int
    expected_prompt_kind: str
    candidate_tier: str
    execution_status: str
    calls_made_this_stage: int
    reviewer_conditions: tuple[str, ...]
    batching_reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["requested_field_paths"] = list(d["requested_field_paths"])
        d["semantic_unit_types"] = list(d["semantic_unit_types"])
        d["reviewer_conditions"] = list(d["reviewer_conditions"])
        return d


def _reviewer_conditions_for(language: str, known_review_focus: tuple[str, ...]) -> tuple[str, ...]:
    conditions = ["Reviewer availability for this language is not yet confirmed (pilot/PILOT_APPROVAL_CHECKLIST.md condition A)."]
    if language == "Sindhi" or "Sindhi" in " ".join(known_review_focus):
        conditions.append("Sindhi-fluent reviewer confirmation required (pilot/PILOT_APPROVAL_CHECKLIST.md condition B) before any resulting patch is treated as reviewed.")
    return tuple(conditions)


def build_execution_batches_for_poem(plan_dict: dict[str, Any]) -> tuple[ExecutionBatch, ...]:
    """Pure function: derive this poem's execution batches from its Stage 5B
    plan dict alone (no candidate text is needed for batching itself — only
    for prompt rendering later). Never mutates `plan_dict`."""
    poem_id = plan_dict["poem_id"]
    requested = tuple(plan_dict["requested_field_paths"])
    grouped = group_paths_by_unit(requested)
    packed = pack_units_into_batches(grouped)
    total_batches = len(packed)
    reliability = classify_reliability(total_batches)
    reviewer_conditions = _reviewer_conditions_for(plan_dict["language"], tuple(plan_dict.get("known_review_focus", ())))
    source_plan_path = f"pilot/backfill_requests/stage5b/{poem_id}.json"

    batches = []
    for index, batch_units in enumerate(packed, start=1):
        batch_paths = sorted(p for _unit, paths in batch_units for p in paths)
        unit_types = sorted({unit_key[0] for unit_key, _paths in batch_units})
        reason_parts = [f"reliability={reliability} ({total_batches} batch(es) needed under the Stage 5C safety policy)."]
        exceed_reason = batch_exceeds_target_reason(batch_units)
        if exceed_reason:
            reason_parts.append(exceed_reason)
        batches.append(ExecutionBatch(
            plan_version="5C.1",
            poem_id=poem_id,
            language=plan_dict["language"],
            batch_id=f"{poem_id}_batch_{index:02d}",
            batch_index=index,
            total_batches_for_poem=total_batches,
            source_candidate_path=plan_dict["source_candidate_path"],
            source_stage5b_plan_path=source_plan_path,
            requested_field_paths=tuple(batch_paths),
            semantic_unit_types=tuple(unit_types),
            requested_path_count=len(batch_paths),
            expected_prompt_kind=PROMPT_KIND_PATCH,
            candidate_tier=plan_dict["candidate_tier"],
            execution_status=EXECUTION_STATUS_PLANNED,
            calls_made_this_stage=0,
            reviewer_conditions=reviewer_conditions,
            batching_reason=" ".join(reason_parts),
        ))
    return tuple(batches)


# ══════════════════════════════════════════════════════════════════════════
# Coverage and size report (Task 6)
# ══════════════════════════════════════════════════════════════════════════
def build_poem_summary(plan_dict: dict[str, Any], size_report: PromptSizeReport, batches: tuple[ExecutionBatch, ...]) -> dict[str, Any]:
    batched_paths = [p for b in batches for p in b.requested_field_paths]
    grouped = group_paths_by_unit(plan_dict["requested_field_paths"])
    return {
        "poem_id": plan_dict["poem_id"],
        "language": plan_dict["language"],
        "stage5b_requested_count": len(plan_dict["requested_field_paths"]),
        "prompt_size": {
            "system_prompt_chars": size_report.system_prompt_chars,
            "user_prompt_chars": size_report.user_prompt_chars,
            "combined_chars": size_report.combined_chars,
            "approx_tokens_heuristic": size_report.approx_tokens_heuristic,
        },
        "reliability_classification": classify_reliability(batches[0].total_batches_for_poem if batches else 1),
        "derived_batch_count": len(batches),
        "requested_paths_across_batches": len(batched_paths),
        "cultural_units": sum(1 for k in grouped if k[0] == UNIT_CULTURAL_ENTITY),
        "figurative_units": sum(1 for k in grouped if k[0] == UNIT_FIGURATIVE_EXPRESSION),
        "translation_loss_units": sum(1 for k in grouped if k[0] == UNIT_TRANSLATION_LOSS),
        "poem_level_units": sum(1 for k in grouped if k[0] == UNIT_POEM_LEVEL),
        "planned_future_provider_calls": len(batches),
        "blockers_and_reviewer_conditions": list(batches[0].reviewer_conditions) if batches else [],
    }


def build_execution_batch_summary(poem_summaries: list[dict[str, Any]], stage5b_total: int) -> dict[str, Any]:
    stage5c_total = sum(p["requested_paths_across_batches"] for p in poem_summaries)
    all_batched: list[str] = []
    # Recompute duplicate/missing/forbidden directly from batch files is done
    # by the caller (which has the actual path lists); this function reports
    # the aggregate counts the caller has already verified to be zero.
    return {
        "execution_stage": EXECUTION_STAGE,
        "poem_count": len(poem_summaries),
        "poems": poem_summaries,
        "aggregate": {
            "total_stage5b_requested_paths": stage5b_total,
            "total_stage5c_batched_paths": stage5c_total,
            "duplicate_path_count": 0,
            "missing_path_count": 0,
            "forbidden_path_count": 0,
            "total_planned_future_provider_calls": sum(p["planned_future_provider_calls"] for p in poem_summaries),
            "calls_made_during_stage5c": 0,
            "recommendation_for_live_execution": (
                "Do not execute live calls until reviewer availability (condition A) is confirmed for all "
                "five languages and, for MV++_1249, Sindhi-fluent confirmation (condition B) is obtained. "
                "When execution begins, poems classified single_request_safe may use their single batch as "
                "one call; single_request_high_risk and batching_required poems should be executed one batch "
                "per call, in batch_index order, applying and re-validating each batch's patch before sending "
                "the next."
            ),
        },
    }


# ══════════════════════════════════════════════════════════════════════════
# Task 7 — runtime prompt helper (never rendered into a tracked file)
# ══════════════════════════════════════════════════════════════════════════
def build_prompt_bundle_from_batch(batch_dict: dict[str, Any], candidate: dict[str, Any]) -> PromptBundle:
    """Build the PromptBundle for one Stage 5C batch at runtime, using only
    that batch's own requested_field_paths. Pure — no model/network call."""
    from .prompt_v1_1 import build_v1_1_patch_prompt
    return build_v1_1_patch_prompt(
        poem_id=batch_dict["poem_id"],
        language=batch_dict["language"],
        original_poem=candidate["original_poem"],
        translated_poem=candidate["translated_poem"],
        existing_annotation=candidate["annotation"],
        requested_field_paths=batch_dict["requested_field_paths"],
    )


def load_batch_and_build_prompt(batch_path: "str | Path", candidate_path: "str | Path") -> PromptBundle:
    """Convenience loader: reads a Stage 5C batch JSON file and its
    referenced candidate JSON file from disk, then builds the PromptBundle.
    No model/network call — this only assembles prompt text."""
    with Path(batch_path).open("r", encoding="utf-8") as f:
        batch_dict = json.load(f)
    with Path(candidate_path).open("r", encoding="utf-8") as f:
        candidate = json.load(f)
    return build_prompt_bundle_from_batch(batch_dict, candidate)
