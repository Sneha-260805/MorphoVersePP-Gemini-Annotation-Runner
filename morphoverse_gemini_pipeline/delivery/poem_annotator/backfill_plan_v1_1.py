"""Stage 5B — five provider-neutral backfill request plans.

This module builds (but never sends) exactly one future patch-request plan
per pilot poem, derived entirely from the Stage 5A missing-field report and
`patch_v1_1.classify_missing_path`. It makes no model, network, or provider
call, and stores no full poem/translation text in a plan file — only
references to where that text already lives on disk.

See docs/PATCH_BACKFILL_STAGE5B.md for the full contract.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import patch_v1_1 as pv
from .prompt_v1_1 import build_v1_1_patch_prompt, PromptBundle, PROMPT_KIND_PATCH

PLANNING_STAGE = "5B_backfill_request_plan"

# Patch-size classification thresholds (Task 9). These are simple bucket
# boundaries over `requested_field_count`, not a cost/effort estimate.
_SIZE_SMALL_MAX = 20
_SIZE_MEDIUM_MAX = 79


def _patch_size_class(requested_field_count: int) -> str:
    if requested_field_count <= _SIZE_SMALL_MAX:
        return "small"
    if requested_field_count <= _SIZE_MEDIUM_MAX:
        return "medium"
    return "large"


@dataclass(frozen=True)
class BackfillPlan:
    poem_id: str
    language: str
    source_candidate_path: str
    source_legacy_annotation_path: str
    requested_field_paths: tuple[str, ...]
    human_review_only_paths: tuple[str, ...]
    intentionally_nullable_paths: tuple[str, ...]
    known_review_focus: tuple[str, ...]
    expected_prompt_kind: str
    candidate_tier: str
    calls_made_this_stage: int
    planned_future_provider_requests: int

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("requested_field_paths", "human_review_only_paths", "intentionally_nullable_paths", "known_review_focus"):
            d[key] = list(d[key])
        return d


def build_backfill_plan(
    poem_id: str,
    language: str,
    source_candidate_path: str,
    source_legacy_annotation_path: str,
    missing_paths: list[str],
    candidate_tier: str,
    known_review_focus: list[str],
) -> BackfillPlan:
    """Pure function: classify every Stage 5A missing-field path for one
    poem and produce exactly one plan with exactly one future provider
    request planned (Task 8). No file I/O, no model/network call."""
    requested = sorted({p for p in missing_paths if pv.classify_missing_path(p) == pv.CLASS_MODEL_BACKFILL_ALLOWED})
    human_review = sorted({p for p in missing_paths if pv.classify_missing_path(p) == pv.CLASS_HUMAN_REVIEW_REQUIRED})
    nullable = sorted({p for p in missing_paths if pv.classify_missing_path(p) == pv.CLASS_INTENTIONALLY_NULLABLE})

    return BackfillPlan(
        poem_id=poem_id,
        language=language,
        source_candidate_path=source_candidate_path,
        source_legacy_annotation_path=source_legacy_annotation_path,
        requested_field_paths=tuple(requested),
        human_review_only_paths=tuple(human_review),
        intentionally_nullable_paths=tuple(nullable),
        known_review_focus=tuple(known_review_focus),
        expected_prompt_kind=PROMPT_KIND_PATCH,
        candidate_tier=candidate_tier,
        calls_made_this_stage=0,
        planned_future_provider_requests=1,
    )


def build_all_plans(
    manifest: dict[str, Any],
    missing_fields_report: dict[str, Any],
    candidate_tiers: dict[str, str],
) -> tuple[BackfillPlan, ...]:
    """Build one plan per poem listed in the manifest, in manifest order.
    `candidate_tiers` maps poem_id -> the candidate file's own
    `candidate_tier` value (read by the caller; this function does no I/O)."""
    plans = []
    for entry in manifest["poems"]:
        poem_id = entry["poem_id"]
        plans.append(build_backfill_plan(
            poem_id=poem_id,
            language=entry["language"],
            source_candidate_path=f"pilot/annotations_v1_1/pre_backfill/{poem_id}.json",
            source_legacy_annotation_path=entry["source_annotation_path"],
            missing_paths=missing_fields_report["per_poem"].get(poem_id, []),
            candidate_tier=candidate_tiers.get(poem_id, "migrated_legacy_gemini_candidate"),
            known_review_focus=entry.get("known_review_focus", []),
        ))
    return tuple(plans)


# ══════════════════════════════════════════════════════════════════════════
# Task 9 — backfill-plan summary
# ══════════════════════════════════════════════════════════════════════════
def summarize_plan(plan: BackfillPlan) -> dict[str, Any]:
    field_type_breakdown: dict[str, int] = {}
    for path in plan.requested_field_paths:
        leaf = pv.leaf_field(path)
        field_type_breakdown[leaf] = field_type_breakdown.get(leaf, 0) + 1

    blockers: list[str] = []
    if plan.human_review_only_paths:
        blockers.append(f"{len(plan.human_review_only_paths)} field(s) require human review before backfill (ambiguous grounding or an unresolved schema decision).")
    if "Sindhi" in " ".join(plan.known_review_focus) or plan.language == "Sindhi":
        blockers.append("Sindhi-fluent reviewer confirmation required (pilot/PILOT_APPROVAL_CHECKLIST.md condition B) before any resulting patch is treated as reviewed.")
    blockers.append("Reviewer availability for this language is not yet confirmed (pilot/PILOT_APPROVAL_CHECKLIST.md condition A).")

    return {
        "poem_id": plan.poem_id,
        "language": plan.language,
        "requested_field_count": len(plan.requested_field_paths),
        "human_review_only_field_count": len(plan.human_review_only_paths),
        "intentionally_nullable_field_count": len(plan.intentionally_nullable_paths),
        "field_type_breakdown": field_type_breakdown,
        "planned_provider_calls": plan.planned_future_provider_requests,
        "estimated_patch_size_class": _patch_size_class(len(plan.requested_field_paths)),
        "blockers": blockers,
    }


def build_plan_summary(plans: tuple[BackfillPlan, ...]) -> dict[str, Any]:
    return {
        "planning_stage": PLANNING_STAGE,
        "poem_count": len(plans),
        "total_planned_provider_calls": sum(p.planned_future_provider_requests for p in plans),
        "poems": [summarize_plan(p) for p in plans],
    }


# ══════════════════════════════════════════════════════════════════════════
# Task 10 — prompt integration (runtime-only; never rendered into a plan file)
# ══════════════════════════════════════════════════════════════════════════
def build_prompt_bundle_from_plan(plan: BackfillPlan, candidate: dict[str, Any]) -> PromptBundle:
    """Build the eventual patch PromptBundle for `plan` at runtime, from an
    already-loaded candidate envelope. Pure — makes no model/network call.
    Raises ValueError (via build_v1_1_patch_prompt) if requested_field_paths
    is empty, exactly like any other caller of that function."""
    return build_v1_1_patch_prompt(
        poem_id=plan.poem_id,
        language=plan.language,
        original_poem=candidate["original_poem"],
        translated_poem=candidate["translated_poem"],
        existing_annotation=candidate["annotation"],
        requested_field_paths=plan.requested_field_paths,
    )


def load_plan_and_build_prompt(plan_path: str | Path, candidate_path: str | Path) -> PromptBundle:
    """Convenience loader: reads a plan JSON file and its referenced
    candidate JSON file from disk, then builds the PromptBundle. Still no
    model/network call — this only assembles prompt text."""
    with Path(plan_path).open("r", encoding="utf-8") as f:
        plan_dict = json.load(f)
    with Path(candidate_path).open("r", encoding="utf-8") as f:
        candidate = json.load(f)
    plan = BackfillPlan(
        poem_id=plan_dict["poem_id"],
        language=plan_dict["language"],
        source_candidate_path=plan_dict["source_candidate_path"],
        source_legacy_annotation_path=plan_dict["source_legacy_annotation_path"],
        requested_field_paths=tuple(plan_dict["requested_field_paths"]),
        human_review_only_paths=tuple(plan_dict["human_review_only_paths"]),
        intentionally_nullable_paths=tuple(plan_dict["intentionally_nullable_paths"]),
        known_review_focus=tuple(plan_dict["known_review_focus"]),
        expected_prompt_kind=plan_dict["expected_prompt_kind"],
        candidate_tier=plan_dict["candidate_tier"],
        calls_made_this_stage=plan_dict["calls_made_this_stage"],
        planned_future_provider_requests=plan_dict["planned_future_provider_requests"],
    )
    return build_prompt_bundle_from_plan(plan, candidate)
