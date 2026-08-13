"""Stage 5M — dedicated, poem-generic Vertex corpus runner.

`vertex_canary_execution_v1_1.py` implements the full five-section
generate -> assemble -> validate -> targeted-repair -> stop-gate pipeline,
but it is *intentionally* restricted to a two-poem canary allowlist
(`require_authorized_poem_id`, a hardcoded `_POEM_SOURCE_PATH` dict, and a
canary-only output directory). This module is the poem-generic counterpart
the team needs to run the rest of the corpus: it imports and reuses every
poem-agnostic piece of that pipeline directly --
`assemble_candidate_from_sections`, `run_validation_pipeline`,
`derive_invalid_paths`, `apply_repair_response`, `section_structurally_valid`,
and `build_provenance` are used unmodified -- and re-implements only the
parts that were deliberately poem-restricted there (source loading, the
per-section/per-repair call wrapper, and the output/checkpoint layout),
generalized to any corpus poem.

Never imports `google.genai` directly. Every provider call goes through
`gemini_backfill_executor_v1_1`'s client factory and retry wrapper, exactly
like `vertex_canary_execution_v1_1.py`. A real client is only ever
constructed on the `--execute` path; `--dry-run` never calls
`client_factory()` and never calls `gx.generate_with_retry`.

Every candidate this module writes carries `candidate_status =
"MODEL_CANDIDATE"`, `review_status = "REVIEW_PENDING"`,
`native_review_required = true`, `not_silver = true`, `not_gold = true`,
`not_human_approved = true`. This module has no code path that can produce
`SILVER`, `GOLD`, `FINAL`, or `APPROVED`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import gemini_backfill_executor_v1_1 as gx
from .annotation_language_profile_v1_1 import load_annotation_language_profiles, get_annotation_profile_for_language
from .dataset import preprocess_poem
from .prompt_assembler_v1_1 import (
    assemble_prompt, StanzaSpec, PromptAssemblyError,
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
from .shared_full_schema_prompt_v1_1 import SHARED_PROMPT_CONTRACT_VERSION, COMPLETENESS_CONTRACT_VERSION
from .targeted_repair_prompt_v1_1 import build_targeted_repair_prompt, InvalidPathIssue
from .vertex_response_schema_v1_1 import build_and_audit_vertex_response_schema, build_and_audit_vertex_repair_response_schema
from . import vertex_canary_execution_v1_1 as vce  # poem-agnostic building blocks only; never call its poem-restricted functions

REQUIRED_SECTIONS: "tuple[str, ...]" = (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
)
GENERATIVE_SECTIONS: "tuple[str, ...]" = (
    SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
    SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS,
)
MAX_REPAIR_ROUNDS = 2
MAX_CONCURRENCY = 8

LIFECYCLE_STATUS = "MODEL_CANDIDATE"
REVIEW_STATE = "REVIEW_PENDING"
NOT_SILVER = True
NOT_GOLD = True
NOT_HUMAN_APPROVED = True
NATIVE_REVIEW_REQUIRED = True
FORBIDDEN_LIFECYCLE_STATUSES = frozenset({"SILVER", "GOLD", "FINAL", "APPROVED", "HUMAN_REVIEWED"})

# The six poems already generated under the pilot (see CLAUDE.md /
# docs/IMPLEMENTATION_PLAN_V1_1.md Stage 5K). Excluded from default
# assignments/execution; regenerating one requires an explicit,
# separately-given override, never a default action.
PILOT_POEM_IDS: "frozenset[str]" = frozenset({
    "MV++_0011", "MV++_0073", "MV++_1118", "MV++_1153", "MV++_1249", "MV++_1443",
})

SOURCE_REQUIRED_KEYS = ("poem_id", "poem_title", "language", "original_poem", "translated_poem")

DEFAULT_SOURCE_CORPUS_DIR = Path("data") / "source_corpus"
DEFAULT_PROFILE_DIR = Path("morphoverse_gemini_pipeline") / "delivery" / "poem_annotator" / "annotation_language_profiles"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "model_candidates"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_LOCAL_PROVIDER_RUN_DIR = Path("local_provider_runs")
DEFAULT_RELEASE_MANIFEST_PATH = Path("corpus") / "execution_release_manifest.json"

FAILURE_CLASSES = (
    "PROVIDER_FAILURE", "TRUNCATION", "SCHEMA_FAILURE", "COMPLETENESS_FAILURE",
    "GROUNDING_FAILURE", "ROMANIZATION_FAILURE", "SOURCE_DATA_FAILURE",
    "PROFILE_FAILURE", "REPAIR_EXHAUSTED", "SECURITY_FAILURE", "UNKNOWN",
)


class CorpusRunnerError(RuntimeError):
    """Base class for runner-level (not provider-level) failures."""


class PilotPoemBlocked(CorpusRunnerError):
    pass


class LanguageProfileMissing(CorpusRunnerError):
    pass


class LanguageBlocked(CorpusRunnerError):
    """Raised whenever corpus/execution_release_manifest.json marks the
    poem's language, or the poem_id itself, as not authorized for team
    generation. There is no override flag for this in this release — see
    SANSKRIT_BLOCK.md. If the manifest file is missing entirely, every
    language is treated as blocked (fail closed, never fail open)."""
    pass


class SourceDataError(CorpusRunnerError):
    pass


class AssignmentError(CorpusRunnerError):
    pass


# ══════════════════════════════════════════════════════════════════════════
# Source loading — source-only, whitelist-enforced (never reads an
# "annotation" key; the exported files don't have one, but this is checked
# defensively regardless of what's on disk).
# ══════════════════════════════════════════════════════════════════════════
def load_source_poem(poem_id: str, language: str, *, repo_root: Path,
                      source_corpus_dir: Path = DEFAULT_SOURCE_CORPUS_DIR) -> dict:
    path = repo_root / source_corpus_dir / language / f"{poem_id}.json"
    if not path.exists():
        raise SourceDataError(f"No source record for {poem_id!r} in language {language!r} at {path}.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SourceDataError(f"{path}: source record must be a JSON object.")
    extra_keys = set(raw.keys()) - set(SOURCE_REQUIRED_KEYS)
    if extra_keys:
        raise SourceDataError(
            f"{path}: source-only record must contain only {SOURCE_REQUIRED_KEYS}; "
            f"found forbidden extra key(s) {sorted(extra_keys)} (possible old-annotation leakage)."
        )
    missing_keys = [k for k in SOURCE_REQUIRED_KEYS if k not in raw]
    if missing_keys:
        raise SourceDataError(f"{path}: missing required key(s) {missing_keys}.")
    if raw["poem_id"] != poem_id or raw["language"] != language:
        raise SourceDataError(
            f"{path}: poem_id/language mismatch — file says {raw['poem_id']!r}/{raw['language']!r}, "
            f"caller asked for {poem_id!r}/{language!r}."
        )
    original_sha = hashlib.sha256(raw["original_poem"].encode("utf-8")).hexdigest()
    translation_sha = hashlib.sha256(raw["translated_poem"].encode("utf-8")).hexdigest()
    return {
        **{k: raw[k] for k in SOURCE_REQUIRED_KEYS},
        "_source_path": str(path.relative_to(repo_root)).replace("\\", "/"),
        "_source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "_original_poem_sha256": original_sha,
        "_translated_poem_sha256": translation_sha,
    }


def stanza_specs_for(source: dict) -> "tuple[StanzaSpec, ...]":
    poem = preprocess_poem({k: v for k, v in source.items() if not k.startswith("_")})
    return tuple(StanzaSpec(s.stanza_index, tuple(s.source_lines), tuple(s.translated_lines)) for s in poem.stanzas)


def require_language_profile(language: str, profile_dir: Path):
    profiles = load_annotation_language_profiles(profile_dir)
    profile = get_annotation_profile_for_language(profiles, language)
    if profile is None:
        raise LanguageProfileMissing(
            f"No reusable language addendum for {language!r} in {profile_dir}. "
            "This runner never invents or auto-translates a missing profile, and never "
            "falls back to a generic one — see BLOCKED_LANGUAGES.md."
        )
    return profile


def require_release_authorized(poem_id: str, language: str, release_manifest: "dict | None") -> None:
    """Refuses execution for any language/poem the current
    corpus/execution_release_manifest.json does not explicitly mark
    AUTHORIZED_FOR_TEAM_GENERATION, and for any poem_id listed under
    blocked_poems regardless of its language's status. Applies identically
    on --assignment, --resume, --language, and --poem-id paths, since every
    one of them funnels through plan_poem_dry_run / execute_poem_live."""
    if release_manifest is None:
        raise LanguageBlocked(
            "corpus/execution_release_manifest.json not found — refusing to run any "
            "language without an explicit release-authorization record on file."
        )
    lang_entry = release_manifest.get("languages", {}).get(language)
    lang_status = lang_entry.get("status") if isinstance(lang_entry, dict) else None
    if lang_status != "AUTHORIZED_FOR_TEAM_GENERATION":
        raise LanguageBlocked(
            f"{language!r} is not authorized for team generation "
            f"(execution_release_manifest.json status={lang_status!r}). See SANSKRIT_BLOCK.md / "
            "BLOCKED_LANGUAGES.md. There is no override flag for this in this release."
        )
    blocked_poem_status = release_manifest.get("blocked_poems", {}).get(poem_id)
    if blocked_poem_status is not None:
        raise LanguageBlocked(
            f"{poem_id!r} is explicitly blocked (status={blocked_poem_status!r}) regardless of "
            f"its language's authorization. See SANSKRIT_BLOCK.md."
        )


def require_not_pilot(poem_id: str, *, allow_pilot_regeneration: bool) -> None:
    if poem_id in PILOT_POEM_IDS and not allow_pilot_regeneration:
        raise PilotPoemBlocked(
            f"{poem_id!r} is one of the six already-generated pilot poems "
            f"({sorted(PILOT_POEM_IDS)}). Pass --allow-pilot-regeneration to override "
            "explicitly and intentionally; this is never a default action."
        )


# ══════════════════════════════════════════════════════════════════════════
# Assignment CSV
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class AssignmentRow:
    poem_id: str
    language: str
    assignee: str


def load_assignment_csv(path: Path) -> "list[AssignmentRow]":
    rows: "list[AssignmentRow]" = []
    seen: "dict[str, int]" = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"poem_id", "language", "assignee"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise AssignmentError(f"{path}: header must contain columns {sorted(required)}; got {reader.fieldnames}.")
        for line_no, row in enumerate(reader, start=2):
            poem_id = (row.get("poem_id") or "").strip()
            language = (row.get("language") or "").strip()
            assignee = (row.get("assignee") or "").strip()
            if not poem_id or not language or not assignee:
                raise AssignmentError(f"{path}:{line_no}: poem_id/language/assignee must all be non-empty.")
            if poem_id in seen:
                raise AssignmentError(
                    f"{path}:{line_no}: duplicate poem_id {poem_id!r} (first seen on line {seen[poem_id]})."
                )
            seen[poem_id] = line_no
            rows.append(AssignmentRow(poem_id=poem_id, language=language, assignee=assignee))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# Checkpoints (one file per poem — minimizes merge conflicts, makes --resume
# a plain existence check)
# ══════════════════════════════════════════════════════════════════════════
def checkpoint_path(checkpoint_dir: Path, poem_id: str) -> Path:
    return checkpoint_dir / f"{poem_id}.json"


def read_checkpoint(checkpoint_dir: Path, poem_id: str) -> "dict | None":
    p = checkpoint_path(checkpoint_dir, poem_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_checkpoint(checkpoint_dir: Path, checkpoint: dict) -> Path:
    p = checkpoint_path(checkpoint_dir, checkpoint["poem_id"])
    gx.atomic_write_json(p, checkpoint)
    return p


# ══════════════════════════════════════════════════════════════════════════
# Failure records (sanitized — no raw provider payload, no credential)
# ══════════════════════════════════════════════════════════════════════════
def failure_path(reports_dir: Path, poem_id: str) -> Path:
    return reports_dir / "failures" / f"{poem_id}.json"


def write_failure(reports_dir: Path, poem_id: str, language: str, assignee: "str | None",
                   classification: str, message: str, *, now_fn: Callable[[], datetime]) -> Path:
    if classification not in FAILURE_CLASSES:
        classification = "UNKNOWN"
    record = {
        "poem_id": poem_id, "language": language, "assignee": assignee,
        "classification": classification, "message": message,
        "timestamp": now_fn().isoformat(),
    }
    p = failure_path(reports_dir, poem_id)
    gx.atomic_write_json(p, record)
    return p


def has_failure_record(reports_dir: Path, poem_id: str) -> bool:
    return failure_path(reports_dir, poem_id).exists()


# ══════════════════════════════════════════════════════════════════════════
# Section / repair execution — poem-generic. Structurally the same request
# construction vertex_canary_execution_v1_1.execute_one_section_call /
# run_targeted_repair_round perform, minus require_authorized_poem_id and
# minus the two-poem-only output directory. Artifacts go to a gitignored
# local_provider_runs/ tree, never committed (see TEAM_RUNBOOK.md).
# ══════════════════════════════════════════════════════════════════════════
def _attempt_dir(local_run_dir: Path, poem_id: str, section: str, attempt_number: int) -> Path:
    return local_run_dir / poem_id / section / f"attempt_{attempt_number:02d}"


def run_section_call(*, client: Any, poem_id: str, section: str, source: dict,
                      stanzas: "tuple[StanzaSpec, ...]", profile_dir: Path, model: str,
                      local_run_dir: Path, attempt_number: int,
                      existing_candidate: "dict | None" = None,
                      sleep_fn: "Callable[[float], None] | None" = None) -> "tuple[vce.SectionCallRecord, dict | None]":
    result = assemble_prompt(
        annotation_language_profile_dir=profile_dir,
        poem_id=source["poem_id"], title=source["poem_title"], language=source["language"],
        original_poem=source["original_poem"], translated_poem=source["translated_poem"],
        stanzas=stanzas, section=section, existing_candidate=existing_candidate,
    )
    vertex_schema = build_and_audit_vertex_response_schema(section)

    a_dir = _attempt_dir(local_run_dir, poem_id, section, attempt_number)
    gx.atomic_write_json(a_dir / "prompt_request.json", {
        "system_instruction": result.system_instruction, "user_content": result.user_content,
        "prompt_hash": result.prompt_hash,
    })

    request = {
        "model": model, "contents": result.user_content,
        "config": gx.build_generation_config(result.system_instruction, vertex_schema, max_output_tokens=8192),
    }
    kwargs = {} if sleep_fn is None else {"sleep_fn": sleep_fn}
    outcome = gx.generate_with_retry(client, request, **kwargs)
    gemini_config = gx.load_gemini_config()

    meta = {
        "provider": gx.PROVIDER_NAME, "model": gemini_config.model, "project": gemini_config.project,
        "region": gemini_config.location, "api_version": gx.API_VERSION, "poem_id": poem_id,
        "section": section, "attempt_dir": a_dir.name,
        "generation_settings": gx.generation_settings_summary(),
        "attempt_count": len(outcome.attempts), "attempts": [asdict(a) for a in outcome.attempts],
        "prompt_sha256": result.prompt_hash,
    }

    if not outcome.success:
        meta["response_status"] = "provider_call_failed"
        gx.atomic_write_json(a_dir / "request_metadata.json", meta)
        return vce.SectionCallRecord(poem_id, section, attempt_number, "provider_call_failed", result.prompt_hash, None, None, None), None

    raw_text = getattr(outcome.response, "text", None)
    gx.atomic_write_text(a_dir / "raw_response.txt", raw_text or "")
    response_hash = hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()
    completion_meta = gx.extract_completion_metadata(outcome.response)
    meta["response_status"] = "success"
    meta["response_sha256"] = response_hash
    meta["completion"] = completion_meta
    gx.atomic_write_json(a_dir / "request_metadata.json", meta)

    parse_result = gx.parse_raw_response(raw_text)
    if parse_result.status != "success":
        return vce.SectionCallRecord(
            poem_id, section, attempt_number, "parse_failed", result.prompt_hash, response_hash,
            completion_meta["finish_reason"], completion_meta["output_token_count"],
        ), None

    gx.atomic_write_json(a_dir / "parsed_response.json", parse_result.parsed)
    return vce.SectionCallRecord(
        poem_id, section, attempt_number, "success", result.prompt_hash, response_hash,
        completion_meta["finish_reason"], completion_meta["output_token_count"],
    ), parse_result.parsed


def run_repair_round(*, client: Any, poem_id: str, source: dict, stanzas: "tuple[StanzaSpec, ...]",
                      profile_dir: Path, model: str, local_run_dir: Path, attempt_number: int,
                      annotation: dict, invalid_paths: "list[InvalidPathIssue]",
                      sleep_fn: "Callable[[float], None] | None" = None) -> "tuple[dict, list[str], vce.SectionCallRecord]":
    import re as _re
    minimal_context = {p.field_path: vce._get_at_path(annotation, p.field_path) for p in invalid_paths}
    stanza_indices = sorted({int(m.group(1)) for p in invalid_paths for m in [_re.search(r"stanzas\[(\d+)\]", p.field_path)] if m})
    relevant_stanzas = tuple(s for s in stanzas if (s.stanza_index - 1) in stanza_indices) or stanzas[:1]

    repair_result = build_targeted_repair_prompt(
        annotation_language_profile_dir=profile_dir,
        poem_id=source["poem_id"], language=source["language"],
        original_poem=source["original_poem"], translated_poem=source["translated_poem"],
        relevant_stanzas=relevant_stanzas, minimal_candidate_context=minimal_context,
        invalid_paths=tuple(invalid_paths),
    )
    vertex_schema = build_and_audit_vertex_repair_response_schema([p.field_path for p in invalid_paths])

    a_dir = _attempt_dir(local_run_dir, poem_id, "TARGETED_REPAIR", attempt_number)
    gx.atomic_write_json(a_dir / "prompt_request.json", {
        "system_instruction": repair_result.system_instruction, "user_content": repair_result.user_content,
        "prompt_hash": repair_result.prompt_hash,
    })

    request = {
        "model": model, "contents": repair_result.user_content,
        "config": gx.build_generation_config(repair_result.system_instruction, vertex_schema, max_output_tokens=4096),
    }
    kwargs = {} if sleep_fn is None else {"sleep_fn": sleep_fn}
    outcome = gx.generate_with_retry(client, request, **kwargs)
    gemini_config = gx.load_gemini_config()
    meta = {
        "provider": gx.PROVIDER_NAME, "model": gemini_config.model, "project": gemini_config.project,
        "region": gemini_config.location, "api_version": gx.API_VERSION, "poem_id": poem_id,
        "section": "TARGETED_REPAIR", "attempt_dir": a_dir.name,
        "generation_settings": gx.generation_settings_summary(),
        "attempt_count": len(outcome.attempts), "attempts": [asdict(a) for a in outcome.attempts],
        "prompt_sha256": repair_result.prompt_hash,
    }

    if not outcome.success:
        meta["response_status"] = "provider_call_failed"
        gx.atomic_write_json(a_dir / "request_metadata.json", meta)
        record = vce.SectionCallRecord(poem_id, "TARGETED_REPAIR", attempt_number, "provider_call_failed", repair_result.prompt_hash, None, None, None)
        return annotation, [p.field_path for p in invalid_paths], record

    raw_text = getattr(outcome.response, "text", None)
    gx.atomic_write_text(a_dir / "raw_response.txt", raw_text or "")
    response_hash = hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()
    completion_meta = gx.extract_completion_metadata(outcome.response)
    meta["response_status"] = "success"
    meta["response_sha256"] = response_hash
    meta["completion"] = completion_meta
    gx.atomic_write_json(a_dir / "request_metadata.json", meta)

    parse_result = gx.parse_raw_response(raw_text)
    record = vce.SectionCallRecord(
        poem_id, "TARGETED_REPAIR", attempt_number,
        "success" if parse_result.status == "success" else "parse_failed",
        repair_result.prompt_hash, response_hash, completion_meta["finish_reason"], completion_meta["output_token_count"],
    )
    if parse_result.status != "success":
        return annotation, [p.field_path for p in invalid_paths], record

    gx.atomic_write_json(a_dir / "parsed_response.json", parse_result.parsed)
    updated, still_unresolved = vce.apply_repair_response(annotation, parse_result.parsed, [p.field_path for p in invalid_paths])
    return updated, still_unresolved, record


# ══════════════════════════════════════════════════════════════════════════
# Dry-run planning — builds every prompt/schema a live run would build, and
# stops immediately before any client is constructed or any provider call is
# made. Zero provider calls, by construction (client_factory is never
# invoked on this path).
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DryRunSectionPlan:
    section: str
    prompt_sha256: str
    system_instruction_chars: int
    user_content_chars: int


@dataclass(frozen=True)
class DryRunPoemPlan:
    poem_id: str
    language: str
    stanza_count: int
    planned_sections: "tuple[DryRunSectionPlan, ...]"
    provider_calls_made: int  # always 0


def plan_poem_dry_run(poem_id: str, language: str, *, repo_root: Path, profile_dir: Path,
                       source_corpus_dir: Path = DEFAULT_SOURCE_CORPUS_DIR,
                       release_manifest: "dict | None" = None) -> DryRunPoemPlan:
    """Builds the same 4 generative-section prompts a live run would build
    (the 5th, CONSISTENCY_REVIEW, needs a real assembled candidate as input
    and is therefore not plannable offline) and returns their hashes/sizes.
    Never constructs a provider client."""
    require_release_authorized(poem_id, language, release_manifest)  # raises LanguageBlocked, no override
    require_language_profile(language, profile_dir)  # raises LanguageProfileMissing, never invents one
    source = load_source_poem(poem_id, language, repo_root=repo_root, source_corpus_dir=source_corpus_dir)
    stanzas = stanza_specs_for(source)
    plans = []
    for section in GENERATIVE_SECTIONS:
        result = assemble_prompt(
            annotation_language_profile_dir=profile_dir,
            poem_id=source["poem_id"], title=source["poem_title"], language=source["language"],
            original_poem=source["original_poem"], translated_poem=source["translated_poem"],
            stanzas=stanzas, section=section, existing_candidate=None,
        )
        build_and_audit_vertex_response_schema(section)  # constructed + audited, never sent
        plans.append(DryRunSectionPlan(
            section=section, prompt_sha256=result.prompt_hash,
            system_instruction_chars=len(result.system_instruction), user_content_chars=len(result.user_content),
        ))
    return DryRunPoemPlan(
        poem_id=poem_id, language=language, stanza_count=len(stanzas),
        planned_sections=tuple(plans), provider_calls_made=0,
    )


# ══════════════════════════════════════════════════════════════════════════
# Live per-poem execution — the poem-generic counterpart of
# vertex_canary_execution_v1_1.execute_poem_candidate.
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PoemExecutionResult:
    poem_id: str
    language: str
    stop_gate_passed: bool
    candidate_status: str
    candidate_path: "str | None"
    checkpoint_path: "str | None"
    unresolved_paths: "tuple[str, ...]"
    repair_rounds_used: int
    failure: "dict | None"


def _classify_failure(record: "vce.SectionCallRecord") -> str:
    if record.outcome == "provider_call_failed":
        return "PROVIDER_FAILURE"
    if record.outcome == "parse_failed":
        return "TRUNCATION" if record.finish_reason == "MAX_TOKENS" else "SCHEMA_FAILURE"
    return "UNKNOWN"


def execute_poem_live(poem_id: str, language: str, *, repo_root: Path, profile_dir: Path,
                       client_factory: Callable[[], Any], output_root: Path, checkpoint_dir: Path,
                       reports_dir: Path, local_run_dir: Path, assignee: "str | None" = None,
                       source_corpus_dir: Path = DEFAULT_SOURCE_CORPUS_DIR,
                       sleep_fn: "Callable[[float], None] | None" = None,
                       now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                       release_manifest: "dict | None" = None) -> PoemExecutionResult:
    require_release_authorized(poem_id, language, release_manifest)  # raises LanguageBlocked, no override
    profile = require_language_profile(language, profile_dir)
    source = load_source_poem(poem_id, language, repo_root=repo_root, source_corpus_dir=source_corpus_dir)
    stanzas = stanza_specs_for(source)
    gemini_config = gx.load_gemini_config()
    client = client_factory()

    section_records: "list[vce.SectionCallRecord]" = []
    parsed_by_section: "dict[str, dict]" = {}
    for section in GENERATIVE_SECTIONS:
        record, parsed = run_section_call(
            client=client, poem_id=poem_id, section=section, source=source, stanzas=stanzas,
            profile_dir=profile_dir, model=gemini_config.model, local_run_dir=local_run_dir,
            attempt_number=1, sleep_fn=sleep_fn,
        )
        section_records.append(record)
        if record.outcome != "success":
            classification = _classify_failure(record)
            write_failure(reports_dir, poem_id, language, assignee, classification,
                          f"{section}: provider call did not succeed ({record.outcome}).", now_fn=now_fn)
            return PoemExecutionResult(poem_id, language, False, "GENERATION_FAILED", None, None, (), 0,
                                        {"classification": classification, "section": section})
        ok, reason = vce.section_structurally_valid(section, parsed)
        if not ok:
            write_failure(reports_dir, poem_id, language, assignee, "SCHEMA_FAILURE",
                          f"{section}: {reason}", now_fn=now_fn)
            return PoemExecutionResult(poem_id, language, False, "GENERATION_FAILED", None, None, (), 0,
                                        {"classification": "SCHEMA_FAILURE", "section": section, "reason": reason})
        parsed_by_section[section] = parsed

    stanza_count = len(stanzas)
    candidate = vce.assemble_candidate_from_sections(
        parsed_by_section[SECTION_POEM_AND_STANZA_OVERVIEW], parsed_by_section[SECTION_CULTURAL_ENTITIES],
        parsed_by_section[SECTION_FIGURATIVE_EXPRESSIONS], parsed_by_section[SECTION_TRANSLATION_LOSS], stanza_count,
    )

    consistency_record, consistency_parsed = run_section_call(
        client=client, poem_id=poem_id, section=SECTION_CONSISTENCY_REVIEW, source=source, stanzas=stanzas,
        profile_dir=profile_dir, model=gemini_config.model, local_run_dir=local_run_dir, attempt_number=1,
        existing_candidate=candidate, sleep_fn=sleep_fn,
    )
    section_records.append(consistency_record)
    consistency_findings = (consistency_parsed or {}).get("consistency_findings", []) if consistency_record.outcome == "success" else []

    preprocessed = preprocess_poem({k: v for k, v in source.items() if not k.startswith("_")})
    validation = vce.run_validation_pipeline(candidate, preprocessed, source["original_poem"], source["translated_poem"])

    repair_records: "list[vce.SectionCallRecord]" = []
    repair_round = 0
    while not validation.all_objective_checks_pass and repair_round < MAX_REPAIR_ROUNDS:
        repair_round += 1
        invalid_paths = vce.derive_invalid_paths(validation)
        if not invalid_paths:
            break
        candidate, still_unresolved, repair_record = run_repair_round(
            client=client, poem_id=poem_id, source=source, stanzas=stanzas, profile_dir=profile_dir,
            model=gemini_config.model, local_run_dir=local_run_dir, attempt_number=repair_round,
            annotation=candidate, invalid_paths=invalid_paths, sleep_fn=sleep_fn,
        )
        repair_records.append(repair_record)
        validation = vce.run_validation_pipeline(candidate, preprocessed, source["original_poem"], source["translated_poem"])

    unresolved_paths = tuple(p.field_path for p in vce.derive_invalid_paths(validation))

    stop_gate_passed = (
        validation.schema_valid and validation.candidate_complete and validation.grounding_valid
        and validation.romanization_consistent and not unresolved_paths
    )

    if not stop_gate_passed:
        classification = (
            "COMPLETENESS_FAILURE" if not validation.candidate_complete else
            "GROUNDING_FAILURE" if not validation.grounding_valid else
            "ROMANIZATION_FAILURE" if not validation.romanization_consistent else
            "REPAIR_EXHAUSTED" if unresolved_paths else
            "SCHEMA_FAILURE"
        )
        write_failure(reports_dir, poem_id, language, assignee, classification,
                      f"stop gate not satisfied after {repair_round} repair round(s); "
                      f"unresolved_paths={list(unresolved_paths)}", now_fn=now_fn)

    provenance = vce.build_provenance(
        gemini_config=gemini_config, source=source, section_records=section_records, repair_records=repair_records,
        profile_version=profile.profile_version,
        shared_prompt_contract_version=SHARED_PROMPT_CONTRACT_VERSION,
        completeness_contract_version=COMPLETENESS_CONTRACT_VERSION,
        schema_contract_version="5K.2.0", generation_timestamp=now_fn().isoformat(),
    )

    final_envelope = {
        "schema_version": "1.1",
        "poem_id": source["poem_id"], "poem_title": source["poem_title"], "language": source["language"],
        "original_poem": source["original_poem"], "translated_poem": source["translated_poem"],
        "candidate_status": LIFECYCLE_STATUS, "review_status": REVIEW_STATE,
        "not_silver": NOT_SILVER, "not_gold": NOT_GOLD, "not_human_approved": NOT_HUMAN_APPROVED,
        "native_review_required": NATIVE_REVIEW_REQUIRED,
        "candidate_complete": validation.candidate_complete,
        "stop_gate_passed": stop_gate_passed,
        "unresolved_paths": list(unresolved_paths),
        "human_review_items": [
            {"field_path": p, "reason": "unresolved after maximum targeted repair rounds — requires human review", "status": "REVIEW_PENDING"}
            for p in unresolved_paths
        ],
        "consistency_review_findings": consistency_findings,
        "annotation": validation.normalized_annotation if validation.normalized_annotation is not None else candidate,
        "source_provenance": {"path": source["_source_path"], "kind": "source_corpus_export", "file_sha256": source["_source_file_sha256"]},
        "vertex_provenance": provenance,
        "assignee": assignee,
    }

    out_path = output_root / language / f"{poem_id}_vertex_model_candidate.json"
    if out_path.exists():
        raise CorpusRunnerError(f"{poem_id}: refusing to overwrite an existing candidate at {out_path}.")
    gx.atomic_write_json(out_path, final_envelope)

    checkpoint = {
        "poem_id": poem_id, "assignee": assignee, "language": language,
        "source_hash": source["_original_poem_sha256"], "translation_hash": source["_translated_poem_sha256"],
        "candidate_path": str(out_path),
        "candidate_sha256": hashlib.sha256(json.dumps(final_envelope, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "completion_timestamp": now_fn().isoformat(),
        "model": gemini_config.model,
        "section_attempts": len(section_records),
        "repair_count": repair_round,
        "stop_gate_result": stop_gate_passed,
    }
    ckpt_path = write_checkpoint(checkpoint_dir, checkpoint)

    return PoemExecutionResult(
        poem_id=poem_id, language=language, stop_gate_passed=stop_gate_passed,
        candidate_status=LIFECYCLE_STATUS if stop_gate_passed else "MODEL_CANDIDATE_INCOMPLETE",
        candidate_path=str(out_path), checkpoint_path=str(ckpt_path),
        unresolved_paths=unresolved_paths, repair_rounds_used=repair_round, failure=None,
    )


# ══════════════════════════════════════════════════════════════════════════
# Resume / retry-failed-only / skip-existing-valid filters — pure functions
# over an already-resolved target list, so --resume etc. are unit-testable
# without running the CLI end to end.
# ══════════════════════════════════════════════════════════════════════════
def filter_resume(targets: "list[AssignmentRow]", *, checkpoint_dir: Path) -> "list[AssignmentRow]":
    return [t for t in targets if read_checkpoint(checkpoint_dir, t.poem_id) is None]


def filter_retry_failed_only(targets: "list[AssignmentRow]", *, reports_dir: Path, checkpoint_dir: Path) -> "list[AssignmentRow]":
    return [t for t in targets if has_failure_record(reports_dir, t.poem_id) and read_checkpoint(checkpoint_dir, t.poem_id) is None]


def filter_skip_existing_valid(targets: "list[AssignmentRow]", *, output_root: Path) -> "list[AssignmentRow]":
    return [t for t in targets if not (output_root / t.language / f"{t.poem_id}_vertex_model_candidate.json").exists()]


# ══════════════════════════════════════════════════════════════════════════
# Target resolution (assignment / language / poem-id filters)
# ══════════════════════════════════════════════════════════════════════════
def resolve_targets(*, repo_root: Path, assignment_csv: "Path | None", language: "str | None",
                     poem_id: "str | None", max_poems: "int | None",
                     allow_pilot_regeneration: bool,
                     source_manifest: "dict | None" = None) -> "list[AssignmentRow]":
    if assignment_csv is not None:
        rows = load_assignment_csv(assignment_csv)
    elif poem_id is not None:
        if language is None:
            raise AssignmentError("--poem-id requires --language.")
        rows = [AssignmentRow(poem_id=poem_id, language=language, assignee="unassigned")]
    elif language is not None:
        if source_manifest is None:
            raise AssignmentError("--language requires a loaded corpus/source_manifest.json (source_manifest not provided).")
        rows = [
            AssignmentRow(poem_id=r["poem_id"], language=r["language"], assignee="unassigned")
            for r in source_manifest["records"] if r["language"] == language
        ]
    else:
        # No filter given at all: the full corpus. Deliberately does NOT
        # pre-filter by profile_status — a blocked-language poem is still
        # listed as a target and will report BLOCKED (dry-run) or raise
        # LanguageProfileMissing (live), rather than being silently dropped.
        if source_manifest is None:
            raise AssignmentError(
                "No --assignment/--language/--poem-id given, and corpus/source_manifest.json "
                "was not found — cannot resolve a full-corpus target list."
            )
        rows = [
            AssignmentRow(poem_id=r["poem_id"], language=r["language"], assignee="unassigned")
            for r in source_manifest["records"]
        ]

    if not allow_pilot_regeneration:
        rows = [r for r in rows if r.poem_id not in PILOT_POEM_IDS]

    if max_poems is not None:
        rows = rows[:max_poems]

    return rows


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus_gemini_runner_v1_1",
        description="MorphoVerse++ Schema v1.1 Gemini (Vertex AI) corpus annotation runner.",
    )
    parser.add_argument("--assignment", type=Path, default=None, help="Path to a teammate assignment CSV (poem_id,language,assignee).")
    parser.add_argument("--language", type=str, default=None, help="Run every non-pilot, non-blocked corpus poem for one language.")
    parser.add_argument("--poem-id", type=str, default=None, help="Run exactly one poem (requires --language).")
    parser.add_argument("--max-poems", type=int, default=None, help="Cap the number of poems processed this invocation.")
    parser.add_argument("--dry-run", action="store_true", help="Build every prompt/schema; make zero provider calls.")
    parser.add_argument("--resume", action="store_true", help="Skip poems that already have a checkpoint.")
    parser.add_argument("--retry-failed-only", action="store_true", help="Process only poems with a recorded failure and no checkpoint.")
    parser.add_argument("--skip-existing-valid", action="store_true", help="Skip poems with an existing output file already on disk.")
    parser.add_argument("--concurrency", type=int, default=1, help=f"Parallel poems in flight (default 1, max {MAX_CONCURRENCY}).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for MODEL_CANDIDATE JSON output.")
    parser.add_argument("--allow-pilot-regeneration", action="store_true", help="Explicit override to (re)run one of the six pilot poems.")
    parser.add_argument("--execute", action="store_true", help="Required (with --acknowledge-billing) to make real, billable Vertex calls.")
    parser.add_argument("--acknowledge-billing", action="store_true", help="Required (with --execute) to make real, billable Vertex calls.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Repo root (defaults to this file's repo).")
    return parser


def _repo_root_from_args(args: argparse.Namespace) -> Path:
    if args.repo_root is not None:
        return args.repo_root
    return Path(__file__).resolve().parents[3]


def main(argv: "list[str] | None" = None, *, client_factory: "Callable[[], Any] | None" = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.concurrency < 1 or args.concurrency > MAX_CONCURRENCY:
        parser.error(f"--concurrency must be between 1 and {MAX_CONCURRENCY} (never an uncontrolled large batch).")

    if not args.dry_run and not (args.execute and args.acknowledge_billing):
        parser.error(
            "Refusing to run: pass --dry-run to plan offline, or pass both --execute and "
            "--acknowledge-billing together to make real, billable Vertex AI calls."
        )

    repo_root = _repo_root_from_args(args)
    profile_dir = repo_root / DEFAULT_PROFILE_DIR
    output_root = repo_root / args.output_root
    checkpoint_dir = repo_root / DEFAULT_CHECKPOINT_DIR
    reports_dir = repo_root / DEFAULT_REPORTS_DIR
    local_run_dir = repo_root / DEFAULT_LOCAL_PROVIDER_RUN_DIR

    source_manifest = None
    manifest_path = repo_root / "corpus" / "source_manifest.json"
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    release_manifest = None
    release_manifest_path = repo_root / DEFAULT_RELEASE_MANIFEST_PATH
    if release_manifest_path.exists():
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))

    try:
        targets = resolve_targets(
            repo_root=repo_root, assignment_csv=args.assignment, language=args.language,
            poem_id=args.poem_id, max_poems=args.max_poems,
            allow_pilot_regeneration=args.allow_pilot_regeneration, source_manifest=source_manifest,
        )
    except AssignmentError as exc:
        parser.error(str(exc))
        return 2

    if args.retry_failed_only:
        targets = filter_retry_failed_only(targets, reports_dir=reports_dir, checkpoint_dir=checkpoint_dir)
    if args.resume:
        targets = filter_resume(targets, checkpoint_dir=checkpoint_dir)
    if args.skip_existing_valid:
        targets = filter_skip_existing_valid(targets, output_root=output_root)

    if args.dry_run:
        for t in targets:
            try:
                plan = plan_poem_dry_run(t.poem_id, t.language, repo_root=repo_root, profile_dir=profile_dir,
                                          release_manifest=release_manifest)
                print(f"[DRY RUN] {plan.poem_id} ({plan.language}): {len(plan.planned_sections)} sections planned, "
                      f"{plan.stanza_count} stanzas, 0 provider calls.")
            except CorpusRunnerError as exc:
                print(f"[DRY RUN] {t.poem_id}: BLOCKED — {exc}")
        return 0

    factory = client_factory or gx.default_client_factory
    results: "list[PoemExecutionResult]" = []

    def _run_one(t: AssignmentRow) -> PoemExecutionResult:
        try:
            return execute_poem_live(
                t.poem_id, t.language, repo_root=repo_root, profile_dir=profile_dir,
                client_factory=factory, output_root=output_root, checkpoint_dir=checkpoint_dir,
                reports_dir=reports_dir, local_run_dir=local_run_dir, assignee=t.assignee,
                release_manifest=release_manifest,
            )
        except LanguageBlocked as exc:
            # A blocked poem must never terminate or corrupt an otherwise
            # successful batch (e.g. a full-corpus run that includes
            # Sanskrit) — report it as a non-fatal per-poem result, exactly
            # like the dry-run path does, instead of propagating.
            print(f"[BLOCKED] {t.poem_id} ({t.language}): {exc}")
            return PoemExecutionResult(
                poem_id=t.poem_id, language=t.language, stop_gate_passed=False,
                candidate_status="BLOCKED_ENGINEERING_REVIEW", candidate_path=None, checkpoint_path=None,
                unresolved_paths=(), repair_rounds_used=0,
                failure={"classification": "BLOCKED_BY_RELEASE_MANIFEST", "message": str(exc)},
            )

    if args.concurrency == 1:
        for t in targets:
            results.append(_run_one(t))
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(_run_one, t): t for t in targets}
            for fut in as_completed(futures):
                results.append(fut.result())

    succeeded = sum(1 for r in results if r.stop_gate_passed)
    print(f"Processed {len(results)} poem(s): {succeeded} passed the stop gate, {len(results) - succeeded} did not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
