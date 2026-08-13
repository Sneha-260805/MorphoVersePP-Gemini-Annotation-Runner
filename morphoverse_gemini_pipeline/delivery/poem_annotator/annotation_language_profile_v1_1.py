"""Stage 5K.1 (Task 4) — annotation-language-profile schema, loader, and
validator.

An AnnotationLanguageProfile records LANGUAGE-SPECIFIC ANNOTATION GUIDANCE
fed into the shared prompt as the "language addendum" (romanization
conventions, cultural-boundary rules, literary-context cautions,
stereotype-avoidance rules, ...). This is deliberately a DIFFERENT concept
from `language_profiles_v1_1.LanguageProfile`, which records operational
REVIEWER-ASSIGNMENT requirements (is a native speaker needed to review the
output). Both may exist for the same language and both may say
"native review required", but one is guidance fed to the model before
generation and the other is a requirement enforced on humans after
generation -- see
docs/SHARED_AND_LANGUAGE_SPECIFIC_PROMPT_ARCHITECTURE_V1_1.md for the full
distinction.

Pure, offline: loading and validating a profile never calls a model,
network, or provider, and never reads a poem or an existing annotation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

APPROVED = "APPROVED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
DISABLED = "DISABLED"
_VALID_EXAMPLE_STATUSES = frozenset({APPROVED, REVIEW_REQUIRED, DISABLED})

_PILOT_POEM_ID_RE = re.compile(r"MV\+\+_\d{4}")

# Heuristic phrases that indicate a profile is steering the model toward a
# single stereotyped visual rather than preserving ambiguity (Task 19
# rejection: "tells the model to infer stereotypical visuals"). Not
# exhaustive -- a documented heuristic scan, not a semantic understanding of
# the text.
_STEREOTYPE_INFERENCE_PHRASES = (
    "always depict", "always show", "always wearing", "must always show",
    "typical villager", "always visualize as", "default visual is",
    "should always be shown wearing", "always draw",
)

# Section headers owned exclusively by shared_full_schema_prompt_v1_1.py.
# A profile that reproduces one of these verbatim is duplicating the shared
# schema rather than supplying language-specific guidance (Task 19
# rejection: "duplicates the full shared schema").
_SHARED_SCHEMA_MARKER_PHRASES = (
    "COMPLETENESS RULES (mandatory)",
    "OUTPUT RESTRICTIONS (mandatory)",
    "EXPLICIT UNCERTAINTY HANDLING (mandatory)",
    "CONTROLLED VOCABULARIES (use exactly these values",
    "DATA SEPARATION AND PROMPT-INJECTION RULES (mandatory)",
    "CONTENT-QUALITY RULES (mandatory)",
)

_GUIDANCE_LIST_FIELDS = (
    "romanization_guidance", "linguistic_guidance", "cultural_boundary_rules",
    "literary_context_cautions", "translation_cautions",
    "figurative_language_cautions", "visual_representation_cautions",
    "stereotype_avoidance_rules", "ambiguity_handling_rules",
    "prohibited_assumptions",
)

_REQUIRED_KEYS = frozenset({
    "language", "language_code", "script", "profile_version",
    "romanization_scheme", "romanization_guidance", "linguistic_guidance",
    "cultural_boundary_rules", "literary_context_cautions",
    "translation_cautions", "figurative_language_cautions",
    "visual_representation_cautions", "stereotype_avoidance_rules",
    "ambiguity_handling_rules", "approved_examples", "prohibited_assumptions",
    "native_review_required", "profile_status", "source_provenance",
})


class AnnotationLanguageProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ApprovedExample:
    example_id: str
    status: str
    shows: str
    example_text: str
    source: str
    purpose: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovedExample":
        return cls(
            example_id=data["example_id"], status=data["status"], shows=data["shows"],
            example_text=data["example_text"], source=data["source"], purpose=data["purpose"],
        )


@dataclass(frozen=True)
class AnnotationLanguageProfile:
    language: str
    language_code: "str | None"
    script: str
    profile_version: str
    romanization_scheme: str
    romanization_guidance: "tuple[str, ...]"
    linguistic_guidance: "tuple[str, ...]"
    cultural_boundary_rules: "tuple[str, ...]"
    literary_context_cautions: "tuple[str, ...]"
    translation_cautions: "tuple[str, ...]"
    figurative_language_cautions: "tuple[str, ...]"
    visual_representation_cautions: "tuple[str, ...]"
    stereotype_avoidance_rules: "tuple[str, ...]"
    ambiguity_handling_rules: "tuple[str, ...]"
    approved_examples: "tuple[ApprovedExample, ...]"
    prohibited_assumptions: "tuple[str, ...]"
    native_review_required: bool
    profile_status: str
    source_provenance: "tuple[str, ...]"

    def active_examples(self) -> "tuple[ApprovedExample, ...]":
        """Only status==APPROVED examples may ever enter an assembled
        Vertex prompt (Task 12)."""
        return tuple(e for e in self.approved_examples if e.status == APPROVED)

    def all_guidance_text(self) -> "tuple[str, ...]":
        """Every free-text guidance string in this profile, flattened, for
        quality/audit scanning. Deliberately excludes structural metadata
        (language, script, profile_version, ...) which are not guidance."""
        texts: list[str] = []
        for field_name in _GUIDANCE_LIST_FIELDS:
            texts.extend(getattr(self, field_name))
        for ex in self.approved_examples:
            texts.append(ex.example_text)
            texts.append(ex.shows)
            texts.append(ex.purpose)
        return tuple(texts)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["romanization_guidance"] = list(self.romanization_guidance)
        d["linguistic_guidance"] = list(self.linguistic_guidance)
        d["cultural_boundary_rules"] = list(self.cultural_boundary_rules)
        d["literary_context_cautions"] = list(self.literary_context_cautions)
        d["translation_cautions"] = list(self.translation_cautions)
        d["figurative_language_cautions"] = list(self.figurative_language_cautions)
        d["visual_representation_cautions"] = list(self.visual_representation_cautions)
        d["stereotype_avoidance_rules"] = list(self.stereotype_avoidance_rules)
        d["ambiguity_handling_rules"] = list(self.ambiguity_handling_rules)
        d["approved_examples"] = [e.to_dict() for e in self.approved_examples]
        d["prohibited_assumptions"] = list(self.prohibited_assumptions)
        d["source_provenance"] = list(self.source_provenance)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "AnnotationLanguageProfile":
        d = dict(data)
        for name in _GUIDANCE_LIST_FIELDS:
            d[name] = tuple(d.get(name, ()))
        d["approved_examples"] = tuple(
            ApprovedExample.from_dict(e) for e in d.get("approved_examples", ())
        )
        d["source_provenance"] = tuple(d.get("source_provenance", ()))
        return cls(**d)


def validate_profile_structure(data: dict, source_label: str) -> None:
    """Structural validation only: required keys present, correct shapes.
    Raises AnnotationLanguageProfileError on any defect. Called before
    quality validation and before prompt assembly (Task 4/13)."""
    if not isinstance(data, dict):
        raise AnnotationLanguageProfileError(f"{source_label}: profile must be a JSON object.")
    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        raise AnnotationLanguageProfileError(f"{source_label}: missing required keys {sorted(missing)}")

    if not isinstance(data["language"], str) or not data["language"].strip():
        raise AnnotationLanguageProfileError(f"{source_label}: 'language' must be a non-empty string.")
    if data["language_code"] is not None and not isinstance(data["language_code"], str):
        raise AnnotationLanguageProfileError(f"{source_label}: 'language_code' must be a string or null.")
    if not isinstance(data["script"], str) or not data["script"].strip():
        raise AnnotationLanguageProfileError(f"{source_label}: 'script' must be a non-empty string.")
    if not isinstance(data["profile_version"], str) or not data["profile_version"].strip():
        raise AnnotationLanguageProfileError(f"{source_label}: 'profile_version' must be a non-empty string.")
    if not isinstance(data["romanization_scheme"], str) or not data["romanization_scheme"].strip():
        raise AnnotationLanguageProfileError(f"{source_label}: 'romanization_scheme' must be a non-empty string.")

    for name in _GUIDANCE_LIST_FIELDS:
        value = data[name]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise AnnotationLanguageProfileError(f"{source_label}: '{name}' must be a list of strings.")

    examples = data["approved_examples"]
    if not isinstance(examples, list):
        raise AnnotationLanguageProfileError(f"{source_label}: 'approved_examples' must be a list.")
    example_required = {"example_id", "status", "shows", "example_text", "source", "purpose"}
    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            raise AnnotationLanguageProfileError(f"{source_label}: approved_examples[{i}] must be an object.")
        ex_missing = example_required - set(ex.keys())
        if ex_missing:
            raise AnnotationLanguageProfileError(f"{source_label}: approved_examples[{i}] missing keys {sorted(ex_missing)}")
        if ex["status"] not in _VALID_EXAMPLE_STATUSES:
            raise AnnotationLanguageProfileError(
                f"{source_label}: approved_examples[{i}].status must be one of "
                f"{sorted(_VALID_EXAMPLE_STATUSES)}, got {ex['status']!r}"
            )

    if not isinstance(data["native_review_required"], bool):
        raise AnnotationLanguageProfileError(f"{source_label}: 'native_review_required' must be a boolean.")
    if not isinstance(data["profile_status"], str) or not data["profile_status"].strip():
        raise AnnotationLanguageProfileError(f"{source_label}: 'profile_status' must be a non-empty string.")
    if not isinstance(data["source_provenance"], list) or not all(isinstance(v, str) for v in data["source_provenance"]):
        raise AnnotationLanguageProfileError(f"{source_label}: 'source_provenance' must be a list of strings.")


def assert_profile_quality_acceptable(profile: AnnotationLanguageProfile, source_label: str) -> None:
    """Task 19 rejection criteria. Raises AnnotationLanguageProfileError
    with the specific reason on the first violation found; does not mutate
    or repair the profile."""
    all_text = profile.all_guidance_text()
    non_empty_text = [t for t in all_text if t.strip()]

    if not non_empty_text:
        raise AnnotationLanguageProfileError(f"{source_label}: profile is empty -- no guidance text in any field.")

    joined = " ".join(non_empty_text).strip()
    # "Contains only the language name" -- a degenerate profile whose entire
    # guidance content is just the language/script name repeated.
    trivial_tokens = {profile.language.lower(), profile.script.lower(), "language", "script"}
    words = {w.strip(".,;:'\"()").lower() for w in joined.split()}
    if words and words <= trivial_tokens:
        raise AnnotationLanguageProfileError(
            f"{source_label}: profile contains only the language/script name, no real guidance."
        )

    for text in non_empty_text:
        if _PILOT_POEM_ID_RE.search(text):
            raise AnnotationLanguageProfileError(
                f"{source_label}: profile contains a poem ID pattern (MV++_####) -- "
                f"poem-specific content is prohibited in a reusable language profile: {text!r}"
            )

    lowered_joined = joined.lower()
    for phrase in _STEREOTYPE_INFERENCE_PHRASES:
        if phrase in lowered_joined:
            raise AnnotationLanguageProfileError(
                f"{source_label}: profile tells the model to infer a stereotypical visual "
                f"(matched phrase {phrase!r})."
            )

    for marker in _SHARED_SCHEMA_MARKER_PHRASES:
        if marker in joined:
            raise AnnotationLanguageProfileError(
                f"{source_label}: profile duplicates shared-schema text owned by "
                f"shared_full_schema_prompt_v1_1.py (matched {marker!r})."
            )

    if not profile.romanization_guidance:
        raise AnnotationLanguageProfileError(f"{source_label}: profile lacks romanization guidance.")
    if not profile.cultural_boundary_rules:
        raise AnnotationLanguageProfileError(f"{source_label}: profile lacks cultural-boundary guidance.")
    # native_review_required's mere presence-as-bool is already enforced by
    # validate_profile_structure(); "lacks native-review metadata" here
    # additionally requires profile_status to actually describe review state.
    if not profile.profile_status.strip():
        raise AnnotationLanguageProfileError(f"{source_label}: profile lacks native-review/profile_status metadata.")


# Findings from pilot/reports/prompt_architecture_stage5k1/legacy_prompt_audit.json
# classified REVIEW_REQUIRED (as opposed to REUSE_UNCHANGED/LANGUAGE_PROFILE_ONLY),
# used only to compute the "unresolved_policy_questions" quality-report count
# below from a profile's own source_provenance references -- never used to
# gate validation.
_REVIEW_REQUIRED_AUDIT_FINDING_IDS = frozenset({"L11", "L16"})


def profile_quality_report(profile: AnnotationLanguageProfile) -> dict:
    """Task 19's per-profile report. Never raises -- a pure read-only
    summary, independent of assert_profile_quality_acceptable()."""
    total_guidance_rules = sum(len(getattr(profile, name)) for name in _GUIDANCE_LIST_FIELDS)
    active = sum(1 for e in profile.approved_examples if e.status == APPROVED)
    review_required = sum(1 for e in profile.approved_examples if e.status == REVIEW_REQUIRED)
    unresolved_policy_questions = sum(
        1 for ref in profile.source_provenance
        if any(fid in ref for fid in _REVIEW_REQUIRED_AUDIT_FINDING_IDS)
    )
    return {
        "language": profile.language,
        "total_guidance_rules": total_guidance_rules,
        "romanization_rules": len(profile.romanization_guidance),
        "cultural_boundary_rules": len(profile.cultural_boundary_rules),
        "stereotype_avoidance_rules": len(profile.stereotype_avoidance_rules),
        "translation_cautions": len(profile.translation_cautions),
        "figurative_language_cautions": len(profile.figurative_language_cautions),
        "active_approved_examples": active,
        "review_required_examples": review_required,
        "prohibited_assumptions": len(profile.prohibited_assumptions),
        "native_review_required": profile.native_review_required,
        "legacy_sources_inspected": list(profile.source_provenance),
        "unresolved_policy_questions": unresolved_policy_questions,
    }


def load_annotation_language_profiles(profile_dir: "str | Path") -> "dict[str, AnnotationLanguageProfile]":
    """Load and fully validate (structure + quality) every *.json file in
    profile_dir, keyed by its own 'language' field. Raises
    AnnotationLanguageProfileError on the first malformed/rejected profile
    rather than silently skipping it -- a missing or bad profile must be
    loud, never silently absent (Task 13's assembler relies on this)."""
    profile_dir = Path(profile_dir)
    if not profile_dir.is_dir():
        raise AnnotationLanguageProfileError(f"Annotation-language-profile directory does not exist: {profile_dir}")

    profiles: "dict[str, AnnotationLanguageProfile]" = {}
    for path in sorted(profile_dir.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        validate_profile_structure(data, str(path))
        profile = AnnotationLanguageProfile.from_dict(data)
        assert_profile_quality_acceptable(profile, str(path))
        if profile.language in profiles:
            raise AnnotationLanguageProfileError(f"Duplicate annotation language profile for {profile.language!r} ({path}).")
        profiles[profile.language] = profile

    return profiles


def get_annotation_profile_for_language(
    profiles: "dict[str, AnnotationLanguageProfile]", language: "str | None",
) -> "AnnotationLanguageProfile | None":
    if language is None:
        return None
    return profiles.get(language)
