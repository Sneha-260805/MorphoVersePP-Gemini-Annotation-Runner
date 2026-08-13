"""Stage 4 — opt-in schema-v1.1 prompt builders. Pure functions, no model calls.

This module is entirely separate from the current production prompt path
(`shared_prompt_builder.py`, `languages/*.py`) and from `main.py`'s
generation loop. Nothing here is imported by, or wired into, any of those —
building a prompt string is the full extent of what this module does; no
network request, no provider client, and no credential of any kind is
touched anywhere below. `config.py`'s active `SCHEMA_VERSION`/`PROMPT_VERSION`
(still 5/9, governing the live pipeline) are not read or referenced here.

Two builders:
  - build_v1_1_candidate_prompt(...)  — a full transitional v1.1 annotation
    request for a poem that has no v1.1 annotation yet.
  - build_v1_1_patch_prompt(...)      — a narrow, explicit-field-only patch
    request for backfilling specific missing fields on an EXISTING
    annotation (Stage 5's concern; only prompt construction happens here,
    never patch application).

Both:
  - Are pure functions — same inputs always produce the same PromptBundle,
    no I/O, no randomness, no side effects, no input mutation.
  - Use grounding.build_line_index() to present the original poem with
    stable L1, L2, ... references, and reuse schema.py's enums directly
    (never a hand-copied duplicate list) so the prompt can never drift out
    of sync with what Stage 2's validators actually accept.
  - Explicitly and repeatedly label their output as a CANDIDATE annotation,
    never gold — see docs/ANNOTATION_LIFECYCLE.md.

See docs/PILOT_AND_PROMPT_V1_1.md for the full design rationale.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .grounding import build_line_index
from .schema import (
    ALLOWED_RECITATION_STYLES,
    ALLOWED_EMOTIONS,
    ALLOWED_TONES,
    ALLOWED_TRANSLATION_QUALITIES,
    ALLOWED_ENTITY_CATEGORIES,
    ALLOWED_VISUAL_PRIORITIES,
    ALLOWED_EXPRESSION_TYPES,
    MORPHOVERSE_SCHEMA_VERSION,
)

PROMPT_KIND_CANDIDATE = "v1_1_candidate"
PROMPT_KIND_PATCH = "v1_1_patch"


@dataclass(frozen=True)
class PromptBundle:
    """A prompt-construction result. Deliberately minimal — no credentials,
    no endpoints, no provider-specific parameters (Task 9). A future
    provider-abstraction stage (Stage 6) decides how to actually send this;
    this module never does."""
    system_prompt: str
    user_prompt: str
    expected_schema_version: str
    prompt_kind: str
    requested_field_paths: tuple[str, ...] = field(default_factory=tuple)


# ── Shared rendering helpers ──────────────────────────────────────────────────
def _render_line_indexed_original(original_poem: str) -> str:
    """Render the original poem as canonical L<n>: <line text> entries, using
    grounding.build_line_index() so numbering and preserved raw line text
    exactly match what validate_cultural_grounding_v1_1/
    validate_figurative_grounding_v1_1 will later check spans/line_ref
    against. Never mutates `original_poem`."""
    index = build_line_index(original_poem)
    if not index.lines:
        return "[NO ORIGINAL TEXT AVAILABLE]"
    return "\n".join(f"L{ln.line_number}: {ln.text}" for ln in index.lines)


def _render_translation(translated_poem: str) -> str:
    """Render the translation as plain lines, deliberately WITHOUT L<n>
    prefixes — line_ref refers only to the original-language poem
    (docs/GROUNDING_AND_LINE_REFERENCES.md), and using a visually distinct
    rendering here reinforces that in the prompt itself."""
    index = build_line_index(translated_poem)
    if not index.lines:
        return "[NO TRANSLATION AVAILABLE]"
    return "\n".join(ln.text for ln in index.lines)


def _enum_list(values: tuple[str, ...]) -> str:
    return " | ".join(values)


# ── Stage 5E.4 Task 3/4 — per-path source_span_translation grounding
# contract (patch prompt only; generic over every poem/path, never
# hardcoding a specific poem, entity index, or word) ─────────────────────────
def _render_translation_with_line_numbers(translated_poem: str) -> str:
    """A translation rendering numbered ONLY for use inside the grounding
    contract block below, so the model can point at specific lines when
    locating an exact substring. Deliberately labeled T<n>, never L<n> —
    these numbers are NOT valid `line_ref` values (line_ref is
    original-poem-only; see _render_translation's own docstring)."""
    index = build_line_index(translated_poem)
    if not index.lines:
        return "[NO TRANSLATION AVAILABLE]"
    return "\n".join(f"T{ln.line_number}: {ln.text}" for ln in index.lines)


_CULTURAL_ENTITY_SPAN_PATH_RE = re.compile(r"^annotation\.cultural_entities\[(\d+)\]\.source_span_translation$")
_FIGURATIVE_SPAN_PATH_RE = re.compile(r"^annotation\.stanzas\[(\d+)\]\.metaphor_spans\[(\d+)\]\.source_span_translation$")


def _related_span_context(existing_annotation: dict, path: str) -> dict:
    """For one *.source_span_translation request path, read whatever
    already-known context helps ground the answer — the sibling
    term/source_term and, when already resolved, source_span_original.
    Read-only, tolerant of a missing/short/malformed existing_annotation
    (returns {} rather than raising); never mutates its input. Generic over
    any poem_id and any cultural_entities/metaphor_spans index — the index
    itself comes only from `path`, never a hardcoded value."""
    match = _CULTURAL_ENTITY_SPAN_PATH_RE.match(path)
    if match:
        index = int(match.group(1))
        entities = existing_annotation.get("cultural_entities") or []
        if index < len(entities):
            entity = entities[index]
            return {
                "evidence_label": entity.get("term"),
                "source_span_original": entity.get("source_span_original"),
            }
        return {}
    match = _FIGURATIVE_SPAN_PATH_RE.match(path)
    if match:
        stanza_index, span_index = int(match.group(1)), int(match.group(2))
        stanzas = existing_annotation.get("stanzas") or []
        if stanza_index < len(stanzas):
            spans = stanzas[stanza_index].get("metaphor_spans") or []
            if span_index < len(spans):
                span = spans[span_index]
                return {
                    "evidence_label": span.get("source_term"),
                    "source_span_original": span.get("source_span_original"),
                }
        return {}
    return {}


_SOURCE_SPAN_TRANSLATION_CONTRACT_HEADER = """\
SOURCE_SPAN_TRANSLATION GROUNDING CONTRACT (mandatory for every path listed
below — this is the single most commonly rejected field, so read carefully):
1. The value must be copied VERBATIM from the TRANSLATION text shown here.
2. It must be ONE exact, contiguous substring of that translation.
3. Do not paraphrase.
4. Do not substitute a synonym for any word in your answer.
5. Do not stem, normalize, or otherwise rewrite the wording.
6. Do not invent an English gloss that merely expresses the same meaning —
   copy the translation's actual words.
7. Preserve the translation's own spelling, punctuation, and casing exactly,
   for every character that is part of the substring you select.
8. Return only the SMALLEST exact translation substring that visibly
   supports the annotated cultural entity or figurative expression.
9. Before returning the JSON, internally verify each source_span_translation
   value with a direct substring search against the translation text below —
   if a direct substring search would not find it, do not return it (return
   null for that field instead).
10. A semantically related word or phrase that does NOT literally occur in
    the translation text is INVALID, even when it expresses the right
    meaning — this includes reusing a controlled value from a DIFFERENT
    field (e.g. a visual_priority word) as if it were a translation span."""


def _build_source_span_translation_block(
    existing_annotation: dict, translated_poem: str, requested_field_paths: "tuple[str, ...]",
) -> str:
    """The full per-path grounding contract for this batch's own requested
    *.source_span_translation paths (Task 3) — empty string when this batch
    requests none (Task 7 item 1: never add an unneeded block). Entirely
    driven by `requested_field_paths`/`existing_annotation`; no poem ID,
    entity index, or specific word is ever hardcoded here."""
    span_paths = tuple(p for p in requested_field_paths if p.endswith(".source_span_translation"))
    if not span_paths:
        return ""

    entries = []
    for path in span_paths:
        context = _related_span_context(existing_annotation, path)
        lines = [f"- JSON path: {path}"]
        if context.get("evidence_label"):
            lines.append(f"  Related term/phrase (for context): {context['evidence_label']!r}")
        if context.get("source_span_original"):
            lines.append(
                f"  Related source_span_original (original-language, for context only, "
                f"NOT what you copy from): {context['source_span_original']!r}"
            )
        entries.append("\n".join(lines))

    return (
        _SOURCE_SPAN_TRANSLATION_CONTRACT_HEADER
        + "\n\nTRANSLATION (numbered for reference inside this contract only; these "
        "T<n> labels are NOT valid line_ref values — line_ref uses only the "
        "original poem's L<n> numbers shown above):\n"
        + _render_translation_with_line_numbers(translated_poem)
        + "\n\nREQUESTED source_span_translation PATHS AND THEIR CONTEXT:\n"
        + "\n".join(entries)
    )


_FINAL_SELF_CHECK = """\
FINAL SELF-CHECK (perform silently before returning your answer — these are
instructions to you, not JSON output, and do not replace or weaken the
independent validation your response will still be checked against after
you return it):
- Every requested field path appears exactly once in your JSON.
- Every source_span_translation value is an exact substring of the
  TRANSLATION text shown above.
- No source_span_translation value is a paraphrase or synonym substitution.
- The JSON you return is complete and syntactically valid.
- Your response contains no Markdown code fences and no text outside the
  single JSON object."""


# ── Shared rule blocks (Task 8 rule 7: patch prompt reuses these verbatim) ───
_CANDIDATE_LABEL_RULES = """\
CANDIDATE LABELING (mandatory):
- Everything you return is an LLM CANDIDATE annotation. It is NEVER gold.
- Do not use the word "gold" to describe your own output.
- Only a human adjudicator can ever produce a gold annotation; your output
  is a candidate to be reviewed, not a final answer."""

_SPAN_AND_LINE_REF_RULES = """\
SPAN AND LINE-REFERENCE RULES (mandatory, exact matching only):
- line_ref MUST be either "L<n>" (a single line, e.g. "L3") or
  "L<n>-L<m>" (an inclusive range, e.g. "L2-L4"), using ONLY the L<n>
  line numbers shown in the ORIGINAL POEM block below. No other format
  ("line 3", "3", "stanza 2", etc.) is valid.
- source_span_original MUST be copied VERBATIM (exact characters, exact
  punctuation, exact diacritics, no case changes) from the ORIGINAL POEM
  text shown below. Do not paraphrase, romanize, gloss, or lightly reword it.
- source_span_translation, when you provide it, MUST be copied VERBATIM
  from the TRANSLATION text shown below — never inferred, guessed, or
  reconstructed from translation_note, gloss, romanization, or from
  source_span_original. If you cannot find an exact verbatim phrase in the
  translation that corresponds to the cue, return null for
  source_span_translation rather than guessing or paraphrasing.
- If you cannot identify an exact verbatim span for a field, return null
  for that field. Never invent, approximate, or "close enough" a span."""

_NO_HALLUCINATION_RULES = """\
GROUNDING AND ANTI-HALLUCINATION RULES (mandatory):
- Do not add a cultural_entities cue merely because the poem is written in
  an Indian language. Every cue must be explicitly or strongly implied by
  the poem's own text.
- Do not substitute a generic pan-Indian symbol, or a neighboring region's
  or religion's culture, for what the poem's own text actually names.
  If the poem names something specific, keep it specific.
- Do not mechanically classify every metaphor_spans item as
  expression_type "metaphor" merely because it is stored in the
  metaphor_spans array. Assess each figurative expression independently and
  choose the expression_type that actually fits (simile, idiom, proverb,
  personification, symbolism, metonymy, allusion, wordplay, or metaphor).
- vehicle and tenor must reflect the SPECIFIC image and meaning of this
  poem's own figurative language — never a generic filler pairing.
- Distinguish visible textual evidence (what the poem's words literally
  show) from your own abstract interpretation (what you infer it means).
  Do not present an inference as if it were directly stated.
- Avoid over-annotation: do not force a cultural_entities or metaphor_spans
  entry where the evidence is weak. Use null or an empty list when evidence
  is insufficient rather than filling every field."""


def _enum_reference_block() -> str:
    return f"""\
CONTROLLED VALUES (use exactly these; do not invent alternatives):
recitation_style: {_enum_list(ALLOWED_RECITATION_STYLES)}
emotion: {_enum_list(ALLOWED_EMOTIONS)}
tone: {_enum_list(ALLOWED_TONES)}
translation_quality: {_enum_list(ALLOWED_TRANSLATION_QUALITIES)}
category (cultural_entities): {_enum_list(ALLOWED_ENTITY_CATEGORIES)}
visual_priority: {_enum_list(ALLOWED_VISUAL_PRIORITIES)}
expression_type (figurative expressions): {_enum_list(ALLOWED_EXPRESSION_TYPES)}

NOT YET FINALIZED (do not invent a permanent enum for these; Stage 2/4 have
deliberately left them open — see docs/SCHEMA_V1_1_DECISIONS.md):
- cultural_specificity_level: null, or a short conservative descriptive
  string (e.g. "high", "regional", "widely recognized"). No fixed list exists.
- visualization_difficulty: null, or a short conservative descriptive
  string (e.g. "low", "requires abstract representation"). No fixed list exists.
- translation_status: null, or a short conservative descriptive string
  (e.g. "preserved", "adapted", "omitted"). This is NOT a finalized enum —
  do not claim precision beyond what the text supports."""


_FULL_JSON_SHAPE = """\
{
  "schema_version": "1.1",
  "recitation_style": "<one of the controlled recitation_style values>",
  "emotional_arc": "<short free text>",
  "theme": "<short free text or null>",
  "stanzas": [
    {
      "index": <1-based int, matching stanza order>,
      "emotion": "<controlled emotion value>",
      "tone": "<controlled tone value>",
      "translation_quality": "<controlled translation_quality value>",
      "loss_note": "<short text, or empty string when translation_quality is faithful>",
      "translation_loss": [
        {
          "what_was_lost": "<short text, required if this item is present>",
          "where": "<L<n> or L<n>-L<m>, or null>",
          "severity": "<short descriptive string, or null>"
        }
      ],
      "metaphor_spans": [
        {
          "source_term": "<verbatim source phrase, legacy field>",
          "abstract_meaning": "<short poem-specific meaning, legacy field>",
          "source_span_original": "<verbatim span from ORIGINAL POEM, or null>",
          "source_span_translation": "<verbatim span from TRANSLATION, or null>",
          "expression_type": "<controlled expression_type value, or null>",
          "literal_meaning": "<short text, or null>",
          "vehicle": "<short text, or null>",
          "tenor": "<short text, or null>",
          "metaphor_mapping": {
            "vehicle_concept": "<short text>",
            "tenor_concept": "<short text>",
            "transferred_attributes": ["<short text>", "..."]
          },
          "line_ref": "<L<n> or L<n>-L<m>, or null>",
          "literalization_risk": "<short descriptive string, or null>",
          "visualization_strategy": "<short text, or null>",
          "acceptable_visual_variants": ["<short scene description>", "..."],
          "visualization_difficulty": "<short descriptive string, or null>"
        }
      ]
    }
  ],
  "cultural_entities": [
    {
      "term": "<verbatim source term, legacy field>",
      "romanization": "<short romanization, or empty string>",
      "category": "<controlled category value>",
      "stanza_index": <1-based int>,
      "preserved": <true, false, or null>,
      "translation_note": "<short text, or empty string>",
      "gloss": "<short text, or null>",
      "line_ref": "<L<n> or L<n>-L<m>, or null>",
      "source_span_original": "<verbatim span from ORIGINAL POEM, or null>",
      "source_span_translation": "<verbatim span from TRANSLATION, or null>",
      "visual_features": ["<short visual feature>", "..."],
      "visual_priority": "<controlled visual_priority value, or null>",
      "acceptable_visual_variants": ["<short scene description>", "..."],
      "negative_confusions": ["<a plausible but INCORRECT rendering to avoid>", "..."],
      "translation_status": "<short descriptive string, or null>",
      "cultural_specificity_level": "<short descriptive string, or null>"
    }
  ]
}"""


def build_v1_1_candidate_prompt(
    poem_id: str,
    language: str,
    original_poem: str,
    translated_poem: str,
) -> PromptBundle:
    """Build a full schema-v1.1 CANDIDATE annotation request for a poem that
    has no v1.1 annotation yet. Does not call any model. Does not mutate its
    string arguments (they are only read, never assigned into)."""
    system_prompt = (
        "You are an annotation engine producing a schema-v1.1 CANDIDATE annotation "
        "for a single MorphoVerse++ poem, for the five-poem pilot described in "
        "docs/PILOT_AND_PROMPT_V1_1.md. Return exactly one minified JSON object and "
        "nothing else: no markdown, no code fences, no commentary, no chain-of-thought, "
        "no explanation outside the JSON.\n\n" + _CANDIDATE_LABEL_RULES
    )

    user_prompt = f"""\
POEM_ID: {poem_id}
LANGUAGE: {language}

ORIGINAL POEM (line-referenced; use ONLY these L<n> numbers for line_ref):
{_render_line_indexed_original(original_poem)}

TRANSLATION (English; for context and source_span_translation only, not line-numbered):
{_render_translation(translated_poem)}

{_enum_reference_block()}

{_SPAN_AND_LINE_REF_RULES}

{_NO_HALLUCINATION_RULES}

VISUAL-VARIANT AND CONFUSION RULES (mandatory):
- acceptable_visual_variants entries must be meaningfully DISTINCT from each
  other while remaining EQUALLY faithful to the same textual evidence — do
  not list near-duplicates, and do not prescribe a single fixed scene
  (docs/SCHEMA_V1_1_DECISIONS.md's gold-annotation principle applies to
  candidates too, as the standard to aim for).
- negative_confusions entries must be plausible INCORRECT renderings a
  well-meaning illustrator might mistakenly produce (e.g. confusing one
  specific named figure/place/practice for a generic or neighboring one) —
  not generic stereotypes disconnected from this poem's specific evidence.

METAPHOR_MAPPING STRUCTURE (when provided):
{{"vehicle_concept": "<short text>", "tenor_concept": "<short text>", "transferred_attributes": ["<short text>", "..."]}}
metaphor_mapping itself is optional/nullable — do not invent one for every
figurative expression; only include it when vehicle and tenor concepts are
both clearly evidenced.

REQUIRED JSON SHAPE (only these keys; use null or [] where evidence is
insufficient rather than omitting a key or guessing a value):
{_FULL_JSON_SHAPE}

Return the minified JSON object now."""

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_schema_version=MORPHOVERSE_SCHEMA_VERSION,
        prompt_kind=PROMPT_KIND_CANDIDATE,
    )


def build_v1_1_patch_prompt(
    poem_id: str,
    language: str,
    original_poem: str,
    translated_poem: str,
    existing_annotation: dict,
    requested_field_paths: list[str] | tuple[str, ...],
) -> PromptBundle:
    """Build a narrow PATCH request for backfilling only the fields listed in
    `requested_field_paths` onto an EXISTING annotation. Stage 5's concern;
    this function only builds the prompt text — it never applies a patch,
    never calls a model, and never mutates `existing_annotation` (it is only
    read, via .get()/iteration, for context rendering)."""
    if not requested_field_paths:
        raise ValueError(
            "requested_field_paths must be a non-empty, explicit list of field paths "
            "(e.g. ['cultural_entities[2].gloss']). A patch prompt with nothing "
            "requested would have no well-defined purpose."
        )
    requested = tuple(requested_field_paths)

    requested_list_block = "\n".join(f"  - {p}" for p in requested)

    system_prompt = (
        "You are an annotation engine producing a PATCH — not a full annotation — "
        "for a single MorphoVerse++ poem that already has a schema-v1.1 candidate "
        "annotation. Return exactly one minified JSON object containing ONLY the "
        "requested fields and nothing else: no markdown, no code fences, no "
        "commentary, no chain-of-thought, no explanation outside the JSON.\n\n"
        + _CANDIDATE_LABEL_RULES
        + "\n\nThe EXISTING annotation shown below is itself a candidate (or, if it "
        "originated from output_v3, a legacy Gemini candidate) — it is not gold "
        "either, but it is the CURRENT STATE you must not disturb outside the "
        "fields you were explicitly asked to fill in."
    )

    user_prompt = f"""\
POEM_ID: {poem_id}
LANGUAGE: {language}

ORIGINAL POEM (line-referenced; use ONLY these L<n> numbers for line_ref):
{_render_line_indexed_original(original_poem)}

TRANSLATION (English; for context and source_span_translation only, not line-numbered):
{_render_translation(translated_poem)}

EXISTING ANNOTATION (current state — context only, not gold, not to be rewritten):
{json.dumps(existing_annotation, ensure_ascii=False, indent=2)}

REQUESTED FIELD PATHS (fill in ONLY these; everything else is IMMUTABLE):
{requested_list_block}

PATCH SAFETY RULES (mandatory):
- Return a PATCH OBJECT ONLY: a JSON object containing values for the
  requested field paths above, and nothing else.
- Every field NOT listed above is IMMUTABLE. Do not rewrite, delete,
  rename, reinterpret, or otherwise change any field that is not explicitly
  requested, even if you believe it is wrong or could be improved.
- Do not return any field outside the requested list, even a field that
  seems related or helpful.
- If you are not confident about a requested field's value, return null for
  that field rather than guessing.

{_enum_reference_block()}

{_SPAN_AND_LINE_REF_RULES}

{_NO_HALLUCINATION_RULES}

{_build_source_span_translation_block(existing_annotation, translated_poem, requested)}

{_FINAL_SELF_CHECK}

Return the minified JSON patch object now, containing only:
{requested_list_block}"""

    return PromptBundle(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expected_schema_version=MORPHOVERSE_SCHEMA_VERSION,
        prompt_kind=PROMPT_KIND_PATCH,
        requested_field_paths=requested,
    )
