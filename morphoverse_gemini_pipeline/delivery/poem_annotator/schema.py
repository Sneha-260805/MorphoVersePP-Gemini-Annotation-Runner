"""Canonical schema definition — the single source of truth.

Every other module (prompt builder, validator, assembler, output writer) imports
its key sets and enums from here so the schema can never drift across files.

NOTE: `visual_motifs` has been removed from the schema entirely.
"""
from __future__ import annotations

# ── Enumerated label sets ────────────────────────────────────────────────────
ALLOWED_RECITATION_STYLES = (
    "lament", "devotional", "celebratory", "reflective", "declarative",
)
ALLOWED_EMOTIONS = (
    "grief", "longing", "devotion", "peace", "celebration",
    "resilience", "anger", "fear",
)
ALLOWED_TONES = (
    "lament", "whisper", "declaration", "prayer",
    "wonder", "defiance", "tenderness",
)
ALLOWED_TRANSLATION_QUALITIES = ("faithful", "partial", "lost")
ALLOWED_ENTITY_CATEGORIES = (
    "DEITY",
    "SACRED_RIVER",
    "FESTIVAL",
    "MUSICAL_TRADITION",
    "DEVOTIONAL_CONCEPT",
    "REGIONAL_SYMBOL",
    "SOCIAL_CUSTOM",
    "MYTHOLOGICAL_EVENT",
)

# ── Exact key sets accepted from the model (no visual_motifs anywhere) ───────
TOPLEVEL_KEYS = frozenset({"recitation_style", "emotional_arc", "stanzas", "cultural_entities"})
STANZA_KEYS = frozenset({"index", "emotion", "tone", "translation_quality", "loss_note", "metaphor_spans"})
METAPHOR_KEYS = frozenset({"source_term", "abstract_meaning"})
ENTITY_KEYS = frozenset({"term", "romanization", "category", "stanza_index", "preserved", "translation_note"})

# ── Status / confidence vocabularies ─────────────────────────────────────────
STATUS_COMPLETED = "completed"
STATUS_SALVAGED = "salvaged"      # valid JSON but hallucinated terms were dropped
STATUS_FAILED = "failed"          # no valid annotation produced
STATUS_PENDING = "pending"

ALIGNMENT_OK = "aligned"
ALIGNMENT_LOW = "aligned_low"
ALIGNMENT_RISK = "alignment_risk"

# ── Abbreviated key maps (model outputs compact keys; pipeline expands before validation) ──
# These save ~40% output tokens and allow simple annotations to fit in the proxy's
# ~37-token completion budget.
ABBREV_TOPLEVEL = {
    "rs": "recitation_style",
    "ea": "emotional_arc",
    "st": "stanzas",
    "ce": "cultural_entities",
}
ABBREV_STANZA = {
    "i":  "index",
    "em": "emotion",
    "to": "tone",
    "tq": "translation_quality",
    "ln": "loss_note",
    "ms": "metaphor_spans",
}
ABBREV_METAPHOR = {
    "src": "source_term",
    "st":  "source_term",   # fallback: format previously showed "st" for source_term
    "am":  "abstract_meaning",
}
ABBREV_ENTITY = {
    "tm": "term",
    "rm": "romanization",
    "ct": "category",
    "si": "stanza_index",
    "pr": "preserved",
    "tn": "translation_note",
}

# ── Schema v1.1 (Stage 2) ─────────────────────────────────────────────────────
# Additive only: nothing above this line is removed, renamed, or reinterpreted.
# See docs/SCHEMA_V1_1_DECISIONS.md for the full rationale, including which
# stored key names are intentionally unchanged in this stage (cultural_entities,
# term, metaphor_spans, source_term, abstract_meaning, preserved,
# translation_quality, loss_note) and which naming/enum decisions remain
# unresolved (cultural_entities vs cultural_cues, term vs term_original, the
# final translation_status enum, etc.).
#
# LEGACY_SCHEMA_VERSION preserves the historical meaning of the integer 5
# already used throughout config.py/main.py/output.py (unchanged by this
# stage). MORPHOVERSE_SCHEMA_VERSION is a new, explicitly-typed marker (a
# string, not an int) for the v1.1 envelope, so a legacy record can never be
# mistaken for a v1.1 record by an equality check gone wrong.
LEGACY_SCHEMA_VERSION = 5
MORPHOVERSE_SCHEMA_VERSION = "1.1"


def is_legacy_schema_version(value: object) -> bool:
    return value == LEGACY_SCHEMA_VERSION


def is_v1_1_schema_version(value: object) -> bool:
    return value == MORPHOVERSE_SCHEMA_VERSION


# ── V1.1 enumerated label sets (new; legacy enums above are unchanged) ───────
ALLOWED_VISUAL_PRIORITIES = ("essential", "supporting", "optional", "non_visual")
ALLOWED_EXPRESSION_TYPES = (
    "metaphor", "simile", "idiom", "proverb", "personification",
    "symbolism", "metonymy", "allusion", "wordplay", "other",
)
# NOTE: cultural_specificity_level and visualization_difficulty are pilot-only
# and intentionally have NO controlled enum yet (see docs/SCHEMA_V1_1_DECISIONS.md).
# They validate as an optional free-text string or null in models.py.

# ── V1.1 key sets ─────────────────────────────────────────────────────────────
# Each is the legacy key set plus new v1.1 fields — a strict superset, so a
# legacy-shaped object always satisfies the corresponding v1.1 key set too.
# Stored key NAMES are unchanged (metaphor_spans still holds figurative
# expressions of any expression_type, not just metaphors — see decisions doc).
TOPLEVEL_KEYS_V1_1 = TOPLEVEL_KEYS | {"theme"}
STANZA_KEYS_V1_1 = STANZA_KEYS | {"translation_loss"}
METAPHOR_KEYS_V1_1 = METAPHOR_KEYS | {
    "source_span_original", "source_span_translation", "expression_type",
    "literal_meaning", "vehicle", "tenor", "metaphor_mapping", "line_ref",
    "literalization_risk", "visualization_strategy", "acceptable_visual_variants",
    "visualization_difficulty",
}
ENTITY_KEYS_V1_1 = ENTITY_KEYS | {
    "gloss", "line_ref", "source_span_original", "source_span_translation",
    "visual_features", "visual_priority", "acceptable_visual_variants",
    "negative_confusions", "translation_status", "cultural_specificity_level",
}
# Each translation_loss list item (Task 6): a structured elaboration of the
# existing stanza-level loss_note, not a replacement for it.
TRANSLATION_LOSS_KEYS = frozenset({"what_was_lost", "where", "severity"})
# metaphor_mapping's own shape (Task 5) — see docs/SCHEMA_V1_1_DECISIONS.md for
# why this representation was chosen over mechanically splitting abstract_meaning.
METAPHOR_MAPPING_KEYS = frozenset({"vehicle_concept", "tenor_concept", "transferred_attributes"})
