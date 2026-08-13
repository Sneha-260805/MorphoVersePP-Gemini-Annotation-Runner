"""Stage 5K.4 — generic, reusable semantic-risk audit.

Rule-based (non-LLM), offline, read-only checks over an already-assembled,
already schema/completeness/grounding-valid candidate. Unlike
`post_generation_risk_audit_v1_1.py` (Stage 5K.3's Hindi/Telugu-specific
audit, which hardcodes per-language keyword lists tied to those two
cultures), every check here uses only GENERIC English risk-domain marker
words and purely structural/statistical heuristics -- nothing here names a
pilot poem ID, a source term from a pilot poem, an expected category, an
expected metaphor, or an expected correction. The same fourteen checks run
unchanged over any language's candidate.

Per the task's explicit instruction, this audit NEVER rewrites the
candidate automatically. Every finding is PASS or REVIEW_REQUIRED;
OBJECTIVE_REPAIR_REQUIRED is defined for API completeness but is never
emitted by any check in this module -- objective schema/completeness/
grounding failures are caught by the separate validation pipeline, not by
this semantic heuristic layer (Task: 'must not automatically rewrite
semantically debatable content').
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def _contains_word(text: str, marker: str) -> bool:
    """Whole-word/whole-phrase match only -- avoids false positives like
    'rape' matching inside 'draped', or 'god' matching inside 'good'."""
    return re.search(r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])", text) is not None

PASS = "PASS"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
OBJECTIVE_REPAIR_REQUIRED = "OBJECTIVE_REPAIR_REQUIRED"  # never emitted here; see module docstring

CATEGORIES = (
    "UNSUPPORTED_SPECIFICITY", "POLYSEMY_CONTEXT_MISMATCH", "CULTURAL_OVERCLAIM",
    "CATEGORY_CONTEXT_MISMATCH", "UNSUPPORTED_SOCIAL_INFERENCE", "UNSUPPORTED_RELIGIOUS_INFERENCE",
    "UNSUPPORTED_GENDER_INFERENCE", "TENOR_TOO_SPECIFIC", "VISUAL_VARIANT_OVERREACH",
    "TRANSLATION_LOSS_OVERCLAIM", "SOURCE_TEXT_ANOMALY", "AMBIGUITY_REQUIRES_REVIEW",
    "POSSIBLE_DUPLICATE_CONCEPT", "POSSIBLE_OVER_EXTRACTION",
)


@dataclass(frozen=True)
class RiskFinding:
    category: str
    status: str
    detail: str
    field_path: "str | None" = None


# ── Generic, language-agnostic marker word lists (English risk-domain terms
# only -- never a script-specific or poem-specific word) ────────────────────
_SPECIFICITY_MARKERS = (
    "caste", "widow", "dowry", "concubine", "slave", "militant", "massacre",
    "genocide", "suicide", "rape", "incest", "trafficking",
)
_CULTURAL_DOMAIN_MARKERS = (
    "temple", "shrine", "festival", "ritual", "deity", "god", "goddess",
    "pilgrimage", "scripture", "folk", "devotional", "mythological",
    "clergy", "priest", "monastery", "pagoda", "mosque", "church", "sufi",
)
_SOCIAL_MARKERS = (
    "caste", "class", "peasant", "laborer", "aristocrat", "untouchable",
    "landlord", "servant", "slave", "dowry", "occupation",
)
_RELIGIOUS_MARKERS = (
    "hindu", "muslim", "islam", "christian", "sikh", "buddhist", "sufi",
    "bhakti", "shaivite", "vaishnav", "quran", "bible", "gita", "namaz", "puja",
)
_GENDER_MARKERS = (
    "widow", "bride", "groom", "husband", "wife", "mother-in-law",
    "son-in-law", "daughter-in-law", "bachelor", "spinster",
)
_TENOR_SPECIFICITY_MARKERS = ("trauma", "abuse", "caste", "oppression", "religion", "occupation")
_SCENERY_MARKERS = (
    "temple", "shrine", "veil", "headdress", "turban", "folk instrument",
    "drum", "flute", "sitar", "tabla", "landscape", "village", "countryside",
    "mountain", "desert", "forest", "riverbank",
)
_HIGH_COMMITMENT_CATEGORIES = ("DEITY", "MYTHOLOGICAL_EVENT")
_HIGH_COMMITMENT_EVIDENCE_MARKERS = ("god", "goddess", "deity", "myth", "legend")


def _free_text_fields(candidate: dict) -> "list[tuple[str, str]]":
    ann = candidate.get("annotation", candidate)
    out: "list[tuple[str, str]]" = []
    for i, e in enumerate(ann.get("cultural_entities", [])):
        for f in ("translation_note", "gloss"):
            if e.get(f):
                out.append((f"cultural_entities[{i}].{f}", e[f]))
    for si, s in enumerate(ann.get("stanzas", [])):
        for mi, m in enumerate(s.get("metaphor_spans", []) or []):
            for f in ("literal_meaning", "vehicle", "tenor", "visualization_strategy", "abstract_meaning"):
                if m.get(f):
                    out.append((f"stanzas[{si}].metaphor_spans[{mi}].{f}", m[f]))
        for li, loss in enumerate(s.get("translation_loss", []) or []):
            if loss.get("what_was_lost"):
                out.append((f"stanzas[{si}].translation_loss[{li}].what_was_lost", loss["what_was_lost"]))
    return out


def _source_text_lower(original_poem: str, translated_poem: str) -> str:
    return (original_poem + " " + translated_poem).lower()


def _marker_grounding_check(candidate: dict, original_poem: str, translated_poem: str, markers: "tuple[str, ...]", category: str) -> RiskFinding:
    source_text = _source_text_lower(original_poem, translated_poem)
    flagged = []
    for path, text in _free_text_fields(candidate):
        lowered = text.lower()
        for kw in markers:
            if _contains_word(lowered, kw) and not _contains_word(source_text, kw):
                flagged.append(f"{path}: {kw!r} not found in source/translation text")
    if not flagged:
        return RiskFinding(category, PASS, f"No unsupported {category.lower()} marker found outside the poem's own text.")
    return RiskFinding(category, REVIEW_REQUIRED, f"Possible unsupported assumption(s): {flagged}")


def check_unsupported_specificity(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    return _marker_grounding_check(candidate, original_poem, translated_poem, _SPECIFICITY_MARKERS, "UNSUPPORTED_SPECIFICITY")


def check_polysemy_context_mismatch(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    by_term: "dict[str, set]" = {}
    for e in ann.get("cultural_entities", []):
        term = e.get("term")
        if not term:
            continue
        by_term.setdefault(term, set()).add((e.get("category"), (e.get("translation_note") or "").strip()))
    flagged = [t for t, variants in by_term.items() if len(variants) > 1]
    if not flagged:
        return RiskFinding("POLYSEMY_CONTEXT_MISMATCH", PASS, "No repeated cultural term received inconsistent category/interpretation.")
    return RiskFinding("POLYSEMY_CONTEXT_MISMATCH", REVIEW_REQUIRED, f"Repeated term(s) with inconsistent interpretation across occurrences (verify each is context-appropriate, not copy-pasted): {flagged}")


def check_cultural_overclaim(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    return _marker_grounding_check(candidate, original_poem, translated_poem, _CULTURAL_DOMAIN_MARKERS, "CULTURAL_OVERCLAIM")


def check_category_context_mismatch(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    flagged = []
    for i, e in enumerate(ann.get("cultural_entities", [])):
        category = e.get("category")
        if category not in _HIGH_COMMITMENT_CATEGORIES:
            continue
        evidence_text = f"{e.get('translation_note') or ''} {e.get('gloss') or ''}".lower()
        if not any(_contains_word(evidence_text, marker) for marker in _HIGH_COMMITMENT_EVIDENCE_MARKERS):
            flagged.append(f"cultural_entities[{i}] category={category!r} lacks explicit supporting evidence in its own translation_note/gloss")
    if not flagged:
        return RiskFinding("CATEGORY_CONTEXT_MISMATCH", PASS, "Every high-commitment category (DEITY/MYTHOLOGICAL_EVENT) has explicit supporting evidence in its own annotation text.")
    return RiskFinding("CATEGORY_CONTEXT_MISMATCH", REVIEW_REQUIRED, f"{flagged}")


def check_unsupported_social_inference(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    return _marker_grounding_check(candidate, original_poem, translated_poem, _SOCIAL_MARKERS, "UNSUPPORTED_SOCIAL_INFERENCE")


def check_unsupported_religious_inference(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    return _marker_grounding_check(candidate, original_poem, translated_poem, _RELIGIOUS_MARKERS, "UNSUPPORTED_RELIGIOUS_INFERENCE")


def check_unsupported_gender_inference(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    return _marker_grounding_check(candidate, original_poem, translated_poem, _GENDER_MARKERS, "UNSUPPORTED_GENDER_INFERENCE")


def check_tenor_too_specific(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    flagged = []
    for si, s in enumerate(ann.get("stanzas", [])):
        for mi, m in enumerate(s.get("metaphor_spans", []) or []):
            tenor = (m.get("tenor") or "").lower()
            if any(_contains_word(tenor, mk) for mk in _TENOR_SPECIFICITY_MARKERS):
                flagged.append(f"stanzas[{si}].metaphor_spans[{mi}].tenor={m.get('tenor')!r}")
    if not flagged:
        return RiskFinding("TENOR_TOO_SPECIFIC", PASS, "No overly specific sensitive-topic tenor found.")
    return RiskFinding("TENOR_TOO_SPECIFIC", REVIEW_REQUIRED, f"Tenors asserting a highly specific sensitive topic, requiring direct-textual-support review: {flagged}")


def check_visual_variant_overreach(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    source_text = _source_text_lower(original_poem, translated_poem)
    flagged = []
    for i, e in enumerate(ann.get("cultural_entities", [])):
        variants = e.get("acceptable_visual_variants") or []
        if len(variants) >= 2 and len({v.strip().lower() for v in variants}) < len(variants):
            flagged.append(f"cultural_entities[{i}] has duplicate/near-duplicate visual variants")
        for v in variants:
            lowered = (v or "").lower()
            for marker in _SCENERY_MARKERS:
                if _contains_word(lowered, marker) and not _contains_word(source_text, marker):
                    flagged.append(f"cultural_entities[{i}].acceptable_visual_variants: unsupported scenery marker {marker!r} in {v!r}")
    if not flagged:
        return RiskFinding("VISUAL_VARIANT_OVERREACH", PASS, "No duplicate variants or unsupported stereotypical scenery found (heuristic only).")
    return RiskFinding("VISUAL_VARIANT_OVERREACH", REVIEW_REQUIRED, f"{flagged}")


def check_translation_loss_overclaim(candidate: dict, original_poem: str, translated_poem: str) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    source_text = _source_text_lower(original_poem, translated_poem)
    flagged = []
    for si, s in enumerate(ann.get("stanzas", [])):
        for li, loss in enumerate(s.get("translation_loss", []) or []):
            if loss.get("severity") == "high" and loss.get("where") is None:
                flagged.append(f"stanzas[{si}].translation_loss[{li}]: severity=high with no line-reference evidence")
            text = (loss.get("what_was_lost") or "").lower()
            for marker in _CULTURAL_DOMAIN_MARKERS + _RELIGIOUS_MARKERS:
                if _contains_word(text, marker) and not _contains_word(source_text, marker):
                    flagged.append(f"stanzas[{si}].translation_loss[{li}]: unsupported marker {marker!r} not evidenced in source/translation")
    if not flagged:
        return RiskFinding("TRANSLATION_LOSS_OVERCLAIM", PASS, "Every translation_loss claim is evidence-scoped.")
    return RiskFinding("TRANSLATION_LOSS_OVERCLAIM", REVIEW_REQUIRED, f"{flagged}")


def check_source_text_anomaly(original_poem: str, translated_poem: str) -> RiskFinding:
    flagged = []
    for label, text in (("original_poem", original_poem), ("translated_poem", translated_poem)):
        if not text or not text.strip():
            flagged.append(f"{label} is empty")
            continue
        non_latin_chars = sum(1 for ch in text if ch.isalpha() and ord(ch) > 0x2AF)
        latin_words = __import__("re").findall(r"[A-Za-z]{2,}", text)
        if non_latin_chars > 0 and latin_words:
            flagged.append(f"{label} mixes a non-Latin script with embedded Latin-alphabet word(s) {latin_words[:5]} -- possible data-entry/OCR artifact, not an annotation defect")
        if __import__("re").search(r"(.)\1{5,}", text):
            flagged.append(f"{label} contains an abnormal repeated-character run")
    if not flagged:
        return RiskFinding("SOURCE_TEXT_ANOMALY", PASS, "No structural anomaly detected in the source or translated poem text.")
    return RiskFinding("SOURCE_TEXT_ANOMALY", REVIEW_REQUIRED, f"{flagged}")


def check_ambiguity_requires_review(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    flagged = []
    for i, e in enumerate(ann.get("cultural_entities", [])):
        if e.get("preserved") is None:
            flagged.append(f"cultural_entities[{i}].preserved is unresolved (None)")
        if e.get("translation_status") == "UNCERTAIN":
            flagged.append(f"cultural_entities[{i}].translation_status is UNCERTAIN")
    for si, s in enumerate(ann.get("stanzas", [])):
        for mi, m in enumerate(s.get("metaphor_spans", []) or []):
            if m.get("expression_type") == "metaphor" and not m.get("metaphor_mapping"):
                flagged.append(f"stanzas[{si}].metaphor_spans[{mi}] is expression_type=metaphor with no metaphor_mapping")
    if not flagged:
        return RiskFinding("AMBIGUITY_REQUIRES_REVIEW", PASS, "No explicit unresolved/uncertain field found.")
    return RiskFinding("AMBIGUITY_REQUIRES_REVIEW", REVIEW_REQUIRED, f"{flagged}")


def check_possible_duplicate_concept(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    terms = [e.get("term") for e in ann.get("cultural_entities", []) if e.get("term")]
    counts: "dict[str, int]" = {}
    for t in terms:
        counts[t] = counts.get(t, 0) + 1
    repeated = {t: c for t, c in counts.items() if c > 1}
    if not repeated:
        return RiskFinding("POSSIBLE_DUPLICATE_CONCEPT", PASS, "No cultural term is annotated as a separate entity more than once.")
    return RiskFinding("POSSIBLE_DUPLICATE_CONCEPT", REVIEW_REQUIRED, f"Terms annotated as multiple separate cultural_entities items: {repeated} -- review whether each occurrence is a genuinely distinct cue or a duplicate concept.")


def check_possible_over_extraction(candidate: dict) -> RiskFinding:
    ann = candidate.get("annotation", candidate)
    stanzas = ann.get("stanzas", [])
    stanza_count = max(len(stanzas), 1)
    entity_count = len(ann.get("cultural_entities", []))
    flagged = []
    if entity_count > 2 * stanza_count:
        flagged.append(f"{entity_count} cultural_entities across {stanza_count} stanzas exceeds the 2-per-stanza heuristic threshold")
    for si, s in enumerate(stanzas):
        n = len(s.get("metaphor_spans", []) or [])
        if n > 3:
            flagged.append(f"stanzas[{si}] has {n} metaphor_spans, exceeding the per-stanza heuristic threshold")
    if not flagged:
        return RiskFinding("POSSIBLE_OVER_EXTRACTION", PASS, "Entity/expression density is within the heuristic threshold for this poem's stanza count.")
    return RiskFinding("POSSIBLE_OVER_EXTRACTION", REVIEW_REQUIRED, f"{flagged}")


def run_semantic_risk_audit(candidate: dict, original_poem: str, translated_poem: str) -> "list[RiskFinding]":
    """The one generic, reusable audit entry point -- identical for every
    language, poem, and pilot stage. Nothing here branches on poem_id or
    language."""
    return [
        check_unsupported_specificity(candidate, original_poem, translated_poem),
        check_polysemy_context_mismatch(candidate),
        check_cultural_overclaim(candidate, original_poem, translated_poem),
        check_category_context_mismatch(candidate),
        check_unsupported_social_inference(candidate, original_poem, translated_poem),
        check_unsupported_religious_inference(candidate, original_poem, translated_poem),
        check_unsupported_gender_inference(candidate, original_poem, translated_poem),
        check_tenor_too_specific(candidate),
        check_visual_variant_overreach(candidate, original_poem, translated_poem),
        check_translation_loss_overclaim(candidate, original_poem, translated_poem),
        check_source_text_anomaly(original_poem, translated_poem),
        check_ambiguity_requires_review(candidate),
        check_possible_duplicate_concept(candidate),
        check_possible_over_extraction(candidate),
    ]


def summarize(findings: "list[RiskFinding]") -> dict:
    return {
        "findings": [{"category": f.category, "status": f.status, "detail": f.detail, "field_path": f.field_path} for f in findings],
        "pass_count": sum(1 for f in findings if f.status == PASS),
        "review_required_count": sum(1 for f in findings if f.status == REVIEW_REQUIRED),
        "objective_repair_required_count": sum(1 for f in findings if f.status == OBJECTIVE_REPAIR_REQUIRED),
        "candidate_automatically_rewritten": False,
        "categories_checked": list(CATEGORIES),
    }
