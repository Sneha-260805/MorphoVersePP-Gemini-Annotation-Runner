"""Stage 3 — line references and Unicode-safe textual grounding for schema v1.1.

Schema validation (models.py's `validate_*_v1_1` functions) and textual
grounding (this module) are deliberately separate concerns. Schema
validation asks "is this payload shaped correctly?" — it never needs the
actual poem text. Grounding asks "does this span/line reference actually
point at real text in this specific poem?" — it always needs the poem text,
and it never raises to halt processing; it returns a list of structured
`GroundingIssue`s for the caller to act on (a poem-level, evidence-quality
concern much closer to `apply_source_term_gate`'s existing `review_items`
than to `SchemaValidationError`'s fail-fast contract).

This module is intentionally dependency-light: it uses only the standard
library. It deliberately does NOT import `dataset.normalize_poem_text` for
line splitting (Stage 3.1 correction) — that helper applies an outer
`.strip()` to the *entire* poem text, which would silently remove
meaningful leading whitespace from the poem's first line or trailing
whitespace from its last line before grounding ever sees it. Only the
line-ENDING normalization (`\r\n`/`\r` -> `\n`) is shared logic; it is
reimplemented locally as `_normalize_line_endings` rather than reusing
`normalize_poem_text`'s stricter (whole-text-stripping) behavior. This
module does not import from `models.py`, so `models.py` can safely import
the small, purely-syntactic `parse_line_ref`/`LineRefError` from here
without creating a cycle.

See docs/GROUNDING_AND_LINE_REFERENCES.md for the full contract, and
docs/SCHEMA_V1_1_DECISIONS.md for how this relates to the existing
production `apply_source_term_gate`.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence


def _normalize_line_endings(text: str) -> str:
    """CRLF/CR -> LF only, for logical line splitting. Deliberately does NOT
    also strip the whole text (unlike dataset.normalize_poem_text) — see the
    module docstring and build_line_index for why an outer strip is unsafe
    here (Stage 3.1 correction)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


# ══════════════════════════════════════════════════════════════════════════
# Line references
# ══════════════════════════════════════════════════════════════════════════


class LineRefError(ValueError):
    """Raised by parse_line_ref()/resolve_line_ref() for a malformed,
    reversed, or out-of-range line reference. This is a low-level parsing
    error, not itself a GroundingIssue — callers in this module and in
    models.py catch it and convert it into the appropriate structured form
    (a SchemaValidationError for pure syntax, a GroundingIssue for
    poem-specific resolution)."""


@dataclass(frozen=True)
class ParsedLineRef:
    """A syntactically valid line reference, not yet resolved against any
    particular poem's line index. start == end for a single-line reference."""
    start: int
    end: int


# Canonical format only: "L<n>" or "L<n>-L<m>", 1-based, no leading zeros,
# no whitespace, uppercase "L". Loose forms ("line 3", "3", "stanza 2", "l1",
# "L03") are deliberately not accepted — see docs/GROUNDING_AND_LINE_REFERENCES.md.
_LINE_REF_SINGLE = re.compile(r"^L([1-9]\d*)$")
_LINE_REF_RANGE = re.compile(r"^L([1-9]\d*)-L([1-9]\d*)$")


def parse_line_ref(line_ref: str) -> ParsedLineRef:
    """Parse a line_ref string's SYNTAX only — does not check it against any
    poem's actual line count. Raises LineRefError for anything malformed or
    reversed, never a bare exception."""
    if not isinstance(line_ref, str):
        raise LineRefError(f"line_ref must be a string, not {type(line_ref).__name__}.")
    candidate = line_ref.strip()

    single = _LINE_REF_SINGLE.match(candidate)
    if single:
        n = int(single.group(1))
        return ParsedLineRef(start=n, end=n)

    rng = _LINE_REF_RANGE.match(candidate)
    if rng:
        start, end = int(rng.group(1)), int(rng.group(2))
        if end < start:
            raise LineRefError(f"line_ref range must not be reversed (start > end): {line_ref!r}.")
        return ParsedLineRef(start=start, end=end)

    raise LineRefError(
        f"line_ref {line_ref!r} is not a recognized reference. "
        "Expected 'L<n>' (e.g. 'L1') or 'L<n>-L<m>' (e.g. 'L2-L4')."
    )


@dataclass(frozen=True)
class IndexedLine:
    """One numbered, non-blank line of a poem.

    line_number: 1-based canonical number, contiguous across the WHOLE poem
        (blank stanza-separator lines are never numbered, so this is not the
        same as the line's raw position in the original text).
    text: the line's EXACT raw content — leading spaces, trailing spaces,
        punctuation, and every Unicode code point (including combining
        characters) are preserved unchanged (Stage 3.1 correction; this
        previously matched dataset.split_stanzas's trimmed convention, which
        is right for that module's own purpose but was wrong for grounding,
        where an exact match must be checked against what the poem actually
        contains). Only the line-ending characters themselves (`\r\n`/`\r`)
        are normalized away, since they are not part of any line's content.
    position: 0-based index into the sequence of numbered lines; always
        equal to line_number - 1. Provided for callers that prefer 0-based
        indexing without recomputing it.
    stanza_index: 1-based stanza this line belongs to, computed with the
        same blank-line-grouping rule dataset.split_stanzas already uses —
        this always agrees with StanzaInput.stanza_index elsewhere in the
        pipeline for the same poem text.
    """
    line_number: int
    text: str
    position: int
    stanza_index: int


@dataclass(frozen=True)
class LineIndex:
    """An immutable, ordered index of a poem's numbered lines. Never
    constructed by mutating its input text — build_line_index() only reads it."""
    lines: tuple[IndexedLine, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.lines)

    def get(self, line_number: int) -> IndexedLine | None:
        if 1 <= line_number <= len(self.lines):
            return self.lines[line_number - 1]
        return None


def build_line_index(text: str) -> LineIndex:
    """Pure function: poem text -> LineIndex. Never mutates `text`.

    CRLF and LF inputs produce an identical logical line index — only line
    ENDINGS are normalized (\\r\\n/\\r -> \\n) before splitting; line
    NUMBERING and stored line TEXT never change between the two, only the
    transient in-memory string representation does.

    Unlike Stage 3's original implementation, this does NOT apply an outer
    `.strip()` to the whole poem (that could silently drop meaningful
    leading whitespace from the first line, or trailing whitespace from the
    last), and does NOT store each line's `.strip()`ed text (that would
    silently drop meaningful leading/trailing whitespace from every line).
    Blankness — the only thing that determines whether a line becomes a
    numbered entry or an unnumbered stanza separator — is still detected via
    `raw_line.strip() == ""`; the stripped value itself is only used for
    that boolean check and is never stored (Stage 3.1 correction).
    """
    normalized = _normalize_line_endings(text) if text else ""
    raw_lines = normalized.split("\n") if normalized else []

    indexed: list[IndexedLine] = []
    stanza_index = 0
    in_stanza = False
    for raw_line in raw_lines:
        is_blank = raw_line.strip() == ""
        if is_blank:
            in_stanza = False
            continue
        if not in_stanza:
            stanza_index += 1
            in_stanza = True
        indexed.append(IndexedLine(
            line_number=len(indexed) + 1,
            text=raw_line,
            position=len(indexed),
            stanza_index=stanza_index,
        ))
    return LineIndex(lines=tuple(indexed))


def resolve_line_ref(line_ref: str, line_index: LineIndex) -> tuple[IndexedLine, ...]:
    """Parse + bounds-check a line_ref against a specific LineIndex, returning
    the IndexedLine(s) it covers (a 1-tuple for a single line). Raises
    LineRefError for anything parse_line_ref() would reject, and additionally
    for out-of-range references (start < 1 or end beyond the poem's numbered
    line count) or a start > 0 with an empty index."""
    parsed = parse_line_ref(line_ref)
    if parsed.start < 1 or parsed.end > len(line_index) or len(line_index) == 0:
        raise LineRefError(
            f"line_ref {line_ref!r} is out of range for a poem with {len(line_index)} numbered line(s)."
        )
    lines = tuple(line_index.lines[parsed.start - 1: parsed.end])
    if not lines:
        # Not reachable given the bounds check above (numbering has no gaps,
        # so any in-range slice is non-empty) — kept as an explicit, defensive
        # restatement of "ranges containing no textual lines are rejected".
        raise LineRefError(f"line_ref {line_ref!r} resolves to no textual lines.")
    return lines


# ══════════════════════════════════════════════════════════════════════════
# Span matching
# ══════════════════════════════════════════════════════════════════════════

SPAN_MATCH_EXACT = "exact"
SPAN_MATCH_NFC_EQUIVALENT = "nfc_equivalent"
SPAN_MATCH_NOT_FOUND = "not_found"
SPAN_MATCH_AMBIGUOUS = "ambiguous"
SPAN_MATCH_INVALID_INPUT = "invalid_input"


@dataclass(frozen=True)
class SpanMatch:
    """The result of searching for a span within a set of poem lines.

    Never carries the poem or line text itself — only line references
    (e.g. "L3"), a status, an occurrence count, and whether the match
    required NFC normalization. Safe to include verbatim in a GroundingIssue
    or a log line.
    """
    status: str
    line_refs: tuple[str, ...] = ()
    occurrence_count: int = 0
    normalized: bool = False


def _count_overlapping_occurrences(text: str, span: str) -> int:
    """Count occurrences of `span` in `text` by START POSITION, including
    overlapping matches. Unlike str.count() (which advances past the whole
    matched substring and so is non-overlapping — e.g. "aaa".count("aa")
    is 1, not 2), this advances by exactly one character after each match,
    so text="aaa", span="aa" correctly counts 2 occurrences (at position 0
    and position 1). Stage 3.1 correction — repeated occurrences must never
    be undercounted, since undercounting could hide a real ambiguity and
    let a match be treated as unique when it is not."""
    if not span:
        return 0
    count = 0
    start = 0
    while True:
        idx = text.find(span, start)
        if idx == -1:
            break
        count += 1
        start = idx + 1
    return count


def _occurrences_per_line(span: str, lines: Sequence[IndexedLine]) -> list[tuple[IndexedLine, int]]:
    result: list[tuple[IndexedLine, int]] = []
    for ln in lines:
        count = _count_overlapping_occurrences(ln.text, span)
        if count > 0:
            result.append((ln, count))
    return result


def match_span_in_lines(span: str, lines: Sequence[IndexedLine]) -> SpanMatch:
    """Search `lines` for `span` using exact code-point substring matching
    first, then (only if exact matching finds nothing at all) Unicode NFC
    normalized substring matching. No fuzzy matching, no case-folding, no
    whitespace collapsing, no diacritic stripping, no transliteration —
    Task 5's rules 4-8. Every occurrence in every candidate line is counted;
    more than one total occurrence across the searched lines is `ambiguous`,
    never silently resolved to the first one (design principle 10)."""
    if not isinstance(span, str) or not span:
        return SpanMatch(status=SPAN_MATCH_INVALID_INPUT)

    exact = _occurrences_per_line(span, lines)
    total_exact = sum(count for _, count in exact)
    if total_exact == 1:
        (matched_line, _), = exact
        return SpanMatch(status=SPAN_MATCH_EXACT, line_refs=(f"L{matched_line.line_number}",), occurrence_count=1)
    if total_exact > 1:
        return SpanMatch(
            status=SPAN_MATCH_AMBIGUOUS,
            line_refs=tuple(f"L{ln.line_number}" for ln, _ in exact),
            occurrence_count=total_exact,
        )

    # Exact matching found nothing anywhere in scope — try NFC-equivalent.
    span_nfc = unicodedata.normalize("NFC", span)
    nfc_lines = [
        IndexedLine(ln.line_number, unicodedata.normalize("NFC", ln.text), ln.position, ln.stanza_index)
        for ln in lines
    ]
    nfc_matches = _occurrences_per_line(span_nfc, nfc_lines)
    total_nfc = sum(count for _, count in nfc_matches)
    if total_nfc == 1:
        (matched_line, _), = nfc_matches
        return SpanMatch(
            status=SPAN_MATCH_NFC_EQUIVALENT, line_refs=(f"L{matched_line.line_number}",),
            occurrence_count=1, normalized=True,
        )
    if total_nfc > 1:
        return SpanMatch(
            status=SPAN_MATCH_AMBIGUOUS,
            line_refs=tuple(f"L{ln.line_number}" for ln, _ in nfc_matches),
            occurrence_count=total_nfc, normalized=True,
        )
    return SpanMatch(status=SPAN_MATCH_NOT_FOUND)


def ground_original_span(span: str, line_ref: str | None, line_index: LineIndex) -> SpanMatch:
    """Locate `span` in the ORIGINAL poem's line index. If `line_ref` is
    given, search is restricted to exactly the line(s) it resolves to
    (Task 5 rule 9) — a span that exists elsewhere in the poem but not
    within the referenced line(s) is `not_found`, not silently accepted
    (Task 5 rule 12 / test "Incorrect line_ref is rejected even when the
    span exists elsewhere"). If `line_ref` itself is malformed or
    out-of-range, that is reported as `invalid_input` here; the caller
    (validate_cultural_grounding_v1_1 / validate_figurative_grounding_v1_1)
    separately reports the malformed line_ref itself and may re-attempt an
    unrestricted search for a more informative combined diagnosis."""
    if not isinstance(span, str) or not span:
        return SpanMatch(status=SPAN_MATCH_INVALID_INPUT)
    if line_ref is not None:
        try:
            scope = resolve_line_ref(line_ref, line_index)
        except LineRefError:
            return SpanMatch(status=SPAN_MATCH_INVALID_INPUT)
    else:
        scope = line_index.lines
    return match_span_in_lines(span, scope)


def ground_translation_span(span: str, translated_line_index: LineIndex) -> SpanMatch:
    """Locate `span` in the TRANSLATED poem's line index. line_ref is never
    used to restrict this search: line_ref refers to the original-language
    poem's line numbering (Task 2), and the translated poem's own lines do
    not necessarily correspond 1:1 to the original's — using the original's
    line_ref to restrict the translation search would silently assume an
    alignment this transitional schema does not establish. See
    docs/GROUNDING_AND_LINE_REFERENCES.md."""
    if not isinstance(span, str) or not span:
        return SpanMatch(status=SPAN_MATCH_INVALID_INPUT)
    return match_span_in_lines(span, translated_line_index.lines)


# ══════════════════════════════════════════════════════════════════════════
# Structured grounding issues and validation modes
# ══════════════════════════════════════════════════════════════════════════

GROUNDING_MODE_TRANSITIONAL_CANDIDATE = "transitional_candidate"
GROUNDING_MODE_STRICT_ELIGIBILITY = "strict_eligibility"
_VALID_GROUNDING_MODES = (GROUNDING_MODE_TRANSITIONAL_CANDIDATE, GROUNDING_MODE_STRICT_ELIGIBILITY)


@dataclass(frozen=True)
class GroundingIssue:
    """A single structured grounding problem. Distinct from models.py's
    ValidationIssue/SchemaValidationError by design (Task 7): grounding is a
    collected-list-of-diagnostics concern (like apply_source_term_gate's
    review_items), not a fail-fast schema-shape concern. Never carries the
    full poem text or unrelated payload content — only the offending
    field's own value, where useful."""
    object_type: str          # "cultural_entity" | "figurative_expression" | "translation_loss"
    field: str                 # e.g. "line_ref", "source_span_original", "source_span_translation"
    code: str                   # short machine-readable code, see docs/GROUNDING_AND_LINE_REFERENCES.md
    message: str                 # human-readable, scoped to this one field
    index: int | None = None       # item's position within its containing list
    severity: str = "review"        # "review" (transitional-mode default) | "error"


def _translation_status_hint(entity_or_expr: dict) -> str | None:
    value = entity_or_expr.get("translation_status")
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    return None


# Because the final translation_status enum is unresolved (see
# docs/SCHEMA_V1_1_DECISIONS.md), this recognizes only a small, explicitly
# documented set of informal hint strings — anything else is left
# unresolved/uninterpreted rather than guessed at. This casefold comparison
# is about matching a handful of known ENGLISH status tokens, not about
# grounding Indic-script source text, so it does not conflict with Task 5
# rule 7 (no case-folding for source-text grounding).
_PRESERVED_STATUS_HINTS = frozenset({"preserved"})
_LOST_STATUS_HINTS = frozenset({"lost", "omitted", "not_translated", "untranslated"})


def _preserved_signal(entity_or_expr: dict) -> bool:
    if entity_or_expr.get("preserved") is True:
        return True
    return _translation_status_hint(entity_or_expr) in _PRESERVED_STATUS_HINTS


def _lost_signal(entity_or_expr: dict) -> bool:
    return _translation_status_hint(entity_or_expr) in _LOST_STATUS_HINTS


def _ground_spans(
    obj: dict,
    *,
    object_type: str,
    original_index: LineIndex,
    translated_index: LineIndex,
    mode: str,
    index: int | None,
) -> list[GroundingIssue]:
    """Shared grounding logic for one cultural entity or figurative
    expression object. Read-only: never mutates `obj`."""
    if mode not in _VALID_GROUNDING_MODES:
        raise ValueError(f"Unknown grounding mode: {mode!r}. Expected one of {_VALID_GROUNDING_MODES}.")

    issues: list[GroundingIssue] = []
    line_ref = obj.get("line_ref")
    line_ref_valid = False

    if line_ref is not None:
        try:
            resolve_line_ref(line_ref, original_index)
            line_ref_valid = True
        except LineRefError as exc:
            issues.append(GroundingIssue(
                object_type=object_type, field="line_ref", code="line_ref_invalid",
                message=str(exc), index=index, severity="error",
            ))

    source_span = obj.get("source_span_original")
    if source_span:
        scoped_line_ref = line_ref if line_ref_valid else None
        match = ground_original_span(source_span, scoped_line_ref, original_index)
        if match.status == SPAN_MATCH_NOT_FOUND and scoped_line_ref is not None:
            # Distinguish "wrong line_ref" from "doesn't exist anywhere" (Task 7:
            # consistency between line_ref and source_span_original).
            whole_poem_match = ground_original_span(source_span, None, original_index)
            if whole_poem_match.status in (SPAN_MATCH_EXACT, SPAN_MATCH_NFC_EQUIVALENT, SPAN_MATCH_AMBIGUOUS):
                issues.append(GroundingIssue(
                    object_type=object_type, field="line_ref", code="line_ref_span_mismatch",
                    message=(
                        f"source_span_original was not found within {line_ref}, but a matching "
                        f"occurrence exists elsewhere in the poem (at {', '.join(whole_poem_match.line_refs)})."
                    ),
                    index=index, severity="error",
                ))
            else:
                issues.append(GroundingIssue(
                    object_type=object_type, field="source_span_original", code="span_not_found",
                    message="source_span_original was not found anywhere in the original poem text.",
                    index=index, severity="error",
                ))
        elif match.status == SPAN_MATCH_NOT_FOUND:
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_original", code="span_not_found",
                message="source_span_original was not found anywhere in the original poem text.",
                index=index, severity="error",
            ))
        elif match.status == SPAN_MATCH_AMBIGUOUS:
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_original", code="span_ambiguous",
                message=(
                    f"source_span_original occurs {match.occurrence_count} times in the searched scope; "
                    "line_ref does not uniquely identify one occurrence."
                ),
                index=index, severity="review",
            ))
        elif match.status == SPAN_MATCH_INVALID_INPUT:
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_original", code="span_invalid_input",
                message="source_span_original could not be searched (empty or non-string).",
                index=index, severity="error",
            ))
    elif mode == GROUNDING_MODE_STRICT_ELIGIBILITY:
        issues.append(GroundingIssue(
            object_type=object_type, field="source_span_original", code="required_field_missing",
            message="strict eligibility mode requires source_span_original to be present.",
            index=index, severity="error",
        ))

    if mode == GROUNDING_MODE_STRICT_ELIGIBILITY and not line_ref:
        issues.append(GroundingIssue(
            object_type=object_type, field="line_ref", code="required_field_missing",
            message="strict eligibility mode requires line_ref to be present.",
            index=index, severity="error",
        ))

    translation_span = obj.get("source_span_translation")
    translation_not_found = False
    if translation_span:
        t_match = ground_translation_span(translation_span, translated_index)
        if t_match.status == SPAN_MATCH_NOT_FOUND:
            translation_not_found = True
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_translation", code="translation_span_not_found",
                message="source_span_translation was not found in the translated poem text.",
                index=index, severity="error",
            ))
        elif t_match.status == SPAN_MATCH_AMBIGUOUS:
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_translation", code="translation_span_ambiguous",
                message=f"source_span_translation occurs {t_match.occurrence_count} times in the translated poem text.",
                index=index, severity="review",
            ))
        elif t_match.status == SPAN_MATCH_INVALID_INPUT:
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_translation", code="translation_span_invalid_input",
                message="source_span_translation could not be searched (empty or non-string).",
                index=index, severity="error",
            ))

    # Preservation/translation evidence consistency (Task 6 rule 6-7 / Task 7).
    if _preserved_signal(obj) and not _lost_signal(obj):
        if not translation_span or translation_not_found:
            severity = "error" if mode == GROUNDING_MODE_STRICT_ELIGIBILITY else "review"
            issues.append(GroundingIssue(
                object_type=object_type, field="source_span_translation",
                code="translation_evidence_missing_for_preserved_cue",
                message=(
                    "preserved/translation_status indicates this cue was preserved, but no locatable "
                    "source_span_translation is present — expected translated evidence."
                ),
                index=index, severity=severity,
            ))

    return issues


def validate_cultural_grounding_v1_1(
    entity: dict,
    original_poem: str,
    translated_poem: str,
    *,
    mode: str = GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
    index: int | None = None,
) -> list[GroundingIssue]:
    """Grounding for one v1.1 cultural entity (already schema-validated by
    models.validate_cultural_entity_v1_1 — this function does not repeat
    structural checks and does not raise bare KeyError/TypeError/ValueError
    for a malformed `entity`; a non-dict input simply yields one GroundingIssue)."""
    if not isinstance(entity, dict):
        return [GroundingIssue(
            object_type="cultural_entity", field="", code="not_an_object",
            message=f"cultural entity must be an object, not {type(entity).__name__}.",
            index=index, severity="error",
        )]
    original_index = build_line_index(original_poem)
    translated_index = build_line_index(translated_poem)
    return _ground_spans(
        entity, object_type="cultural_entity", original_index=original_index,
        translated_index=translated_index, mode=mode, index=index,
    )


def validate_figurative_grounding_v1_1(
    expression: dict,
    original_poem: str,
    translated_poem: str,
    *,
    mode: str = GROUNDING_MODE_TRANSITIONAL_CANDIDATE,
    index: int | None = None,
) -> list[GroundingIssue]:
    """Grounding for one v1.1 figurative expression (a stanzas[].metaphor_spans
    item). Does NOT derive source_span_original from the legacy source_term
    field — an item with source_term but no source_span_original is treated
    exactly like any other item with that field absent (Task 8)."""
    if not isinstance(expression, dict):
        return [GroundingIssue(
            object_type="figurative_expression", field="", code="not_an_object",
            message=f"figurative expression must be an object, not {type(expression).__name__}.",
            index=index, severity="error",
        )]
    original_index = build_line_index(original_poem)
    translated_index = build_line_index(translated_poem)
    return _ground_spans(
        expression, object_type="figurative_expression", original_index=original_index,
        translated_index=translated_index, mode=mode, index=index,
    )


def validate_translation_loss_grounding(
    translation_loss_items: list[dict],
    original_poem: str,
) -> list[GroundingIssue]:
    """Validate translation_loss[].where against the ORIGINAL poem's line
    index (Task 2 rule 8: `where` uses the same convention as line_ref).
    Does not validate what_was_lost/severity content (Task 9)."""
    issues: list[GroundingIssue] = []
    if not translation_loss_items:
        return issues
    original_index = build_line_index(original_poem)
    for idx, item in enumerate(translation_loss_items):
        if not isinstance(item, dict):
            issues.append(GroundingIssue(
                object_type="translation_loss", field="", code="not_an_object",
                message=f"translation_loss[{idx}] must be an object, not {type(item).__name__}.",
                index=idx, severity="error",
            ))
            continue
        where = item.get("where")
        if where is None:
            continue
        try:
            resolve_line_ref(where, original_index)
        except LineRefError as exc:
            issues.append(GroundingIssue(
                object_type="translation_loss", field="where", code="line_ref_invalid",
                message=str(exc), index=idx, severity="error",
            ))
    return issues
