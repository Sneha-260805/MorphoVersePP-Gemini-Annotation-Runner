"""Stage 5M.4D — additive raw-substring multi-line grounding fallback.

Root cause (Stage 5M.4C read-only audit): `_match_span_across_lines`
(Stage 5N.3) only accepts a multi-line span that equals a join of N
*complete*, `.strip()`ed lines. A legitimate span that is an exact literal
substring of the RAW (unstripped) scope text, but starts or ends partway
through a line, can never match that way — reproduced exactly by
MV++_1339, whose model-returned span preserves the first line's raw
trailing whitespace but naturally ends before the last line's own trailing
whitespace.

This file exercises the new additive fallback
(`grounding._match_raw_multiline_substring`, wired into
`grounding.match_span_in_lines`) purely through the public grounding API
(`build_line_index`, `ground_original_span`, `ground_translation_span`).
No provider/network call anywhere in this file.
"""
from __future__ import annotations

import json
from pathlib import Path

from morphoverse_gemini_pipeline.delivery.poem_annotator import grounding as g

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── 1. raw multiline, partial LAST line ─────────────────────────────────────
def test_raw_multiline_partial_last_line_matches_exactly():
    original = "abc  \ndef  \n"  # L1='abc  ', L2='def  ' (raw trailing whitespace on both)
    index = g.build_line_index(original)
    span = "abc  \ndef"  # raw L1 (with trailing ws) + "\n" + L2 with trailing ws trimmed off
    match = g.ground_original_span(span, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_EXACT
    assert match.line_refs == ("L1-L2",)
    assert match.occurrence_count == 1


# ── 2. partial FIRST and LAST line ──────────────────────────────────────────
def test_raw_multiline_partial_first_and_last_line_matches_exactly():
    original = "xxx abc yyy\nzzz def www\n"  # L1='xxx abc yyy', L2='zzz def www'
    index = g.build_line_index(original)
    span = "abc yyy\nzzz def"  # begins after "xxx " in L1, ends before " www" in L2
    match = g.ground_original_span(span, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_EXACT
    assert match.line_refs == ("L1-L2",)


# ── 3. existing Stage 5N.3 stripped-line path remains accepted (essential) ──
def test_existing_stripped_multiline_path_still_matches_when_raw_join_would_not():
    # Both lines have leading AND trailing whitespace. The model's span is
    # the fully-STRIPPED join -- this is NOT a literal substring of the raw
    # text at all (raw text has "line  \n  second", not "line\nsecond"), so
    # only the pre-existing Stage 5N.3 stripped-line matcher can accept it;
    # the new additive raw-substring fallback must never be reached/needed.
    original = "  first line  \n  second line  \n"
    index = g.build_line_index(original)
    span = "first line\nsecond line"  # fully stripped -- not a raw substring
    full_raw = original.replace("\r\n", "\n").replace("\r", "\n")
    assert span not in full_raw  # sanity: confirms this only works via stripping
    match = g.ground_original_span(span, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_EXACT
    assert match.line_refs == ("L1-L2",)


# ── 4. wrong line_ref still rejects an out-of-scope raw multiline match ────
def test_raw_multiline_substring_outside_declared_line_ref_is_rejected():
    original = "alpha\nxxx abc yyy\nzzz def www\n"  # L1='alpha', L2-L3 contain the real span
    index = g.build_line_index(original)
    span = "abc yyy\nzzz def"  # only exists spanning L2-L3
    match = g.ground_original_span(span, "L1", index)  # wrong scope: declares L1 only
    assert match.status == g.SPAN_MATCH_NOT_FOUND


# ── 5. ambiguous when the same raw multiline span occurs twice in scope ────
def test_raw_multiline_substring_ambiguous_when_repeated_in_scope():
    original = "abc  \ndef  \nabc  \ndef  \n"  # the same 2-line raw pattern repeated
    index = g.build_line_index(original)
    span = "abc  \ndef"
    match = g.ground_original_span(span, "L1-L4", index)
    assert match.status == g.SPAN_MATCH_AMBIGUOUS
    assert match.occurrence_count == 2
    assert match.line_refs == ("L1-L2", "L3-L4")


# ── 6. NFC-equivalent raw multiline (decomposed vs precomposed Unicode) ────
def test_raw_multiline_substring_nfc_equivalent_match():
    decomposed_e_acute = "é"  # combining acute accent -> NFC-normalizes to 'é'
    original = f"caf{decomposed_e_acute}  \nbar  \n"  # raw source stored in decomposed form
    index = g.build_line_index(original)
    span = "café  \nbar"  # model returns the precomposed form, partial last line
    match = g.ground_original_span(span, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_NFC_EQUIVALENT
    assert match.normalized is True
    assert match.line_refs == ("L1-L2",)


# ── 7. single-line behavior is completely unaffected ───────────────────────
def test_single_line_grounding_behavior_unchanged():
    original = "hello world  \nsecond line\n"
    index = g.build_line_index(original)
    exact = g.ground_original_span("hello world  ", "L1", index)  # exact raw single line
    assert exact.status == g.SPAN_MATCH_EXACT
    assert exact.line_refs == ("L1",)

    # single-line matching already tolerates a span that is a substring of
    # one line's raw text (no multi-line window/raw-substring logic is
    # involved here at all -- confirms that path is untouched)
    prefix_only = g.ground_original_span("hello world", "L1", index)
    assert prefix_only.status == g.SPAN_MATCH_EXACT
    assert prefix_only.line_refs == ("L1",)

    not_present = g.ground_original_span("nonexistent phrase", "L1", index)
    assert not_present.status == g.SPAN_MATCH_NOT_FOUND


# ── 8. no fuzzy/approximate normalization is introduced ────────────────────
def test_whitespace_collapsed_multiline_span_does_not_match():
    original = "abc  \ndef  \n"  # 2 trailing spaces on L1
    index = g.build_line_index(original)
    collapsed_span = "abc \ndef"  # only 1 space, not the true 2 -- must NOT be accepted
    match = g.ground_original_span(collapsed_span, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_NOT_FOUND


def test_punctuation_differing_multiline_span_does_not_match():
    original = "abc,  \ndef  \n"
    index = g.build_line_index(original)
    span_with_wrong_punct = "abc.  \ndef"  # comma -> period
    match = g.ground_original_span(span_with_wrong_punct, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_NOT_FOUND


def test_case_differing_multiline_span_does_not_match():
    original = "Abc  \ndef  \n"
    index = g.build_line_index(original)
    span_wrong_case = "abc  \ndef"  # lowercase 'a' vs raw 'A'
    match = g.ground_original_span(span_wrong_case, "L1-L2", index)
    assert match.status == g.SPAN_MATCH_NOT_FOUND


# ── MV++_1339 offline regression (read-only fixture; not hardcoded logic) ──
def test_mv_1339_source_and_translation_spans_now_ground_successfully():
    """Uses the EXISTING generated candidate purely as a regression fixture
    -- the production grounding rule added above is fully generic (no
    poem ID, language, or phrase is referenced in grounding.py); this test
    only verifies that generic rule against real, already-generated data.
    Read-only: never writes to the candidate/source files."""
    source_path = REPO_ROOT / "data" / "source_corpus" / "Tamil" / "MV++_1339.json"
    candidate_path = REPO_ROOT / "outputs" / "model_candidates" / "Tamil" / "MV++_1339_vertex_model_candidate.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    orig_index = g.build_line_index(source["original_poem"])
    trans_index = g.build_line_index(source["translated_poem"])
    mspan = candidate["annotation"]["stanzas"][0]["metaphor_spans"][0]

    match_original = g.ground_original_span(mspan["source_span_original"], mspan["line_ref"], orig_index)
    match_translation = g.ground_translation_span(mspan["source_span_translation"], trans_index)

    assert match_original.status == g.SPAN_MATCH_EXACT
    assert match_original.line_refs == ("L1-L2",)
    assert match_translation.status == g.SPAN_MATCH_EXACT
