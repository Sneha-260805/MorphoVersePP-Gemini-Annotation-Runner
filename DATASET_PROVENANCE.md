# Dataset provenance

## Source of truth

```text
File:   data/raw/Indian_poem_dataset.xlsx
SHA-256: d0dbc5c8c387d5d9bbc2c566f5dd4f9a5a665c7c72e96df5a9cf65c6314e0cd2
```

This workbook, supplied for Stage 5M.1, is the **authoritative** source for
the MorphoVerse++ corpus. It replaces the Stage 5M mistake of treating
`morphoverse_gemini_pipeline/delivery/output_v3/` (166 legacy Gemini
candidate records — a partial, already-annotated subset) as if it were the
complete corpus. `output_v3` is not copied into this repository and is used
only for historical cross-checks in the development repo.

The raw workbook is never modified and never read directly by the live
generation path — see "How derived artifacts are produced" below.

## Structure (independently audited, not assumed)

- Single worksheet (`Sheet1`), header row 1:
  `Language | Poem Title | Translated Title | Poet | Translator | Original Poems | Translated Poems`.
- **1,570 canonical poem rows** with a valid `Language` value (one of the 21
  names in `poem_annotator/config.py`'s `SUPPORTED_LANGUAGES`).
- **1 embedded duplicate header row** at worksheet row 1210, using a
  different label set (`Original Language | Original Title | English Title
  | Poet | Translator | Original Poem | English Poem`). Discovered
  programmatically — its `Language`-column value doesn't match any of the
  21 supported language names — not by hardcoding a row number. Excluded
  from the canonical corpus and from the exported source records.
- **4 trailing poem-like rows** immediately after the canonical boundary
  (worksheet rows 1573–1576), each missing a `Language` value. Classified
  `TRAILING_UNLABELED_EXTRA_RECORD`, preserved for audit only in
  `corpus/trailing_unlabeled_records.json` / `corpus/TRAILING_RECORD_AUDIT.md`.
  Never assigned an ID, never exported, never assignable.

## MV++ ID assignment rule

Poem IDs are assigned **by canonical worksheet position**, `MV++_0001`
through `MV++_1570`, skipping the top header row and the one embedded
header row — never alphabetically, never grouped by language. This is the
same rule the original pilot-era IDs already followed; it was verified
against all six pilot poems using **position plus source-content hash**,
not title matching (duplicate titles exist in the corpus — 110
`(language, title)` pairs occur more than once):

| Poem ID | Language | Title | Verified |
|---|---|---|---|
| MV++_0011 | Bengali | O Amar Desher Mati | ✓ |
| MV++_0073 | Hindi | Ek Chadar Maili Si | ✓ |
| MV++_1118 | Kannada | Udayavagali Namma Cheluva Kannada Naadu | ✓ |
| MV++_1153 | Kashmiri | Ghazal | ✓ |
| MV++_1249 | Sindhi | Sufi Kafis | ✓ |
| MV++_1443 | Telugu | Oh Jabilamma | ✓ |

## How derived artifacts are produced

```text
data/raw/Indian_poem_dataset.xlsx  (raw, unmodified, hashed)
        ↓  scripts/build_corpus_from_excel.py  (stdlib-only xlsx parse; offline; no network)
        ↓
data/source_corpus/<language>/MV++_XXXX.json   (5-field generation input)
corpus/corpus_metadata_manifest.json            (poet/translator/titles — NOT a generation input)
corpus/corpus_inventory.json / .csv
corpus/language_inventory.json
corpus/source_manifest.json
corpus/trailing_unlabeled_records.json / TRAILING_RECORD_AUDIT.md
```

`corpus_gemini_runner_v1_1.py` (the live generation path) never opens the
`.xlsx` file — it only ever reads the deterministic, already-exported JSON
under `data/source_corpus/`. Re-run
`python scripts/build_corpus_from_excel.py` only if the raw workbook is
intentionally updated; it fully regenerates every derived artifact above
(never incremental) and refuses to proceed (raises before writing anything)
if its own independent count doesn't come out to exactly 1,570.
