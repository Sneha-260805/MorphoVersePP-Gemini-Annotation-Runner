# MorphoVerse++ Gemini Annotation Runner

## What this repository does

Generates MorphoVerse++ Schema v1.1 Gemini `MODEL_CANDIDATE` annotations for
the MorphoVerse poetry corpus, using Google Vertex AI (Gemini) directly —
**no Claude, no Claude Code, no Claude Pro, no OpenAI/GPT** is required to
run it. Anyone on the team can clone this repository and generate
annotations for their assigned poems with only Python, Git, and their own
Google Cloud identity.

## What it does not do

This repository does **not**:

- use Claude or the Anthropic SDK, at build time or at runtime;
- use GPT/OpenAI;
- build silver annotations (that requires cross-model agreement, not yet
  computed);
- build gold annotations (gold is human-only — see below);
- perform human review or adjudication;
- perform IndicBERT clustering or inter-model agreement;
- regenerate the six existing pilot poems by default;
- generate Sanskrit — see "Current release limitation" below.

Those are later pipeline stages (or, for Sanskrit, a blocked one), out of
scope here.

## Runtime lifecycle

```text
source-only poem record
        ↓
language detection / language field
        ↓
execution_release_manifest.json authorization check (refuses Sanskrit)
        ↓
matching reusable language addendum (annotation_language_profiles/*.json)
        ↓
shared Schema v1.1 annotation prompt
        ↓
five-section prompt construction
        ↓
Gemini through Vertex AI
        ↓
candidate assembly
        ↓
schema / completeness / grounding / romanization / cross-field validation
        ↓
targeted repair, only when objectively necessary
        ↓
full revalidation
        ↓
per-poem stop gate
        ↓
MODEL_CANDIDATE JSON
        ↓
checkpoint
```

## Important warning

Every annotation this runner produces remains:

```text
candidate_status = MODEL_CANDIDATE
review_status    = REVIEW_PENDING
native_review_required = true
not_silver        = true
not_gold          = true
not_human_approved = true
```

Nothing in this repository can produce `SILVER`, `GOLD`, `FINAL`, or
`APPROVED` output. Human review and adjudication happen in a later,
separate stage of the project.

## Where things live

- `morphoverse_gemini_pipeline/delivery/poem_annotator/` — the runtime
  package: schema, models, prompt assembly, language profiles, Vertex
  execution primitives, validators, targeted repair, and
  `corpus_gemini_runner_v1_1.py`, the poem-generic corpus orchestrator.
- `data/source_corpus/<language>/<poem_id>.json` — source-only poem records
  (`poem_id`, `poem_title`, `language`, `original_poem`, `translated_poem`
  only — no prior annotation content).
- `corpus/` — corpus inventory, language inventory, language-profile
  coverage, and `execution_release_manifest.json` (the sole authority on
  which language/poem is authorized for team generation).
- `assignments/` — per-teammate assignment CSVs (1,563 poems total, 521
  each, across the 3 teammates).
- `outputs/model_candidates/<language>/` — generated `MODEL_CANDIDATE` JSON,
  one file per poem.
- `checkpoints/` — one JSON checkpoint per completed poem (resume support).
- `reports/failures/` — sanitized failure records, classified by cause.
- `scripts/` — `preflight.py`, `verify_environment.py`,
  `create_assignments.py`, `build_corpus_from_excel.py`.
- `tests/` — offline test suite (fake Vertex client; zero network access).

See `TEAMMATE_SETUP.md` to get started, `TEAM_RUNBOOK.md` for the day-to-day
workflow, `ASSIGNMENT_GUIDE.md` for how the corpus is split,
`TROUBLESHOOTING.md` for common failure classes, and your own
`TEAMMATE_<N>_INSTRUCTIONS.md` for exact commands.

## The corpus

**Authoritative corpus:** the supplied `data/raw/Indian_poem_dataset.xlsx`
workbook (SHA-256-verified — see `DATASET_PROVENANCE.md`). This is the
*only* source of truth for what poems exist; nothing else in this repo or
in the development repo (in particular, the old `output_v3` legacy-annotation
directory) is treated as authoritative corpus data.

```text
Canonical poems:                    1,570  (MV++_0001 .. MV++_1570)
Languages:                          21
Engineering-authorized languages:   20
Already-generated pilots:           6   (excluded from default assignments)
New team generation target:         1,563
Blocked:                            1 poem — MV++_1235 / Sanskrit
```

## Current release limitation

```text
Supported for team generation:  20 languages / 1,569 corpus poems
Already generated pilots:       6 poems
New team generation target:     1,563 poems
Blocked:                        MV++_1235 / Sanskrit

Full corpus ready:              NO
Supported corpus ready:         YES
```

Sanskrit's sole corpus poem (`MV++_1235`, the Bhagavad Gita) fails source
grounding on a pre-sandhi lexical-substitution defect that a generic Stage
5N.5 profile revision did not resolve. It is refused **programmatically** —
not just documented — by every normal execution path
(`--assignment`/`--resume`/`--language`/`--poem-id`), with no override flag
in this release. See `SANSKRIT_BLOCK.md`.

See `FULL_CORPUS_READINESS.json`, `SUPPORTED_LANGUAGES.md`,
`BLOCKED_LANGUAGES.md`, and `corpus/execution_release_manifest.json` for the
exact, programmatically-derived numbers, and `DATASET_PROVENANCE.md` for how
the corpus was extracted from the workbook.
