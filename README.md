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
- regenerate the six existing pilot poems by default.

Those are later pipeline stages, out of scope here.

## Runtime lifecycle

```text
source-only poem record
        ↓
language detection / language field
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
- `corpus/` — corpus inventory, language inventory, and language-profile
  coverage reports.
- `assignments/` — per-teammate assignment CSVs.
- `outputs/model_candidates/<language>/` — generated `MODEL_CANDIDATE` JSON,
  one file per poem.
- `checkpoints/` — one JSON checkpoint per completed poem (resume support).
- `reports/failures/` — sanitized failure records, classified by cause.
- `scripts/` — `preflight.py`, `verify_environment.py`,
  `create_assignments.py`.
- `tests/` — offline test suite (fake Vertex client; zero network access).

See `TEAMMATE_SETUP.md` to get started, `TEAM_RUNBOOK.md` for the day-to-day
workflow, `ASSIGNMENT_GUIDE.md` for how the corpus is split, and
`TROUBLESHOOTING.md` for common failure classes.

## The corpus

**Authoritative corpus:** the supplied `data/raw/Indian_poem_dataset.xlsx`
workbook (SHA-256-verified — see `DATASET_PROVENANCE.md`). This is the
*only* source of truth for what poems exist; nothing else in this repo or
in the development repo (in particular, the old `output_v3` legacy-annotation
directory) is treated as authoritative corpus data.

```text
Canonical poems:              1,570  (MV++_0001 .. MV++_1570)
Languages:                    21
Existing active language profiles: 6  (Bengali, Hindi, Kannada, Kashmiri, Sindhi, Telugu)
Missing profiles:             15
Pilot poems (already generated): 6  (excluded from default assignments)
```

**This repository is technically ready to generate annotations for the six
supported languages** (1,258 new poems, beyond the 6 already-generated
pilots). **It is not ready to generate the full 1,570-poem corpus** — 15
languages (306 poems) have no approved, reusable language addendum yet, and
this runner never invents or substitutes one. See
`FULL_CORPUS_READINESS.json`, `SUPPORTED_LANGUAGES.md`, and
`BLOCKED_LANGUAGES.md` for the exact, programmatically-derived numbers, and
`DATASET_PROVENANCE.md` for how the corpus was extracted from the workbook.
