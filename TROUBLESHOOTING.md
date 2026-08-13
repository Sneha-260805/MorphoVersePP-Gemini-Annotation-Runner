# Troubleshooting

## `ConfigError: Missing required Gemini configuration`

`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, or `VERTEX_GEMINI_MODEL` is
not set in your shell. Set them (see `TEAMMATE_SETUP.md`) and re-run
`python scripts/verify_environment.py` to confirm.

## `Google authentication available: NO`

Run `gcloud auth application-default login`. If it succeeds but generation
still fails with a permission error, you likely don't have IAM access to
the `morphoverse-pilot` project yet — ask the project supervisor.

## `LanguageProfileMissing`

The poem's language has no approved profile in
`morphoverse_gemini_pipeline/delivery/poem_annotator/annotation_language_profiles/`.
306 of the corpus's 1,570 poems, across 15 languages, are currently in this
state — see `BLOCKED_LANGUAGES.md`. This is intentional — do not add a
profile yourself without team/native-speaker review; flag it instead.

## "Why does my dry run show 1,564 poems, not 1,570 or 1,258?"

A bare `--dry-run` (no `--assignment`/`--language`/`--poem-id`) plans the
**full corpus minus the 6 pilot poems** (1,570 − 6 = 1,564) — it
deliberately includes blocked-language poems too, reporting them as
`BLOCKED` rather than hiding them, so you can see exactly what's excluded
and why. Your actual assignment CSV only ever contains the 6-language,
non-pilot subset (1,258 poems total across the team) — see
`ASSIGNMENT_GUIDE.md`.

## Corpus counts look wrong / don't match `DATASET_PROVENANCE.md`

Someone may have edited `data/source_corpus/`, `corpus/*.json`, or
`assignments/*.csv` by hand. Every one of those is a deterministic export of
`data/raw/Indian_poem_dataset.xlsx` — never hand-edit them. Regenerate with:

```powershell
python scripts/build_corpus_from_excel.py
python scripts/create_assignments.py --teammates <N>
```

`build_corpus_from_excel.py` refuses to write anything if its own
independent count doesn't come out to exactly 1,570 canonical poems — if it
stops with an error, do not proceed; flag it to the project supervisor
instead of working around it.

## `PilotPoemBlocked`

You're targeting one of the six pilot poems (`corpus_gemini_runner_v1_1.PILOT_POEM_IDS`).
These are excluded by default. If you genuinely need to regenerate one,
pass `--allow-pilot-regeneration` — but check with the project supervisor
first; this is not a normal part of the assignment workflow.

## A poem's run failed — what do I do?

Check `reports/failures/<poem_id>.json`. It carries one of these
classifications:

| Classification | Meaning |
|---|---|
| `PROVIDER_FAILURE` | The Vertex call itself failed (network, quota, auth). |
| `TRUNCATION` | The model hit the output-token budget mid-response. |
| `SCHEMA_FAILURE` | The response didn't match the expected JSON shape. |
| `COMPLETENESS_FAILURE` | A required field stayed missing after repair. |
| `GROUNDING_FAILURE` | A term/span isn't a verbatim substring of the source or translation. |
| `ROMANIZATION_FAILURE` | Inconsistent romanization across the candidate. |
| `SOURCE_DATA_FAILURE` | The source-only record itself is missing/malformed. |
| `PROFILE_FAILURE` | The language profile failed to load. |
| `REPAIR_EXHAUSTED` | Targeted repair ran its maximum rounds and paths are still unresolved. |
| `SECURITY_FAILURE` | Reserved for a detected safety/security issue. |
| `UNKNOWN` | Anything not covered above — inspect the raw local artifacts. |

A failed poem never touches or invalidates any other poem's checkpoint or
output — you can always safely re-run the rest of your assignment. To
retry only the failed ones (after understanding and, if needed, discussing
the cause):

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_1.csv --retry-failed-only --execute --acknowledge-billing
```

## Where are the raw provider request/response payloads?

`local_provider_runs/<poem_id>/<section>/attempt_NN/` — gitignored, never
committed. Useful for debugging a specific failure locally; do not paste
raw poem/translation text or model output into a shared channel without
checking whether that's appropriate for the language in question.

## "Refusing to overwrite an existing candidate"

The runner never overwrites a `MODEL_CANDIDATE` file that's already on
disk. If you genuinely need to regenerate a poem, delete (or move aside)
its file under `outputs/model_candidates/<language>/` and its checkpoint
under `checkpoints/` first — deliberately, not automatically.

## Tests fail on a fresh clone

Run `python -m pytest --collect-only -q` first — if collection itself
fails, it's almost always a missing dependency (`pip install -r
requirements.txt`) or a wrong working directory (run pytest from the repo
root, where `pytest.ini` lives).
