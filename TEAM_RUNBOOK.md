# Team runbook

The canonical corpus is 1,570 poems across 21 languages, extracted from
`data/raw/Indian_poem_dataset.xlsx` (see `DATASET_PROVENANCE.md`). Only 6
languages (1,258 non-pilot poems) are currently assignable — the rest are
blocked pending profile approval. This is not a full-corpus release; treat
every assignment as scoped to the currently-supported languages only.

## Workflow

1. Clone the repo (see `TEAMMATE_SETUP.md`).
2. Create your own branch: `git checkout -b <you>/annotate-<language>`.
3. Authenticate locally (`gcloud auth application-default login`) — your own
   identity, never a shared credential.
4. Run the offline test suite: `python -m pytest -q`. Do not proceed if
   anything fails.
5. Run `--dry-run` over your assignment. This makes **zero** provider calls
   and confirms every prompt for your assigned poems builds correctly.
6. Generate exactly one assigned smoke-test poem
   (`--max-poems 1 --execute --acknowledge-billing`). Inspect its
   `outputs/model_candidates/<language>/<poem_id>_vertex_model_candidate.json`
   and confirm `stop_gate_passed`, `candidate_complete`, and
   `unresolved_paths` look reasonable before continuing.
7. Run the rest of your assignment: repeat the same command with `--resume`
   so completed poems are never regenerated.
8. Commit and push your branch, open a PR.

## What to commit

- validated `MODEL_CANDIDATE` JSON files under `outputs/model_candidates/`;
- a sanitized batch summary, if you generate one under `reports/` (it must
  never contain a raw provider request/response — see below);
- your assignment CSV, if the team has decided to check those in.

## What never to commit

- credentials of any kind, `.env`, or anything under `local_provider_runs/`
  (raw provider request/response payloads — always gitignored);
- local checkpoints, unless the team has explicitly decided otherwise for a
  specific reason (they're personal progress state, not shared data);
- cache files (`__pycache__/`, `.pytest_cache/`).

The `.gitignore` in this repo already blocks the credential-shaped and raw
provider-artifact paths above — if `git status` shows one of them staged,
stop and ask before committing.

## Why per-poem output files

Every poem's `MODEL_CANDIDATE` is its own file under
`outputs/model_candidates/<language>/`. This means two teammates working on
different poems (even in the same language) never touch the same file, so
branches merge cleanly with no manual conflict resolution.

## Billing and safety gates

Any command that would make a real Vertex call requires **both**
`--execute` and `--acknowledge-billing` together — a plain run with neither
flag (and without `--dry-run`) refuses to start. `--concurrency` defaults to
`1` and is capped; this runner will never silently launch a large,
uncontrolled parallel batch. If you want more than one poem in flight at
once, pass `--concurrency N` deliberately and start small.

## Pilot poems

The six pilot poems (`MV++_0011`, `MV++_0073`, `MV++_1118`, `MV++_1153`,
`MV++_1249`, `MV++_1443`) are excluded from every default assignment and
from `--language`/`--dry-run` runs unless you pass
`--allow-pilot-regeneration` explicitly. Do not pass that flag without
checking with the project supervisor first — those poems already have
generated candidates from the pilot stage.

## If a batch fails

See `TROUBLESHOOTING.md`. In short: a failed poem never touches or
invalidates any other poem's checkpoint or output. Re-run the same command
with `--retry-failed-only` once the underlying cause is understood; do not
loop retries blindly.
