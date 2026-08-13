# Teammate 3 instructions

Your assignment: `assignments/teammate_3.csv` — **521 poems** across
20 of the 20 authorized languages. No Sanskrit, no pilot poem
appears in this file (verified by `tests/test_stage5m2_team_sync.py`).

## 1. Setup (once)

See `TEAMMATE_SETUP.md` for full detail. Summary:

```powershell
git clone <REPOSITORY_URL>
cd MorphoVersePP_Gemini_Annotation_Runner
git checkout -b teammate-3/gemini-generation

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

gcloud auth application-default login

$env:GOOGLE_CLOUD_PROJECT="morphoverse-pilot"
$env:GOOGLE_CLOUD_LOCATION="global"
$env:VERTEX_GEMINI_MODEL="gemini-3.5-flash"
```

## 2. Verify before touching your assignment

```powershell
python -m pytest -q
python scripts/preflight.py
```

`preflight.py` should report `Assignment 3: 521` among the three
assignment counts, and `Provider calls during preflight: 0`. Do not proceed
if any test fails.

## 3. Dry run your assignment (zero provider calls)

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_3.csv `
  --dry-run
```

Expect exactly 521 `[DRY RUN] ... 0 provider calls.` lines, no `BLOCKED`
lines (this assignment file contains no blocked poem).

## 4. Smoke test — exactly ONE real poem (billable)

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_3.csv `
  --max-poems 1 `
  --execute `
  --acknowledge-billing
```

Then check the generated candidate passed the stop gate: open the newly
written file under `outputs/model_candidates/Assamese/` (the
first assignment row's language) and confirm `stop_gate_passed: true`,
`candidate_status: "MODEL_CANDIDATE"`, and `unresolved_paths: []`. Also
check `checkpoints/MV++_0003.json` exists.

If the smoke-test poem does **not** pass the stop gate, stop and read
`TROUBLESHOOTING.md` / `reports/failures/MV++_0003.json` before
continuing — do not just re-run blindly.

## 5. Run the rest of your assignment

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_3.csv `
  --resume `
  --execute `
  --acknowledge-billing
```

`--resume` never regenerates a poem that already has a checkpoint (including
the one from step 4), so this is safe to re-run if interrupted.

## 6. If something failed

Retry only the failed poems, once you understand the cause
(`reports/failures/<poem_id>.json`):

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_3.csv `
  --retry-failed-only `
  --execute `
  --acknowledge-billing
```

See `TROUBLESHOOTING.md` for the full failure-classification table.

## 7. Expected output root

```text
outputs/model_candidates/<language>/<poem_id>_vertex_model_candidate.json
checkpoints/<poem_id>.json
```

One file per poem — your branch will merge cleanly against the other two
teammates' work since no two of you ever touch the same poem_id (or the same
output file).

## 8. What to commit

- validated `MODEL_CANDIDATE` JSON files you generated, under
  `outputs/model_candidates/`;
- a sanitized batch summary, if you produce one under `reports/` (never a
  raw provider request/response).

## 9. What to never commit

- `.env`, any credential, ADC file, service-account JSON, access/refresh
  token;
- anything under `local_provider_runs/` (raw provider payloads — already
  gitignored);
- `__pycache__/`, `.pytest_cache/`, other local cache/log files.

`git status` before every commit; if anything credential- or
provider-artifact-shaped is staged, stop and ask before committing.

## Do not

- pass `--allow-pilot-regeneration` without checking with the project
  supervisor first;
- target Sanskrit (`--language Sanskrit` / `--poem-id MV++_1235`) — it is
  programmatically blocked in this release, see `SANSKRIT_BLOCK.md`;
- hand-edit `corpus/execution_release_manifest.json`,
  `corpus/source_manifest.json`, or your assignment CSV.
