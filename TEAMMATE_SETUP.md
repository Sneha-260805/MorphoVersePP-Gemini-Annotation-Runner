# Teammate setup

You need: **Python 3.10+, Git, the Google Cloud SDK (`gcloud`), and access to
the `morphoverse-pilot` Google Cloud project.** You do **not** need Claude,
Claude Code, Claude Pro, or an OpenAI account.

The corpus is the authoritative 1,570-poem workbook
(`data/raw/Indian_poem_dataset.xlsx` — see `DATASET_PROVENANCE.md`), already
exported to `data/source_corpus/`. You do not need Excel, openpyxl, or
pandas to run anything below — the export is a one-time, offline step
someone else already ran. 20 of the 21 languages (1,563 new-generation
poems, beyond the 6 already-generated pilots) are engineering-authorized and
assignable; Sanskrit's single poem is blocked and refused programmatically
— see `SUPPORTED_LANGUAGES.md` / `BLOCKED_LANGUAGES.md` / `SANSKRIT_BLOCK.md`.

## Windows (PowerShell)

```powershell
git clone <REPOSITORY_URL>
cd MorphoVersePP_Gemini_Annotation_Runner

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

gcloud auth application-default login
```

Set safe, non-secret configuration for this session (or add them to a local
`.env` copied from `.env.example` and load it however your shell prefers —
`.env` is gitignored):

```powershell
$env:GOOGLE_CLOUD_PROJECT="morphoverse-pilot"
$env:GOOGLE_CLOUD_LOCATION="global"
$env:VERTEX_GEMINI_MODEL="gemini-3.5-flash"
```

Run the offline test suite:

```powershell
python -m pytest -q
```

Check your environment (no provider calls, no secret ever printed):

```powershell
python scripts/preflight.py
python scripts/verify_environment.py
```

Dry run your assignment (builds every prompt; makes **zero** provider
calls):

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_1.csv `
  --dry-run
```

One-poem smoke test (this one **does** call Vertex and is billable — see
`docs` in `TEAM_RUNBOOK.md` before running it):

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_1.csv `
  --max-poems 1 --execute --acknowledge-billing
```

Then, to inspect the result, open the written file under
`outputs/model_candidates/<language>/` and the checkpoint under
`checkpoints/`.

Run the rest of your assignment, resuming safely if interrupted:

```powershell
python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 `
  --assignment assignments/teammate_1.csv `
  --resume --execute --acknowledge-billing
```

## macOS / Linux (bash/zsh)

```bash
git clone <REPOSITORY_URL>
cd MorphoVersePP_Gemini_Annotation_Runner

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

gcloud auth application-default login

export GOOGLE_CLOUD_PROJECT=morphoverse-pilot
export GOOGLE_CLOUD_LOCATION=global
export VERTEX_GEMINI_MODEL=gemini-3.5-flash

python -m pytest -q
python scripts/preflight.py
python scripts/verify_environment.py

python -m morphoverse_gemini_pipeline.delivery.poem_annotator.corpus_gemini_runner_v1_1 \
  --assignment assignments/teammate_1.csv --dry-run
```

## Google Cloud access

Use **your own** Google identity — never a shared credential file, never
someone else's ADC. Configuration:

```text
Project:        morphoverse-pilot
Location:       global
Model:          gemini-3.5-flash
Authentication: Application Default Credentials (ADC) / IAM
```

Run `gcloud auth application-default login` once per machine; it opens a
browser login and stores ADC locally (never committed, never read into any
file this repository writes). You need IAM permission on the
`morphoverse-pilot` project for Vertex AI — ask the project supervisor if
`gcloud auth application-default login` succeeds but generation still fails
with a permission error.

This repository never puts a shared API key, access token, refresh token,
or service-account file anywhere in source control. `.env.example` lists
only safe, non-secret configuration names.
