# Assignment guide

The full canonical corpus is 1,570 poems (`corpus/corpus_inventory.json`).
Of those, 1,264 are in a currently-supported language and 6 of those are
already-generated pilots, leaving **1,258 poems currently assignable**
across Bengali, Hindi, Kannada, Kashmiri, Sindhi, and Telugu. The remaining
306 poems (15 languages) are not assignable yet — `create_assignments.py`
excludes them automatically and will start including them the moment their
language profile is approved and `FULL_CORPUS_READINESS.json` reflects it,
with no other change required.

## How the corpus is split

Run:

```powershell
python scripts/create_assignments.py --teammates 3
```

This reads `corpus/source_manifest.json` and writes one CSV per teammate
under `assignments/`. The split:

1. **Excludes any language without an approved profile.** See
   `SUPPORTED_LANGUAGES.md` / `BLOCKED_LANGUAGES.md` /
   `corpus/language_profile_coverage.json`. This script never invents a
   profile or substitutes a generic one for a missing language.
2. **Excludes the six pilot poems** (already generated in the pilot stage —
   see `TEAM_RUNBOOK.md`).
3. Optionally excludes poems that already have a `MODEL_CANDIDATE` output
   file (`--exclude-completed`) — useful when re-splitting remaining work
   partway through.
4. Splits **deterministically**: the same corpus state always produces the
   same split, so re-running the script is safe.
5. Is **language-balanced**: poems are grouped by language, then dealt
   round-robin across teammates so no one person gets stuck with only the
   largest or smallest language group.
6. Guarantees **no poem ID appears in more than one teammate's file** — this
   is enforced by an assertion in the script and independently re-checked at
   runtime by every teammate's own CSV load (`corpus_gemini_runner_v1_1.load_assignment_csv`
   rejects a CSV with an internal duplicate poem_id).

## Assignment CSV format

```csv
poem_id,language,assignee
MV++_0061,Hindi,teammate_1
MV++_0084,Hindi,teammate_1
MV++_0009,Bengali,teammate_1
```

See `assignments/example_assignment.csv` for a static example, and
`assignments/README.md` for more detail.

## Re-splitting mid-project

If the team's headcount changes or a language profile becomes newly
approved, just re-run `create_assignments.py` with the current teammate
count. Poems already completed (with a `MODEL_CANDIDATE` file on disk) can
be excluded with `--exclude-completed` so the new split only covers
remaining work.
