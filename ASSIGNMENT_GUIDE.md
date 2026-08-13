# Assignment guide

The full canonical corpus is 1,570 poems (`corpus/corpus_inventory.json`).
Of those, 1,569 are in an engineering-authorized language
(`corpus/execution_release_manifest.json`) and 6 of those are
already-generated pilots, leaving **1,563 poems currently assignable**
across the 20 authorized languages, split evenly across 3 teammates
(521 each). The 1 remaining poem — `MV++_1235` / Sanskrit — is not
assignable — `create_assignments.py` excludes it automatically (see
`SANSKRIT_BLOCK.md`) and would start including it the moment the release
manifest marks Sanskrit `AUTHORIZED_FOR_TEAM_GENERATION`, with no other
change required.

## How the corpus is split

Run:

```powershell
python scripts/create_assignments.py --teammates 3
```

This reads `corpus/source_manifest.json` and writes one CSV per teammate
under `assignments/`. The split:

1. **Excludes any language not marked `AUTHORIZED_FOR_TEAM_GENERATION` in
   `corpus/execution_release_manifest.json`** (currently: Sanskrit only),
   and any individual poem_id listed under that manifest's `blocked_poems`
   regardless of its language's status. See `SUPPORTED_LANGUAGES.md` /
   `BLOCKED_LANGUAGES.md` / `SANSKRIT_BLOCK.md`. A language profile file
   being present on disk does **not** by itself imply authorization — all
   21 profile files exist, but the release manifest is the sole authority.
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
