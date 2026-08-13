# Assignments

Each CSV in this directory assigns a disjoint set of corpus poems to one
teammate. Format:

```csv
poem_id,language,assignee
MV++_0061,Hindi,teammate_1
```

`example_assignment.csv` is a static, checked-in example (not a real
assignment) showing the exact shape the runner expects.

## Current assignment (Stage 5M.2)

`teammate_1.csv` / `teammate_2.csv` / `teammate_3.csv` each contain exactly
**521** poems — **1,563** unique poem IDs combined, covering all 20
engineering-authorized languages. This is:

```text
1,570 total corpus poems
  -     1 blocked (Sanskrit, MV++_1235 — see SANSKRIT_BLOCK.md)
  -     6 already-generated pilots (Bengali/Hindi/Kannada/Kashmiri/Sindhi/Telugu)
  = 1,563 eligible poems, split 521 / 521 / 521
```

No teammate CSV contains Sanskrit or any of the six pilot poem IDs.

## Generating real assignments

```powershell
python scripts/create_assignments.py --teammates 3
```

writes `assignments/teammate_1.csv`, `assignments/teammate_2.csv`,
`assignments/teammate_3.csv`. The split:

- excludes any language not marked `AUTHORIZED_FOR_TEAM_GENERATION` in
  `corpus/execution_release_manifest.json` (currently: Sanskrit only) —
  a language profile file being present on disk does **not** by itself
  imply authorization; the release manifest is the sole authority;
- excludes any poem_id listed under that manifest's `blocked_poems`
  (currently: `MV++_1235`), regardless of its language's status;
- excludes the six already-generated pilot poems
  (`corpus_gemini_runner_v1_1.PILOT_POEM_IDS`) unless you pass
  `--exclude-completed` as well to also drop poems that already have a
  `MODEL_CANDIDATE` output file;
- is deterministic (same input → same split every time) and
  language-balanced;
- guarantees no poem ID appears in more than one teammate's file.

The committed CSVs in this directory **are** the current real assignment —
regenerating them (e.g. after a future release-manifest change) will
overwrite these files; commit the result if the team agrees the new split
should replace this one.
