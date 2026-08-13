# Assignments

Each CSV in this directory assigns a disjoint set of corpus poems to one
teammate. Format:

```csv
poem_id,language,assignee
MV++_0061,Hindi,teammate_1
```

`example_assignment.csv` is a static, checked-in example (not a real
assignment) showing the exact shape the runner expects.

## Generating real assignments

```powershell
python scripts/create_assignments.py --teammates 3
```

writes `assignments/teammate_1.csv`, `assignments/teammate_2.csv`,
`assignments/teammate_3.csv`. The split:

- excludes any language without an approved profile (`BLOCKED_LANGUAGES.md`)
  — never invents or substitutes one;
- excludes the six already-generated pilot poems
  (`corpus_gemini_runner_v1_1.PILOT_POEM_IDS`) unless you pass
  `--exclude-completed` as well to also drop poems that already have a
  `MODEL_CANDIDATE` output file;
- is deterministic (same input → same split every time) and
  language-balanced;
- guarantees no poem ID appears in more than one teammate's file.

Real, generated assignment CSVs are not committed by this script — decide as
a team whether to commit them (they contain no secrets, only poem IDs) or
regenerate them locally as needed.
