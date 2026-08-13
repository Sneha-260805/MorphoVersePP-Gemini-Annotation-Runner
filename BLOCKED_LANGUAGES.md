# Blocked languages

Languages marked anything other than `AUTHORIZED_FOR_TEAM_GENERATION` in `corpus/execution_release_manifest.json`. As of Stage 5M.2 this is Sanskrit only. Per the project's research-safety requirement, this repository does **not** invent, auto-translate, or silently substitute a generic profile for a missing one — and, separately, does not treat a *present* profile file as authorization by itself (Sanskrit's profile file, version 5N.1, exists on disk; it is still blocked). See `SANSKRIT_BLOCK.md` for the full reason. Counts below are computed from the authoritative 1,570-poem corpus.

| Language | Blocked poem count | Poem IDs | Reason |
|---|---|---|---|
| Sanskrit | 1 | MV++_1235 | Sole Sanskrit poem (MV++_1235) fails source grounding on a pre-sandhi lexical substitution defect; a generic Stage 5N.5 profile revision did not resolve it. See SANSKRIT_BLOCK.md. Not authorized for team generation. |

**Total blocked poems: 1** across 1 language(s).
