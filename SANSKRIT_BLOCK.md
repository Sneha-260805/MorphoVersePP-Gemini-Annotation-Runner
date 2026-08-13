# Sanskrit block

## Summary

The corpus contains exactly one Sanskrit poem — `MV++_1235`, the Bhagavad
Gita. It is **not authorized for team generation** in this release. This is
enforced programmatically, not just documented: every normal execution path
(`--assignment`, `--resume`, `--language Sanskrit`, `--poem-id MV++_1235`,
or a bare full-corpus run) refuses it. There is no override flag.

## Why

- Gemini's generated `source_span_original` for one metaphor span returned
  the pre-sandhi (grammatically decomposed, un-fused) form of a Sanskrit
  word, rather than the word's actual sandhi-fused written form as it
  appears on the poem's line.
- Exact source grounding correctly rejects this: the pre-sandhi form is not
  a verbatim substring of the poem's actual text, so it fails the same
  exact-match grounding rule every other language's spans must also pass.
  This is the grounding rule working as intended, not a bug to route
  around.
- This happened twice, independently: once in the original candidate, and
  again after a generic profile revision (a new `linguistic_guidance` rule
  telling the model to always preserve sandhi-fused written form, with no
  poem-specific content) was added and a fresh candidate was generated
  under the revised profile. Both attempts returned the identical
  pre-sandhi text at the identical field path.
- A purely generic prompt/profile intervention was insufficient to change
  the model's fresh-generation behavior at this specific sandhi boundary.

## What this means for team generation

- Sanskrit's poem is excluded from every assignment CSV.
- `corpus/execution_release_manifest.json` marks Sanskrit's language status
  `BLOCKED_ENGINEERING_REVIEW` and `MV++_1235` specifically
  `BLOCKED_PROFILE_REVISION_INSUFFICIENT`.
- The runner consults that manifest before generating anything, for every
  entry point — this is a fail-closed check: if the manifest file were ever
  missing, every language would be refused, not just Sanskrit.
- Sanskrit's profile file (`annotation_language_profiles/sanskrit.json`,
  version `5N.1`) **does** exist on disk — its presence does not by itself
  authorize generation. Runtime release status is the only thing that does.

## What this does not mean

- This is not a claim that Sanskrit is permanently unsupportable — only
  that a generic, non-poem-specific engineering fix did not resolve it.
- Do not attempt to bypass this block, hand-edit the release manifest, or
  invent a poem-specific prompt patch to force the pre-sandhi answer to
  "pass." Per this project's process, a repair round is never given the
  expected/corrected answer, and no test in this repository force-corrects
  the outcome to make totals agree.

## Next step

A native-Sanskrit human reviewer, or a scoped, still-generic experiment
(e.g. testing whether an explicit worked example of sandhi-fused vs.
pre-sandhi spans changes model behavior), is the recommended path — to be
decided by the project supervisor. This is out of scope for this
repository, which is a team execution runner, not a research environment.
