# Supported languages

Languages with a reusable, existing v1.1 language addendum (`profile_version=5K.1.0`, `profile_status=DRAFT_REQUIRES_LANGUAGE_REVIEW`, `native_review_required=true`). These are validated-for-pilot-generation, not yet native-speaker-approved profiles — see each profile JSON. Counts below are computed from the authoritative 1,570-poem corpus (`data/raw/Indian_poem_dataset.xlsx`) — see DATASET_PROVENANCE.md.

| Language | Poems in corpus | Profile file |
|---|---|---|
| Bengali | 28 | `annotation_language_profiles/bengali.json` |
| Hindi | 850 | `annotation_language_profiles/hindi.json` |
| Kannada | 245 | `annotation_language_profiles/kannada.json` |
| Kashmiri | 12 | `annotation_language_profiles/kashmiri.json` |
| Sindhi | 8 | `annotation_language_profiles/sindhi.json` |
| Telugu | 121 | `annotation_language_profiles/telugu.json` |

**Total supported poems: 1264** (including the 6 already-generated pilot poems — see `corpus/corpus_inventory.csv`'s `pilot_status` column). **New, currently-assignable generations: 1258** (1264 supported − 6 pilot).

