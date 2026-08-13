# Supported languages

Languages marked `AUTHORIZED_FOR_TEAM_GENERATION` in `corpus/execution_release_manifest.json` — engineering canary passed (Stage 5N.1, or Stage 5N.3/5N.4 recovery). All 21 profile files exist on disk; profile *presence* is not what authorizes generation — the release manifest is. These are validated-for-team-generation, not yet native-speaker-approved profiles (`native_review_required=true` on every one) — see each profile JSON. Counts below are computed from the authoritative 1,570-poem corpus (`data/raw/Indian_poem_dataset.xlsx`) — see DATASET_PROVENANCE.md.

| Language | Poems in corpus | Profile file |
|---|---|---|
| Assamese | 7 | `annotation_language_profiles/assamese.json` |
| Bengali | 28 | `annotation_language_profiles/bengali.json` |
| Bodo | 9 | `annotation_language_profiles/bodo.json` |
| Dogri | 10 | `annotation_language_profiles/dogri.json` |
| Gujarati | 1 | `annotation_language_profiles/gujarati.json` |
| Hindi | 850 | `annotation_language_profiles/hindi.json` |
| Kannada | 245 | `annotation_language_profiles/kannada.json` |
| Kashmiri | 12 | `annotation_language_profiles/kashmiri.json` |
| Konkani | 11 | `annotation_language_profiles/konkani.json` |
| Malayalam | 10 | `annotation_language_profiles/malayalam.json` |
| Manipuri | 5 | `annotation_language_profiles/manipuri.json` |
| Marathi | 10 | `annotation_language_profiles/marathi.json` |
| Odia | 10 | `annotation_language_profiles/odia.json` |
| Punjabi | 19 | `annotation_language_profiles/punjabi.json` |
| Rajasthani | 7 | `annotation_language_profiles/rajasthani.json` |
| Santhali | 6 | `annotation_language_profiles/santhali.json` |
| Sindhi | 8 | `annotation_language_profiles/sindhi.json` |
| Tamil | 179 | `annotation_language_profiles/tamil.json` |
| Telugu | 121 | `annotation_language_profiles/telugu.json` |
| Urdu | 21 | `annotation_language_profiles/urdu.json` |

**Total supported poems: 1569** (including the 6 already-generated pilot poems — see `corpus/corpus_inventory.csv`'s `pilot_status` column). **New team generation target: 1563** (1569 supported − 6 pilot), split 521/521/521 across 3 teammates — see `assignments/`.
