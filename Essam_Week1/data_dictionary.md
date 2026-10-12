# Field definitions

## Artifact records

| Field | Meaning |
| --- | --- |
| `id` | Existing stable project ID. Unique within this file. |
| `name` | Artifact or exhibit title from the project catalogue. |
| `civilization` | Existing catalogue civilization wording. |
| `category` | Existing object category. |
| `period` | Period description from the catalogue. |
| `dynasty` | Existing dynasty description; an empty string means not recorded. |
| `date` | Existing date wording; may be descriptive rather than a normalized date. An empty string means not recorded. |
| `material` | Existing material description, including any uncertainty. |
| `description` | Existing English description. Presence does not prove independent verification. |
| `objectNumber` | Source accession/object identifier when recorded. This is separate from the project `id`. |
| `source` | Existing source institution or provider name. |
| `sourceUrl` | Existing HTTP(S) source link. Some links document a scan or replica rather than a historical object record. |
| `hall` | Existing project display grouping. |

Blank values are preserved. Underscore placeholders are normalized to empty strings, with the original values retained in the correction notes. Missing data must not be inferred from a similar object. The export excludes image and model fields because those assets are not part of this text handoff.

## English knowledge

`artifact_id` joins to `artifact_records.json` by `id`. `title` repeats the record name, `language` is `en`, `text` copies the cleaned existing description, and `source_url` copies its source link. Empty text means the description is missing and must not be indexed as evidence. `review_status` records the limit of the available verification evidence.

## Sources and notes

`sources.json` records the input filename and SHA-256 hash, the extraction procedure, and the original source link for each ID. `data_notes.json` lists empty fields and unresolved review points. `corrections_applied_in_this_export` records each placeholder normalization and its original value; it does not indicate that new historical facts were added.

## Handoff example

`handoff_example.json` shows one existing artifact's ID, title, English context, source and review status. It is a data input example, not a generated answer or a retrieval test result.
