# Doha — Week 1

Artifact data and English knowledge handoff for M1-DA.

This submission uses the 67 active artifact records already present in the project. It keeps their IDs, descriptions and source links, with one record per ID. The JSON files use UTF-8 and can be read without installing software dependencies.

| Week 1 deliverable | File |
| --- | --- |
| Artifact dataset | `artifact_records.json` |
| English knowledge text | `knowledge_en.json` |
| Sources and provenance | `sources.json` |
| Missing fields and correction notes | `data_notes.json` |
| Field definitions | `data_dictionary.md` |
| Machine-readable RAG input example | `handoff_example.json` |

The dataset and knowledge file are exports of existing project records. Underscore placeholders were converted to empty strings and each change is recorded in `data_notes.json`. There are 57 usable English descriptions; ten records need a sourced description before RAG can answer questions about them. Packaging checked structure and stable IDs; it did not independently verify historical claims. No unverified values were filled in.

The knowledge file contains source-linked English context for the RAG owner. It does not contain a RAG service. Images, model weights, computer-vision processing and pilot evaluation files belong to the separate CV work and are excluded from this text handoff.

Start with `artifact_records.json`, check the issues in `data_notes.json`, and use `handoff_example.json` as the input shape for an individual artifact.
