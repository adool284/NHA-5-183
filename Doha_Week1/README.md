# Essam — Week 1

Dataset audit and preparation for the Week 2 computer vision baseline.

This handoff uses the project's existing shared data preparation work from `person1-cv-data` v0.2. The folder name identifies the Week 1 submission responsibility; it does not change the authorship or recorded review history of the shared source files.

## What to review

| Week 1 requirement | Evidence |
| --- | --- |
| Artifact IDs and usable image mapping | `catalog.json`, `manifests/combined_reference.json` |
| Supported identities and isolated unsupported classes | `supported_ids.json`, `unsupported_classes.json`, `manifests/reference_only.json`, `class_coverage.csv` |
| Missing, corrupt, duplicate and similar image checks | `prepare.py`, `data_audit.json`, `manifests/duplicates.json`, `manifests/near_duplicates.json` |
| Quarantined and rejected images | `manifests/review_queue.json`, `manifests/rejected.json`, `review/decisions.json` |
| Split status and leakage checks | `split_manifest.json`, `manifests/train.json`, `manifests/validation.json`, `manifests/test.json`, `SPLIT_AND_HANDOFF.md`, `tests/test_integrity.py` |
| Image sources, rights and checksums | Manifest rows, `download_results.json`, `independent_results.json`, `catalog_references.json`, `licenses/` |
| Preprocessing and Week 2 handoff | `preprocessing.json`, `SPLIT_AND_HANDOFF.md` |

The 390 image files are required input data: 370 references, 6 validation images, 9 pilot test images, 3 quarantined images and 2 rejected images. The last five remain only to reproduce the review audit. Use the approved manifests; do not train on every file in the images folder.

The attribution notes in `licenses/` are preserved from the source project. References in those notes to 3D models or upstream source attachments describe the original project; this handoff contains the images and attribution notes only.

## Reproduce

Use Python 3.11 or newer with Pillow 11.3 installed:

```text
python -m pip install -r requirements.txt
python prepare.py
python audit_report.py
python -m unittest discover -s tests
```

After the dependency is installed, these commands use only the supplied files. `prepare.py` validates image paths and checksums, applies the existing review ledger and rebuilds the manifests. `audit_report.py` produces the data audit, coverage table and supported-ID list. `collect.py` contains only the local helper functions required by the preparation script.

There are 67 reference identities. For this handoff, the 365 photographic reference images are assigned to `manifests/train.json`; the five scan renders remain reference-only. This gives photographic training coverage for 62 identities. The original reference manifest is also retained to preserve the source assignment. No model training is included or claimed.

Validation and test cover very few identities and were previously inspected during development. They remain pilot data. The supplied train/validation/test assignment is reproducible and passes the implemented hash and capture-group overlap checks, but a fresh final evaluation set is still needed.
