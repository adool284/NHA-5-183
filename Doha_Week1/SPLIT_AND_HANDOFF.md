# Split policy and Week 2 handoff

The existing reference and pilot assignments are reproduced from image hashes and the decisions in `review/decisions.json`. These are fixed reviewed assignments, not a new random split. The original retrieval reference pool remains available in `combined_reference.json`.

For this Week 1 handoff, `audit_report.py` assigns the 365 non-render photographic reference rows to `manifests/train.json`. This is a new documented data role for the existing reviewed images, not evidence that a model was trained. `source_split` preserves each row's original reference role. The five scan renders are separated into `manifests/reference_only.json`; the corresponding five identities have no photographic training images. `supported_ids.json` lists both reference support and photographic training support.

The train, validation and pilot test manifests are checked for file-hash, decoded-pixel, original-source-hash and capture-group overlap. Their existing assignments remain fixed; no frames from a capture sequence are randomly divided across sets. `split_manifest.json` records the counts, policy and reproducibility commands. Passing these checks does not establish a fresh or representative final evaluation set.

An image must have a recognized artifact ID, matching checksum, recorded rights and a capture group before it enters a usable split. An unreviewed or changed image stays quarantined. Exact pixel duplicates are removed within the same identity and split. Conflicting labels, shared file hashes, original hashes, decoded-pixel hashes or capture groups across splits stop preparation. Similarity candidates use dHash; any candidate across splits also requires review before preparation can pass.

Photogrammetry sequences from one object share a conservative capture group. Several views from that sequence are not independent evaluation samples. Renders are reference-only. Five identities have render-only reference coverage, and six have no additional collected source photographs; the ID lists are in `supported_ids.json`.

The present reference pool has 370 images covering 67 identities. Validation has 6 images: 2 known images for one identity and 4 unknown controls. The pilot test has 9 images: 5 known images for two identities and 4 unknown controls. These evaluation images have already been exposed during development. No fresh final test set is available, and final-test readiness remains false.

For Week 2, use `manifests/train.json` for photographic training experiments and `manifests/combined_reference.json` when reproducing the existing retrieval reference pool. Respect the five identities lacking photographic training data. The existing validation data can support exploratory work, with all changes recorded. Report the existing test as a pilot result only. Fresh, independently photographed data covering more identities is needed for a final evaluation; do not rename the current pilot test as unseen data.

All manifest image paths resolve relative to the manifest file within this folder. Keep image attribution, rights and original checksums with the data when copying it into a training project. The source records retain the existing assisted-review labels; packaging them here does not add a human sign-off.

Only the offline data checks are included. Model weights, image embeddings, training results, recognition code, story generation and the application are outside this Week 1 handoff.
