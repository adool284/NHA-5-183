"""Summarize the prepared shared dataset without training or loading a model."""
import csv
from collections import Counter
from collect import ROOT, save
from prepare import checked, read, validate_splits


def main():
    catalog = read("catalog.json")
    source = read("download_results.json") + read("independent_results.json")
    failures = [row for row in source if "error" in row]
    inputs = [row for row in source if "error" not in row] + read("catalog_references.json")
    for row in inputs:
        checked(row)
    groups = {split: read(f"manifests/{name}.json") for split, name in (
        ("reference", "combined_reference"), ("validation", "validation"), ("test", "test"))}
    validate_splits(groups)
    counts = {split: Counter(row["artifact_id"] for row in rows if row.get("artifact_id"))
              for split, rows in groups.items()}
    refs = groups["reference"]
    train = [{**row, "source_split": "reference", "split": "train",
              "evaluation_use": "training_assigned_for_week1_handoff",
              "assignment_note": "Assigned for the Week 1 handoff from the reviewed photographic reference pool; no model training is implied."}
             for row in refs if row.get("kind") != "scan_render"]
    reference_only = [row for row in refs if row.get("kind") == "scan_render"]
    validate_splits({"train": train, "validation": groups["validation"], "test": groups["test"]})
    save(ROOT / "manifests/train.json", train)
    save(ROOT / "manifests/reference_only.json", reference_only)
    save(ROOT / "split_manifest.json", {
        "status": "reproducible_pilot_split_with_limited_coverage",
        "train_validation_test_manifests_available": True,
        "final_evaluation_ready": False,
        "train": {"manifest": "manifests/train.json", "images": len(train),
                  "known_identities": len({row["artifact_id"] for row in train}),
                  "status": "Photographic reference rows assigned to training for this handoff; no training run is claimed."},
        "validation": {"manifest": "manifests/validation.json", "images": len(groups["validation"]),
                       "status": "Previously exposed pilot data."},
        "test": {"manifest": "manifests/test.json", "images": len(groups["test"]),
                 "status": "Previously exposed pilot data, not a fresh final test."},
        "reference": {"manifest": "manifests/combined_reference.json", "images": len(refs),
                      "status": "Original retrieval reference pool preserved for provenance."},
        "reference_only": {"manifest": "manifests/reference_only.json", "images": len(reference_only),
                           "status": "Scan renders excluded from the photographic training split."},
        "reproduce": ["python prepare.py", "python audit_report.py"],
        "assignment_source": "Existing hash-bound review/decisions.json for reference and pilot data; audit_report.py deterministically assigns approved non-render reference rows to train.",
        "leakage_checks": "SHA-256, original-file hash, decoded-pixel hash and conservative capture group must not cross existing splits.",
        "independence_limit": "No observed hash/group overlap does not establish independent capture or a fresh final evaluation set."
    })
    source_ids = {row["artifact_id"] for row in read("manifests/source_gallery.json")}
    by_id = {item["id"]: [row for row in refs if row["artifact_id"] == item["id"]]
             for item in catalog}
    render_only = sorted(identity for identity, rows in by_id.items()
                         if rows and all(row.get("kind") == "scan_render" for row in rows))
    supported = sorted(identity for identity, rows in by_id.items() if rows)
    unsupported = sorted(identity for identity, rows in by_id.items() if not rows)
    training_ids = {row["artifact_id"] for row in train}
    save(ROOT / "supported_ids.json", {
        "support_definition": "At least one approved or existing catalog reference; not a claim of recognition accuracy.",
        "supported_reference_ids": supported,
        "active_ids_without_usable_references": unsupported,
        "supported_photographic_training_ids": sorted(training_ids),
        "active_ids_without_photographic_training_images": sorted(set(by_id) - training_ids),
        "render_only_reference_ids": render_only,
        "ids_without_additional_collected_source_photographs": sorted(set(by_id) - source_ids),
        "ids_without_validation_known_examples": sorted(set(by_id) - set(counts["validation"])),
        "ids_without_pilot_test_known_examples": sorted(set(by_id) - set(counts["test"]))
    })
    save(ROOT / "unsupported_classes.json", {
        "scope": "Photographic training coverage for this Week 1 handoff.",
        "classes": [{"artifact_id": identity,
                     "reason": "Only a scan render is available; no photographic training image."
                               if identity in render_only else "No usable reference image is available.",
                     "included_in_train": False,
                     "isolated_image_manifest": "manifests/reference_only.json"}
                    for identity in sorted(set(by_id) - training_ids)],
        "unknown_controls": "Rows with artifact_id=null in validation/test are deliberate unknown controls, not supported artifact classes."
    })
    with (ROOT / "class_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "artifact_id", "name", "category", "reference_images", "validation_known_images",
            "pilot_test_known_images", "photographic_training_images", "reference_status"])
        writer.writeheader()
        for item in sorted(catalog, key=lambda value: value["id"]):
            identity = item["id"]
            writer.writerow({
                "artifact_id": identity, "name": item.get("name", ""),
                "category": item.get("category", ""), "reference_images": counts["reference"][identity],
                "validation_known_images": counts["validation"][identity],
                "pilot_test_known_images": counts["test"][identity],
                "photographic_training_images": sum(row["artifact_id"] == identity for row in train),
                "reference_status": "missing" if identity in unsupported else
                    "render_only" if identity in render_only else "source_photo_available"})
    save(ROOT / "data_audit.json", {
        "source": "Existing shared person1-cv-data v0.2; repackaged for Week 1 dataset review",
        "summary": read("manifests/summary.json"),
        "checked_input_image_files": len(inputs),
        "missing_corrupt_or_checksum_failures": 0,
        "validation_note": "Report is written only after every input passes path, file hash and image decode checks.",
        "recorded_download_failures": failures,
        "cross_split_hash_and_capture_overlap": "passed",
        "train_validation_test_hash_and_capture_overlap": "passed",
        "photographic_training_images": len(train),
        "photographic_training_identities": len(training_ids),
        "near_duplicate_candidates": read("manifests/near_duplicates.json"),
        "exact_duplicates_removed": read("manifests/duplicates.json"),
        "split_counts": {split: {
            "images": len(rows), "known_identities": len(counts[split]),
            "unknown_control_images": sum(row.get("artifact_id") is None for row in rows),
            "capture_groups": len({row["capture_group"] for row in rows})}
            for split, rows in groups.items()},
        "review_basis": "Source metadata and the existing assisted-review ledger; no new human sign-off.",
        "limitations": [
            "Similarity checks cannot establish independence of camera captures.",
            "Five identities have render-only reference images.",
            "Validation and test images are previously exposed pilot data.",
            "A fresh final evaluation set remains outstanding."]
    })
    print(f"Data audit written: {len(inputs)} images checked; {len(supported)} reference identities.")


if __name__ == "__main__":
    main()
