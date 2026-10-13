# Notion task: M1-DA. Owner: Doha.
"""Week 1 image manifests and quality audit; no model training or inference.

Install: python -m pip install Pillow
Run: python prepare_images.py --person2-root PERSON2 --output RUN/doha

Reads the current reference and published split manifests. Output image paths
are relative to Person 2, so images are not duplicated. Existing evaluation
partitions are exposed pilot data; they are preserved, not relabeled as unseen.

Shared helpers: Doha_Week1/week1_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageOps
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Doha_Week1"
if not (_COMMON_FOLDER / "week1_common.py").is_file():
    raise FileNotFoundError(
        "Missing Doha_Week1/week1_common.py. Extract the complete Week 1 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week1_common import fingerprint, load_source, output_path, read_json, resolve_id, sha256, write_json


ACCEPTED = {"approved", "existing_catalog_reference", "institutional_source_verified"}
MANIFESTS = {"reference": "open_reference.json", "validation": "validation.json",
             "test": "test.json", "tutankhamun_regression": "tutankhamun_test.json"}
PHOTO_KINDS = {"camera_photo", "museum_image", "source_photo"}


def difference_hash(image):
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


def similar_image_pairs(rows, max_distance=4):
    """Flag possible duplicates for review without merging similar artifacts."""
    pairs = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            distance = (int(left["audit_dhash"], 16) ^ int(right["audit_dhash"], 16)).bit_count()
            if distance <= max_distance:
                pairs.append({"image_ids": [left["image_id"], right["image_id"]],
                              "artifact_ids": [left["artifact_id"], right["artifact_id"]],
                              "hamming_distance": distance,
                              "crosses_partitions": left["partition"] != right["partition"],
                              "same_capture_group": left["capture_group"] == right["capture_group"],
                              "review_status": "candidate_only_not_confirmed_duplicate"})
    return pairs


def inspect_image(root, row):
    dataset = (root / "dataset").resolve()
    path = (dataset / "manifests" / row["image"]).resolve()
    if dataset not in path.parents:
        raise ValueError("Image path leaves the source dataset.")
    if sha256(path) != row["sha256"]:
        raise ValueError("Image checksum mismatch: " + row["image_id"])
    with Image.open(path) as original:
        if original.format not in {"JPEG", "PNG", "WEBP"} or getattr(original, "n_frames", 1) != 1:
            raise ValueError("Expected a single JPEG, PNG or WebP image.")
        image = ImageOps.exif_transpose(original).convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        if min(image.size) < 32 or image.width * image.height > 20_000_000:
            raise ValueError("Unsupported image dimensions: " + row["image_id"])
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if row.get("pixel_sha256") and pixel_hash != row["pixel_sha256"]:
            raise ValueError("Pixel checksum mismatch: " + row["image_id"])
        declared = (row.get("width"), row.get("height"))
        dimension_note = ("verified" if declared == image.size else "measured_missing_dimensions"
                          if None in declared else "corrected_manifest_dimensions")
        return {"image_path": path.relative_to(root).as_posix(), "width": image.width,
                "height": image.height, "source_dimensions": list(declared),
                "dimension_note": dimension_note, "verified_pixel_sha256": pixel_hash,
                "audit_dhash": difference_hash(image)}


def prepare(root, output):
    root = Path(root).resolve()
    output = output_path(root, output)
    catalog, mapping = load_source(root)
    ids = {row["id"] for row in catalog}
    version = fingerprint(root)
    partitions, isolated, issues = defaultdict(list), [], []
    seen_images, seen_hashes, capture_splits = set(), {}, {}
    manifests = {}
    for split, filename in MANIFESTS.items():
        manifest = root / "dataset/manifests" / filename
        manifests[filename] = sha256(manifest)
        for number, original in enumerate(read_json(manifest), 1):
            row = dict(original)
            if row["image_id"] in seen_images:
                raise ValueError("Repeated image ID: " + row["image_id"])
            seen_images.add(row["image_id"])
            aid = resolve_id(row["artifact_id"], mapping) if row["artifact_id"] is not None else None
            if aid is not None and aid not in ids:
                raise ValueError("Image has an unknown canonical artifact ID.")
            if split == "reference" and aid is None:
                raise ValueError("Training references require a known identity.")
            row.update(artifact_id=aid, **inspect_image(root, row))
            if number % 100 == 0:
                print(f"Checked {number} images in {filename}", flush=True)
            if row["dimension_note"] != "verified":
                issues.append({"image_id": row["image_id"], "issue": row["dimension_note"],
                               "source_dimensions": row["source_dimensions"], "measured_dimensions": [row["width"], row["height"]]})
            target = "train" if split == "reference" else split
            reason = None
            if row.get("review_status") not in ACCEPTED:
                reason = "Reference review is incomplete."
            elif row.get("kind") not in PHOTO_KINDS:
                reason = "Non-photographic reference excluded from classifier training and photographic evaluation."
            if reason:
                row["exclusion_reason"] = reason
                isolated.append(row)
                continue
            for field in ("sha256", "original_sha256", "verified_pixel_sha256"):
                value = row.get(field)
                if value:
                    previous = seen_hashes.get((field, value))
                    if previous:
                        raise ValueError(f"Duplicate image content: {previous} and {row['image_id']}.")
                    seen_hashes[field, value] = row["image_id"]
            group = row.get("capture_group")
            if not group:
                raise ValueError("Capture group required to check split leakage: " + row["image_id"])
            if group in capture_splits and capture_splits[group] != target:
                raise ValueError("A capture group crosses data partitions: " + group)
            capture_splits[group] = target
            row["partition"] = target
            partitions[target].append(row)
            if not row.get("license") or not row.get("source_url"):
                issues.append({"image_id": row["image_id"], "issue": "incomplete_attribution"})
    for split in ("train", "validation", "test", "tutankhamun_regression"):
        partitions[split].sort(key=lambda row: row["image_id"])
    train_counts = Counter(row["artifact_id"] for row in partitions["train"])
    supported = sorted(train_counts)
    all_visual = {row["artifact_id"] for rows in partitions.values() for row in rows if row["artifact_id"]}
    all_visual.update(row["artifact_id"] for row in isolated if row["artifact_id"])
    declared = {row["id"] for row in catalog if row.get("has_visual_reference")}
    if all_visual != declared:
        raise ValueError("Catalog visual coverage disagrees with the manifests.")
    coverage = []
    for aid in sorted(ids):
        counts = {split: sum(row["artifact_id"] == aid for row in rows) for split, rows in partitions.items()}
        groups = {row["capture_group"] for row in partitions["train"] if row["artifact_id"] == aid}
        coverage.append({"artifact_id": aid, **counts, "training_capture_groups": len(groups),
                         "training_supported": aid in train_counts, "generalization_evidence_limited": len(groups) < 2})
    common = {"source_fingerprint": version, "input_manifest_sha256": manifests}
    limitation = "Published validation/test/regression images have been exposed. Most classes have one reference or capture group; do not claim final generalization from these splits."
    write_json(output / "image_manifest.json", {**common, "path_base": "person2_root", "records": [row for rows in partitions.values() for row in rows] + isolated})
    write_json(output / "supported_ids.json", {**common, "artifact_ids": supported})
    write_json(output / "unsupported_classes.json", {**common, "metadata_only": sorted(ids - all_visual),
               "visual_without_training_photographs": sorted(all_visual - set(supported)), "isolated_images": isolated})
    write_json(output / "split_manifest.json", {**common, "policy": "Preserve published capture-group-separated partitions; all accepted reference photographs become train. No random re-splitting.",
               "limitation": limitation, **dict(partitions)})
    write_json(output / "class_coverage.json", {**common, "records": coverage})
    similar_pairs = similar_image_pairs([row for rows in partitions.values() for row in rows])
    summary = {**common, "canonical_artifacts": len(ids), "visual_artifacts": len(all_visual), "supported_training_classes": len(supported),
               "counts": {split: len(rows) for split, rows in partitions.items()}, "isolated_images": len(isolated),
               "single_training_image_classes": sum(n == 1 for n in train_counts.values()), "issues": issues, "limitation": limitation,
               "near_duplicate_review": {"method": "64-bit dHash with Hamming distance at most 4",
                   "candidate_pairs": similar_pairs,
                   "cross_partition_candidates": sum(pair["crosses_partitions"] for pair in similar_pairs),
                   "policy": "Review candidates against source provenance. Similar appearance alone does not prove duplication; no images are automatically removed or re-split."}}
    write_json(output / "data_quality.json", summary)
    write_json(output / "preprocessing.json", {**common, "orientation": "Apply EXIF orientation", "color": "RGB with transparency composited on white",
               "resize": "Preserve aspect ratio; white letterbox to 224 x 224", "normalization_mean": [0.485, 0.456, 0.406],
               "normalization_std": [0.229, 0.224, 0.225], "augmentation": "Week 2 training only; never augment validation or test images.",
               "week2_handoff": "Read image_path relative to Person 2. Train only on supported IDs and partition=train. Keep the published pilot splits separate."})
    if fingerprint(root) != version or any(sha256(root / "dataset/manifests" / name) != value for name, value in manifests.items()):
        raise RuntimeError("Source changed during preparation; rerun the audit.")
    return {key: summary[key] for key in ("canonical_artifacts", "visual_artifacts", "supported_training_classes", "counts", "isolated_images")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.person2_root, args.output))


if __name__ == "__main__":
    main()
