"""Review the current catalog and export field-level evidence for the text team.

Usage:
    python review_metadata.py --person2-root /path/to/MuseAI-Person2-CV --output /path/to/review

Only Python's standard library is required. The catalog and museum metadata
cache are read without changing them. A cache match is not a new human review
or a check of the museum's live website. Unmatched descriptions remain excluded
from approved_facts.json even when the catalog provides a plausible source URL.

An optional --review-decisions JSON file can record real human review. Its
format is {"source_fingerprint": "...", "records": [{"artifact_id": "...",
"reviewer": "...", "reviewed_at": "2026-10-09", "fields": {"description":
{"value": "exact current catalog value", "source_url": "https://...",
"quote": "source passage containing that value"}}}]}. No reviewer is inferred.
Decisions must refer to the current fingerprint and current field value.

Shared helpers: Essam_Week2/week2_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

from __future__ import annotations
import argparse
import csv
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Essam_Week2"
if not (_COMMON_FOLDER / "week2_common.py").is_file():
    raise FileNotFoundError(
        "Missing Essam_Week2/week2_common.py. Extract the complete Week 2 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week2_common import file_sha256, load_catalog, load_mapping, resolve_id, safe_output, source_fingerprint, write_json


FIELDS = {
    "title": "name", "period": "period", "dynasty": "dynasty", "date": "date",
    "material": "material", "description": "description",
    "object_number": "objectNumber", "civilization": "civilization",
}
CACHE_FIELDS = {
    "title": "name", "period": "period", "dynasty": "dynasty", "date": "date",
    "material": "material", "description": "description",
    "object_number": "accession", "civilization": "culture",
}
OFFICIAL_DOMAINS = {
    "metmuseum.org", "clevelandart.org", "britishmuseum.org", "smb.museum",
    "egyptianmuseumcairo.eg", "visit-gem.com", "gem.eg", "egymonuments.gov.eg",
    "louvre.fr", "museoegizio.it", "bonn.de", "uni-bonn.de",
}


def normalized(value):
    """Normalize spacing/case only; don't erase dates, punctuation or meaning."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def has_content(value):
    return normalized(value) not in {"", "_", "n/a", "none", "null", "unknown", "nan"}


def source_key(url):
    parsed = urlparse(str(url or ""))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname.lower().removeprefix("www.")
    return urlunparse(("https", host, parsed.path.rstrip("/"), "", "", ""))


def source_quality(url):
    key = source_key(url)
    if not key:
        return "missing_or_invalid_url"
    host = urlparse(key).hostname
    if any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_DOMAINS):
        return "recognized_institution_domain_not_content_verification"
    return "source_requires_review"


def museum_cache(root):
    """Index independently cached museum records by their museum page URL."""
    root = Path(root)
    by_url = defaultdict(list)
    provenance = []
    db_path = root / "knowledge/met-egypt.sqlite"
    manifest_path = db_path.with_suffix(".source.json")
    if db_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        provenance.append({
            "cache_file": "knowledge/met-egypt.sqlite",
            "cache_sha256": file_sha256(db_path),
            "provenance_file": "knowledge/met-egypt.source.json",
            "provenance_sha256": file_sha256(manifest_path),
            "source_manifest": manifest,
        })
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            for (raw,) in db.execute("SELECT record FROM objects"):
                record = json.loads(raw)
                key = source_key(record.get("source_url"))
                if key:
                    by_url[key].append((record, provenance[-1]))
    # These are raw institutional API responses, not the project descriptions.
    api_path = root / "dataset/provenance/source-records.json"
    if api_path.exists():
        provenance.append({
            "cache_file": "dataset/provenance/source-records.json",
            "cache_sha256": file_sha256(api_path),
            "source_manifest": {"scope": "Cached Metropolitan Museum API responses"},
        })
        source_records = json.loads(api_path.read_text(encoding="utf-8-sig"))
        for name, raw in source_records.items():
            if not name.startswith("met-extension/") or not raw.get("objectID"):
                continue
            url = raw.get("objectURL", "")
            key = source_key(url)
            if not key.startswith("https://metmuseum.org/art/collection/search/"):
                continue
            record = {
                "name": raw.get("title"), "period": raw.get("period"),
                "dynasty": raw.get("dynasty"), "date": raw.get("objectDate"),
                "material": raw.get("medium"), "accession": raw.get("accessionNumber"),
                "culture": raw.get("culture"), "source_url": url,
            }
            evidence = dict(provenance[-1], cache_record=name)
            by_url[key].append((record, evidence))
    return by_url, provenance


def read_decisions(path, fingerprint, catalog, mapping):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if data.get("source_fingerprint") != fingerprint:
        raise ValueError("Human review decisions belong to a different source fingerprint")
    records = {row["id"]: row for row in catalog}
    result = {}
    for entry in data.get("records", []):
        artifact_id = resolve_id(entry["artifact_id"], mapping)
        if artifact_id not in records or artifact_id in result:
            raise ValueError("Unknown or duplicate artifact ID in review decisions")
        if not str(entry.get("reviewer", "")).strip():
            raise ValueError("A human review decision must identify its actual reviewer")
        date.fromisoformat(entry.get("reviewed_at", ""))
        fields = entry.get("fields", {})
        if not isinstance(fields, dict) or not fields:
            raise ValueError("Human review decisions must contain reviewed fields")
        for field, decision in fields.items():
            if field not in FIELDS:
                raise ValueError("Unknown reviewed field: " + field)
            value = records[artifact_id].get(FIELDS[field], "")
            if not has_content(value) or normalized(value) != normalized(decision.get("value")):
                raise ValueError("Human review value differs from the current catalog: " + artifact_id)
            if not source_key(decision.get("source_url")):
                raise ValueError("Human review evidence needs a valid source URL")
            if normalized(value) not in normalized(decision.get("quote")):
                raise ValueError("Human review evidence must quote the approved value")
        result[artifact_id] = entry
    return result


def identity_audit(root, catalog, mapping):
    ids = {row["id"] for row in catalog}
    references = json.loads((root / "model/references.json").read_text(encoding="utf-8-sig"))
    with (root / "dataset/rag/combined_rag_chunks.csv").open(encoding="utf-8-sig", newline="") as stream:
        chunks = list(csv.DictReader(stream))
    reference_ids = {resolve_id(row["artifact_id"], mapping) for row in references}
    chunk_ids = [row["Chunk ID"] for row in chunks]
    rag_ids = {resolve_id(row["Artifact ID"], mapping) for row in chunks}
    orphans = {
        "mapping_targets": sorted({resolve_id(key, mapping) for key in mapping} - ids),
        "visual_reference_ids": sorted(reference_ids - ids),
        "rag_artifact_ids": sorted(rag_ids - ids),
    }
    duplicate_chunks = [key for key, count in Counter(chunk_ids).items() if count > 1]
    blank_chunks = [row["Chunk ID"] for row in chunks if not has_content(row.get("Chunk Text"))]
    declared_visual_ids = {row["id"] for row in catalog if row.get("has_visual_reference")}
    report = {
        "canonical_artifacts": len(ids), "legacy_aliases": len(mapping),
        "visual_artifacts": len(reference_ids), "reference_images": len(references),
        "metadata_only_artifacts": len(ids - reference_ids), "rag_chunks": len(chunks),
        "rag_artifacts": len(rag_ids), "orphan_ids": orphans,
        "artifacts_without_chunks": sorted(ids - rag_ids),
        "duplicate_chunk_ids": duplicate_chunks, "blank_chunk_ids": blank_chunks,
        "visual_flag_disagreements": sorted(reference_ids ^ declared_visual_ids),
    }
    report["consistent"] = not any(orphans.values()) and not any(
        report[key] for key in ["artifacts_without_chunks", "duplicate_chunk_ids", "blank_chunk_ids", "visual_flag_disagreements"]
    )
    return report


def review(person2_root, output, review_decisions=None):
    root = Path(person2_root).resolve()
    output = safe_output(root, output)
    catalog = load_catalog(root)
    mapping = load_mapping(root)
    fingerprint = source_fingerprint(root)
    cache, cache_provenance = museum_cache(root)
    decisions = read_decisions(review_decisions, fingerprint, catalog, mapping)
    reviewed, approved, issues, sources = [], [], [], []
    for record in catalog:
        artifact_id = record["id"]
        url = record.get("sourceUrl", "")
        url_candidates = cache.get(source_key(url), [])
        candidates = [(raw, origin) for raw, origin in url_candidates
                      if not has_content(record.get("objectNumber"))
                      or normalized(record["objectNumber"]) == normalized(raw.get("accession"))]
        facts, evidence, field_status = {}, [], {}
        row_issues = []
        if url_candidates and not candidates:
            row_issues.append({
                "field": "object_number", "issue": "source_identity_mismatch",
                "catalog_value": record.get("objectNumber"),
                "cached_source_values": [raw.get("accession", "") for raw, _ in url_candidates],
            })
        for field, catalog_field in FIELDS.items():
            value = record.get(catalog_field, "")
            if not has_content(value):
                field_status[field] = "missing"
                row_issues.append({"field": field, "issue": "missing_content"})
                continue
            if "\ufffd" in str(value):
                row_issues.append({"field": field, "issue": "replacement_character_requires_correction"})
            matches = [(raw, origin) for raw, origin in candidates
                       if has_content(raw.get(CACHE_FIELDS[field]))
                       and normalized(raw[CACHE_FIELDS[field]]) == normalized(value)]
            if matches and "\ufffd" not in str(value):
                raw, origin = matches[0]
                facts[field] = value
                field_status[field] = "matched_cached_museum_record"
                evidence.append({
                    "fields": [field], "source_url": raw["source_url"],
                    "method": "exact_normalized_cached_museum_match",
                    "cache_file": origin["cache_file"],
                    "cache_sha256": origin["cache_sha256"],
                    "cache_record": origin.get("cache_record", raw.get("museum_id", "")),
                    "source_value": raw[CACHE_FIELDS[field]],
                })
            else:
                source_values = list(dict.fromkeys(str(raw[CACHE_FIELDS[field]]) for raw, _ in candidates
                                                 if has_content(raw.get(CACHE_FIELDS[field]))))
                field_status[field] = "mismatch" if source_values else "needs_source_review"
                row_issues.append({
                    "field": field, "issue": field_status[field],
                    "catalog_value": value, "cached_source_values": source_values,
                })
        decision = decisions.get(artifact_id)
        if decision:
            for field, item in decision["fields"].items():
                facts[field] = record[FIELDS[field]]
                field_status[field] = "human_reviewed_with_quote"
                evidence.append({
                    "fields": [field], "source_url": item["source_url"],
                    "method": "explicit_human_review", "source_quote": item["quote"],
                    "reviewer": decision["reviewer"], "reviewed_at": decision["reviewed_at"],
                    "review_file_sha256": file_sha256(review_decisions),
                })
            row_issues = [item for item in row_issues if item["field"] not in decision["fields"]]
        content_fields = {field for field, key in FIELDS.items() if has_content(record.get(key))}
        status = ("source_matched" if content_fields and content_fields <= facts.keys()
                  else "partially_source_matched" if facts else "needs_review")
        shared = {"artifact_id": artifact_id, "title": record.get("name", ""),
                  "source_url": url, "review_status": status}
        approved.append(dict(shared, facts=facts, evidence=evidence))
        reviewed.append(dict(shared, **{field: record.get(key, "") for field, key in FIELDS.items() if field != "title"},
                             category=record.get("category", ""),
                             field_status=field_status, has_visual_reference=record.get("has_visual_reference", False)))
        quality = source_quality(url)
        if quality != "recognized_institution_domain_not_content_verification":
            row_issues.append({"field": "source_url", "issue": quality, "catalog_value": url})
        if not has_content(record.get("license")):
            row_issues.append({"field": "license", "issue": "catalog_license_missing_check_individual_sources"})
        issues.extend(dict(item, artifact_id=artifact_id) for item in row_issues)
        sources.append({
            "artifact_id": artifact_id, "source_name": record.get("source", ""),
            "source_url": url, "source_quality": quality, "catalog_license": record.get("license", ""),
            "cached_record_matches": len(candidates), "approved_fields": sorted(facts),
            "evidence": evidence,
        })
    audit = identity_audit(root, catalog, mapping)
    common = {"source_fingerprint": fingerprint,
              "review_method": "Field-level comparison with cached institutional metadata; unverified fields excluded from facts.",
              "limitations": ["Cache matches are not a live source check or a new human sign-off.",
                              "A source URL and project-derived RAG text do not independently verify a fact.",
                              "Project categories are retrieval labels and are not included as approved historical facts."]}
    for filename, records in [("reviewed_metadata.json", reviewed), ("approved_facts.json", approved),
                              ("sources_and_provenance.json", sources), ("missing_content_and_corrections.json", issues)]:
        write_json(output / filename, dict(common, records=records))
    write_json(output / "id_consistency.json", dict(common, **audit))
    summary = {
        "source_fingerprint": fingerprint, "artifacts": len(catalog),
        "artifacts_with_approved_facts": sum(bool(row["facts"]) for row in approved),
        "approved_fact_count": sum(len(row["facts"]) for row in approved),
        "approved_fields": dict(Counter(field for row in approved for field in row["facts"])),
        "review_status": dict(Counter(row["review_status"] for row in approved)),
        "issue_counts": dict(Counter(row["issue"] for row in issues)),
        "id_consistency": audit["consistent"], "cached_sources": cache_provenance,
        "human_review_records": len(decisions), "source_was_modified": False,
    }
    write_json(output / "review_summary.json", summary)
    if source_fingerprint(root) != fingerprint:
        raise RuntimeError("Source data changed during review; rerun before handing off outputs")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--person2-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--review-decisions", type=Path,
                        help="Optional actual human decisions with quoted source evidence")
    args = parser.parse_args()
    summary = review(args.person2_root, args.output, args.review_decisions)
    print(json.dumps({key: summary[key] for key in ["artifacts", "artifacts_with_approved_facts", "approved_fact_count", "id_consistency"]}))


if __name__ == "__main__":
    main()
