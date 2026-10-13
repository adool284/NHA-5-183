"""Shared Week 1 helpers, packaged once with Doha's submission.

Extract the complete weekly archive to keep the person folders side by side.
Other scripts load this file from Doha_Week1; do not make per-person copies.
"""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(root):
    digest = hashlib.sha256()
    for name in ("dataset/combined_catalog.json", "dataset/id_mapping.json", "dataset/rag/combined_rag_chunks.csv"):
        digest.update(name.encode() + b"\0" + sha256(Path(root) / name).encode() + b"\n")
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_id(identifier, mapping):
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("A nonempty artifact ID is required.")
    visited = set()
    while identifier in mapping and mapping[identifier] != identifier:
        if identifier in visited:
            raise ValueError("Cycle in artifact ID aliases.")
        visited.add(identifier)
        identifier = mapping[identifier]
    return identifier


def load_source(root):
    root = Path(root).resolve()
    records = read_json(root / "dataset/combined_catalog.json")
    mapping = read_json(root / "dataset/id_mapping.json")
    if not isinstance(records, list) or not isinstance(mapping, dict):
        raise ValueError("Expected a catalog list and an alias dictionary.")
    ids = [row.get("id") for row in records]
    if any(not isinstance(aid, str) or not aid.strip() for aid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Catalog IDs must be nonempty and unique.")
    for key, value in mapping.items():
        if not isinstance(value, str) or resolve_id(key, mapping) not in ids:
            raise ValueError("Alias does not resolve to a canonical catalog ID.")
    return records, mapping


def output_path(root, output):
    target = Path(output).expanduser().resolve()
    for protected in (Path(root).expanduser().resolve(), Path(__file__).resolve().parent.parent):
        if target == protected or protected in target.parents:
            raise ValueError("Write generated results outside the source dataset and code folder.")
    return target


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
