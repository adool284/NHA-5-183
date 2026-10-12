"""Adel, Week 2: retrieval over Person 2's current catalog and reviewed answers.

Install: python -m pip install numpy torch "transformers>=4.40,<5" huggingface-hub sentencepiece faiss-cpu
Run ``python rag.py --help``. Use --allow-download for the first index build.
Keep the source dataset, reviewed fact pack and generated index outside this
code directory. Build with the required multilingual L12 encoder. Week 1 uses
different chunk boundaries, so rebuild this index from the Week 2 source rows.
A raw retrieval result is a candidate, not a verified
historical fact. Only Doha's reviewed fact pack can supply an answer.

Examples (replace paths):
  python rag.py build --person2-root PERSON2 --output RUN/index
  python rag.py ask --person2-root PERSON2 --index RUN/index \
    --reviewed-data RUN/doha/approved_facts.json --artifact-id GEM_TUT_MASK \
    --question "What is the mask made of?" --output RUN/mask.json
  python rag.py evaluate --person2-root PERSON2 --index RUN/index \
    --reviewed-data RUN/doha/approved_facts.json --output RUN/rag_evaluation.json

The evaluate command includes a documented English development set, reports
unique-artifact Recall@K, answer correctness, citations and fallback behavior.
Optional --queries FILE supplies independent retrieval and answer expectations.
These development examples are not a held-out benchmark or a prose-quality review.

The embedding policy explicitly truncates each existing source row to the
encoder's 128-token training limit. The full row, original length and truncation
flag are retained as lineage. No additional vector is silently substituted for
a long row. This keeps the 2,333-row release aligned with its source metadata.

Shared helpers: Essam_Week2/week2_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any
import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Essam_Week2"
if not (_COMMON_FOLDER / "week2_common.py").is_file():
    raise FileNotFoundError(
        "Missing Essam_Week2/week2_common.py. Extract the complete Week 2 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week2_common import load_catalog, load_mapping, resolve_id, source_fingerprint


MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
MAX_TOKENS = 128
DIMENSIONS = 384
FORMAT = 2
FALLBACK = "I do not have enough verified information to answer this question."
CHUNKS = "dataset/rag/combined_rag_chunks.csv"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_path(path: str | Path, person2_root: str | Path) -> Path:
    """Prevent generated files from overwriting source data or submitted code."""
    target = Path(path).resolve()
    for protected in (Path(person2_root).resolve(), Path(__file__).resolve().parent):
        if target == protected or protected in target.parents:
            raise ValueError("Choose an output location outside Person 2 and the code package")
    return target


class Encoder:
    def __init__(self, allow_download: bool = False):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        directory = snapshot_download(repo_id=MODEL, revision=REVISION,
                                      local_files_only=not allow_download,
                                      allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModel.from_pretrained(directory, use_safetensors=True, local_files_only=True).eval()
        if int(self.model.config.hidden_size) != DIMENSIONS:
            raise ValueError("Unexpected embedding dimension")

    def metadata(self, text: str) -> dict:
        complete = self.tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
        encoded = self.tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
        return {
            "original_token_count": len(complete), "encoded_token_count": len(encoded),
            "truncated_for_embedding": len(complete) > MAX_TOKENS,
            "embedding_text": self.tokenizer.decode(encoded, skip_special_tokens=True),
        }

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.empty((0, DIMENSIONS), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[start:start + batch_size], padding=True,
                                    truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1)
            batches.append(vectors.cpu().numpy().astype("float32"))
            completed = min(start + batch_size, len(texts))
            if len(texts) >= 100 and (completed % 128 == 0 or completed == len(texts)):
                print(f"Embedded {completed}/{len(texts)} source rows", flush=True)
        return np.ascontiguousarray(np.concatenate(batches), dtype=np.float32)


def load_rows(root: Path, encoder: Encoder) -> list[dict]:
    catalog = {record["id"]: record for record in load_catalog(root)}
    mapping = load_mapping(root)
    rows, seen = [], set()
    with (root / CHUNKS).open(encoding="utf-8-sig", newline="") as stream:
        for position, row in enumerate(csv.DictReader(stream)):
            chunk_id = row["Chunk ID"].strip()
            artifact_id = resolve_id(row["Artifact ID"].strip(), mapping)
            text = row["Chunk Text"].strip()
            if not chunk_id or chunk_id in seen or not text or artifact_id not in catalog:
                raise ValueError(f"Invalid or duplicated source chunk at row {position + 2}")
            seen.add(chunk_id)
            record = catalog[artifact_id]
            rows.append({"row": position, "chunk_id": chunk_id, "artifact_id": artifact_id,
                         "title": record.get("name", ""), "source_url": record.get("sourceUrl", ""),
                         "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "source_csv_row": position + 2, "source_file": CHUNKS,
                         "evidence_status": "unreviewed_candidate", **encoder.metadata(text)})
    if not rows:
        raise ValueError("No source chunks found")
    return rows


def build_index(root: str | Path, output: str | Path, encoder: Encoder) -> dict:
    root = Path(root).resolve()
    directory = output_path(output, root)
    directory.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root, encoder)
    vectors = encoder.encode([row["text"] for row in rows])
    if vectors.shape != (len(rows), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding output")
    index = faiss.IndexFlatIP(DIMENSIONS)
    index.add(vectors)
    faiss.write_index(index, str(directory / "index.faiss"))
    save_json(directory / "rows.json", rows)
    config = {
        "format": FORMAT, "source_fingerprint": source_fingerprint(root),
        "source_chunks_sha256": sha256(root / CHUNKS),
        "embedding_model": MODEL, "embedding_revision": REVISION,
        "dimensions": DIMENSIONS, "dtype": "float32", "normalized": True,
        "metric": "INNER_PRODUCT", "pooling": "attention_mask_mean",
        "max_tokens": MAX_TOKENS, "long_row_policy": "explicit_truncation_with_lineage",
        "number_of_chunks": len(rows), "number_of_artifacts": len({r["artifact_id"] for r in rows}),
        "truncated_rows": sum(r["truncated_for_embedding"] for r in rows),
        "rows_sha256": sha256(directory / "rows.json"),
        "index_sha256": sha256(directory / "index.faiss"),
    }
    # The commit marker is written last. Incomplete or mixed bundles fail loading.
    save_json(directory / "index_config.json", config)
    return config


class RAG:
    def __init__(self, person2_root: str | Path, index_dir: str | Path,
                 encoder: Encoder, reviewed_data: str | Path | None = None):
        self.root = Path(person2_root).resolve()
        self.catalog = {r["id"]: r for r in load_catalog(self.root)}
        self.mapping = load_mapping(self.root)
        self.encoder = encoder
        directory = Path(index_dir)
        self.config = read_json(directory / "index_config.json")
        expected = {"format": FORMAT, "embedding_model": MODEL, "embedding_revision": REVISION,
                    "dimensions": DIMENSIONS, "dtype": "float32", "normalized": True,
                    "metric": "INNER_PRODUCT", "pooling": "attention_mask_mean",
                    "max_tokens": MAX_TOKENS, "long_row_policy": "explicit_truncation_with_lineage"}
        if any(self.config.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible encoder/index configuration; rebuild the Week 2 index")
        if self.config.get("source_fingerprint") != source_fingerprint(self.root):
            raise ValueError("Person 2 data changed; rebuild the index and reviewed fact pack")
        for name, key in (("rows.json", "rows_sha256"), ("index.faiss", "index_sha256")):
            if sha256(directory / name) != self.config.get(key):
                raise ValueError(f"Index bundle hash mismatch: {name}")
        if sha256(self.root / CHUNKS) != self.config["source_chunks_sha256"]:
            raise ValueError("Source chunks changed; rebuild the index")
        self.rows = read_json(directory / "rows.json")
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if (self.index.ntotal != len(self.rows) or self.index.ntotal != self.config["number_of_chunks"]
                or self.index.d != DIMENSIONS or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise ValueError("Index and row metadata are not aligned")
        self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Index must contain normalized finite vectors")
        self.by_id: dict[str, list[int]] = {}
        for position, row in enumerate(self.rows):
            if row["row"] != position or row["artifact_id"] not in self.catalog:
                raise ValueError("Invalid row lineage")
            self.by_id.setdefault(row["artifact_id"], []).append(position)
        self.reviewed: dict[str, dict] = {}
        self.reviewed_hash = None
        if reviewed_data is not None:
            payload = read_json(reviewed_data)
            if payload.get("source_fingerprint") != self.config["source_fingerprint"]:
                raise ValueError("Reviewed facts belong to another source release")
            checked_cache = {}
            for record in payload.get("records", []):
                artifact_id = record["artifact_id"]
                if artifact_id not in self.catalog or artifact_id in self.reviewed:
                    raise ValueError("Unknown or duplicate canonical ID in reviewed fact pack")
                for item in record.get("evidence", []):
                    if isinstance(item, dict) and item.get("method") == "explicit_human_review":
                        continue
                    if not isinstance(item, dict) or not item.get("cache_file"):
                        raise ValueError("Reviewed fact evidence must identify its cached source")
                    cache = (self.root / item["cache_file"]).resolve()
                    if self.root not in cache.parents:
                        raise ValueError("Reviewed evidence cache must belong to Person 2")
                    if cache not in checked_cache:
                        checked_cache[cache] = sha256(cache)
                    if item.get("cache_sha256") != checked_cache[cache]:
                        raise ValueError("A reviewed source cache changed; run Doha's review again")
                self.reviewed[artifact_id] = record
            self.reviewed_hash = sha256(reviewed_data)

    def canonical_id(self, artifact_id: str) -> str:
        return resolve_id(artifact_id.strip(), self.mapping)

    def retrieve(self, question: str, artifact_id: str | None = None, top_k: int = 5) -> list[dict]:
        if not question.strip() or len(question) > 1000:
            raise ValueError("Question must contain between 1 and 1,000 characters")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        query = self.encoder.encode([question])
        if artifact_id is None:
            scores, positions = self.index.search(query, min(top_k, len(self.rows)))
            matches = zip(positions[0].tolist(), scores[0].tolist())
        else:
            selected = self.by_id.get(self.canonical_id(artifact_id), [])
            scores = self.vectors[selected] @ query[0]
            order = np.argsort(-scores, kind="stable")[:top_k]
            matches = ((selected[i], float(scores[i])) for i in order)
        return [{**self.rows[position], "similarity": round(float(score), 6)}
                for position, score in matches if position >= 0]

    def ask(self, artifact_id: str, question: str, top_k: int = 5) -> dict:
        canonical = self.canonical_id(artifact_id)
        hits = self.retrieve(question, canonical, top_k)
        answer = {"schema_version": 2, "requested_artifact_id": artifact_id, "artifact_id": canonical,
                  "alias_resolved": canonical != artifact_id, "question": question,
                  "status": "insufficient_evidence", "answer": FALLBACK, "sources": [],
                  "retrieved_chunks": hits, "source_fingerprint": self.config["source_fingerprint"],
                  "reviewed_facts_sha256": self.reviewed_hash, "answer_method": "reviewed_fact_extraction"}
        if canonical not in self.catalog:
            answer["reason"] = "Unknown artifact ID."
            return answer
        if not hits:
            answer["reason"] = "No indexed text for this artifact."
            return answer
        record = self.reviewed.get(canonical)
        if not record or record.get("review_status") not in {"source_matched", "partially_source_matched"}:
            answer["reason"] = "No verified fact record has been supplied for this artifact."
            return answer
        field, reason = requested_field(question, self.catalog[canonical].get("name", ""))
        if field is None:
            answer["reason"] = reason
            return answer
        facts = record.get("facts", {})
        value = facts.get(field)
        if not isinstance(value, str) or not value.strip():
            answer["reason"] = f"The reviewed source does not supply the requested {field} fact."
            return answer
        catalog_field = {"title": "name", "object_number": "objectNumber"}.get(field, field)
        if normalize(value) != normalize(str(self.catalog[canonical].get(catalog_field, ""))):
            answer["reason"] = "The reviewed fact differs from the current catalog value."
            return answer
        evidence = field_evidence(record, field)
        if not evidence:
            answer["reason"] = f"The {field} field has no source verification evidence."
            return answer
        # Prevent a fact from a reviewed file being attached to an unrelated retrieval.
        raw = " ".join(hit["text"] for hit in hits)
        if normalize(value) not in normalize(raw):
            answer["reason"] = "The verified fact is not present in the retrieved source rows."
            return answer
        text = render_fact(field, value)
        answer.update(status="answered", answer=text + " [1]", fact_field=field, fact_value=value,
                      sources=[{"label": "[1]", "artifact_id": canonical,
                                "title": record.get("title", self.catalog[canonical].get("name", "")),
                                "url": evidence["source_url"], "field": field,
                                "evidence": evidence, "fact_value": value,
                                "chunk_ids": [h["chunk_id"] for h in hits
                                              if normalize(value) in normalize(h["text"])]}],
                      reason="The requested field was retrieved and verified in the reviewed fact pack.")
        return answer


def normalize(value: str) -> str:
    """Normalize text without erasing uncertainty, ranges or punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def query_words(value: str) -> str:
    return " ".join(re.findall(r"\w+", normalize(value)))


def field_evidence(record: dict, field: str) -> Any:
    """A citation URL alone does not prove that an individual field was reviewed."""
    evidence = record.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if not (isinstance(item, dict) and field in item.get("fields", [])
                    and re.match(r"^https?://[^/\s]+", str(item.get("source_url", "")))):
                continue
            if (item.get("method") == "exact_normalized_cached_museum_match"
                    and re.fullmatch(r"[0-9a-f]{64}", str(item.get("cache_sha256", "")))
                    and item.get("cache_record") and item.get("cache_file")
                    and isinstance(item.get("source_value"), str)
                    and normalize(item["source_value"]) == normalize(record["facts"][field])):
                return item
            if item.get("method") == "explicit_human_review":
                try:
                    date.fromisoformat(item.get("reviewed_at", ""))
                except (TypeError, ValueError):
                    continue
                if (str(item.get("reviewer", "")).strip()
                        and re.fullmatch(r"[0-9a-f]{64}", str(item.get("review_file_sha256", "")))
                        and normalize(record["facts"][field]) in normalize(item.get("source_quote", ""))):
                    return item
    return None


def requested_field(question: str, title: str = "") -> tuple[str | None, str]:
    """Recognize direct, single-field questions about the selected object only.

    Whole-question matching prevents dates of restoration or theft, artists'
    preferences and display-case materials from becoming catalog fact answers.
    """
    q = query_words(question)
    references = {"it", "this", "this artifact", "the artifact", "this object",
                  "the object", "this item", "the item"}
    title_words = query_words(title)
    if title_words:
        references.update({title_words, "the " + title_words, "this " + title_words})
        for noun in ("mask", "throne", "statue", "stela", "papyrus", "coffin", "amulet", "vessel"):
            if noun in title_words.split():
                references.update({"the " + noun, "this " + noun})
    subject = "(?:" + "|".join(re.escape(term) for term in sorted(references, key=len, reverse=True)) + ")"
    patterns = (
        ("material", rf"what (?:material )?is {subject} made (?:of|from)"),
        ("material", rf"(?:what|which) materials? (?:is|was) {subject} made (?:of|from)"),
        ("material", r"what (?:is|are) its materials?"),
        ("date", rf"when was {subject} (?:made|created)"),
        ("date", rf"(?:what|which) date is {subject} from"),
        ("date", r"what is its (?:date|creation date)"),
        ("period", rf"(?:what|which) (?:period|era) is {subject} from"),
        ("period", r"what is its (?:period|era)"),
        ("dynasty", rf"(?:what|which) dynasty is {subject} from"),
        ("dynasty", r"what is its dynasty"),
        ("dimensions", r"what (?:is|are) its (?:dimensions|height|width|weight)"),
        ("title", rf"what is {subject} called"),
        ("description", rf"(?:tell me about|describe|what is|give me a description of) {subject}"),
    )
    for field, pattern in patterns:
        if re.fullmatch(pattern, q):
            return field, ""
    return None, "Ask one direct question about the selected object's material, date, period, dynasty, title or description. Other events and relationships need more evidence."


def render_fact(field: str, value: str) -> str:
    labels = {"material": "Material", "date": "Date", "period": "Period", "dynasty": "Dynasty",
              "dimensions": "Dimensions"}
    return f"{labels[field]}: {value}" if field in labels else value


DEVELOPMENT_QUERIES = {
    "retrieval": [
        {"question": "Golden burial mask of Tutankhamun", "expected_ids": ["GEM_TUT_MASK"]},
        {"question": "Golden throne of Tutankhamun", "expected_ids": ["GEM_TUT_THRONE"]},
        {"question": "Upper half of a woman from a pair or group statue 2021.41.2",
         "expected_ids": ["MET_2021.41.2"]},
        {"question": "Statuette of a female figure with a braid 2021.41.15",
         "expected_ids": ["MET_2021.41.15"]},
        {"question": "Papyrus sheet 14.1.393i", "expected_ids": ["MET_14.1.393i"]},
    ],
    "answers": [
        {"artifact_id": "EG_PA_001", "question": "What material is this artifact made of?",
         "expected_status": "answered", "expected_field": "material", "expected_value": "Papyrus"},
        {"artifact_id": "MET_14.1.393i", "question": "What period is the papyrus sheet from?",
         "expected_status": "answered", "expected_field": "period", "expected_value": "Late Period"},
        {"artifact_id": "EG_PA_001", "question": "When was this artifact made?",
         "expected_status": "answered", "expected_field": "date", "expected_value": "664\u2013332 B.C."},
        {"artifact_id": "MET_14.1.393i", "question": "Which dynasty is this from?",
         "expected_status": "answered", "expected_field": "dynasty", "expected_value": "Dynasty 26\u201330"},
        {"artifact_id": "GEM_TUT_MASK", "question": "What is the mask made of?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "GEM_TUT_THRONE", "question": "What is the throne made of?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "What is the current price?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "What is its weight?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "Who won the football match today?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "NO_SUCH_ARTIFACT", "question": "What is it made of?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "Why was this artifact created?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "When was this artifact restored?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "When was this artifact stolen?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "What material is the display case made of?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "EG_PA_001", "question": "What material did the artist prefer?",
         "expected_status": "insufficient_evidence"},
        {"artifact_id": "MET_2021.41.2", "question": "What is this artifact made of?",
         "expected_status": "answered", "expected_field": "material", "expected_value": "Granite (?)"},
    ],
}


def ratio(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


def evaluate(rag, queries, k=5):
    retrieval, answers = [], []
    for item in queries.get("retrieval", []):
        expected = set(item["expected_ids"])
        if not expected or not expected <= rag.catalog.keys():
            raise ValueError("Evaluation contains missing or unknown expected artifact IDs")
        # Retrieve all chunk rows so artifact Recall@K counts unique artifacts.
        vector = rag.encoder.encode([item["question"]])
        _, positions = rag.index.search(vector, rag.index.ntotal)
        ranked = list(dict.fromkeys(rag.rows[i]["artifact_id"] for i in positions[0]))[:k]
        found = expected.intersection(ranked)
        rank = next((position + 1 for position, artifact in enumerate(ranked) if artifact in expected), None)
        retrieval.append({**item, "retrieved_ids": ranked, "recall_at_k": len(found) / len(expected),
                          "reciprocal_rank_at_k": 1 / rank if rank else 0})
    for item in queries.get("answers", []):
        if item["expected_status"] not in {"answered", "insufficient_evidence"}:
            raise ValueError("Invalid expected answer status")
        result = rag.ask(item["artifact_id"], item["question"])
        supported = None
        if result["status"] == "answered":
            record = rag.reviewed.get(result["artifact_id"], {})
            field, value = result.get("fact_field"), result.get("fact_value")
            supported = bool(field and value and record.get("facts", {}).get(field) == value
                             and field_evidence(record, field) and result["sources"]
                             and all(source["artifact_id"] == result["artifact_id"]
                                     and source["fact_value"] == value for source in result["sources"]))
        correct = result["status"] == item["expected_status"]
        if item["expected_status"] == "answered":
            if "expected_field" not in item or "expected_value" not in item:
                raise ValueError("Answered expectations require independent field and value labels")
            correct = correct and result.get("fact_field") == item["expected_field"] and (
                normalize(result.get("fact_value", "")) == normalize(item["expected_value"]))
        answers.append({**item, "correct": correct, "factual_support": supported, "result": result})
    answered = [item for item in answers if item["result"]["status"] == "answered"]
    fallbacks = [item for item in answers if item["expected_status"] == "insufficient_evidence"]
    return {"scope": "Small English development set; not held-out or general accuracy",
            "source_fingerprint": rag.config["source_fingerprint"],
            "reviewed_facts_sha256": rag.reviewed_hash, "k": k,
            "retrieval_count": len(retrieval), "answer_count": len(answers),
            "recall_at_k": ratio(sum(item["recall_at_k"] for item in retrieval), len(retrieval)),
            "mrr_at_k": ratio(sum(item["reciprocal_rank_at_k"] for item in retrieval), len(retrieval)),
            "expected_answer_behavior_rate": ratio(sum(item["correct"] for item in answers), len(answers)),
            "factual_support_rate_among_answered": ratio(sum(item["factual_support"] for item in answered), len(answered)),
            "fallback_recall": ratio(sum(item["correct"] for item in fallbacks), len(fallbacks)),
            "answer_coverage": ratio(len(answered), len(answers)),
            "limitations": ["Evidence approval is based on Doha's cached museum source review.",
                            "Unverified descriptions and new objects can retrieve but cannot answer.",
                            "The answer method is reviewed fact extraction, not free-form generation.",
                            "128-token embedding truncation may lower long-row recall.",
                            "Queries were selected during development and are not a held-out test."],
            "retrieval": retrieval, "answers": answers}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = cli.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Rebuild all vectors from the updated source chunks")
    build.add_argument("--person2-root", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--allow-download", action="store_true")
    for name in ("ask", "retrieve"):
        command = commands.add_parser(name)
        command.add_argument("--person2-root", required=True)
        command.add_argument("--index", required=True)
        command.add_argument("--artifact-id", required=name == "ask")
        command.add_argument("--question", required=True)
        command.add_argument("--reviewed-data", required=name == "ask")
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--output", required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate retrieval, answers, citations and fallback")
    evaluation.add_argument("--person2-root", required=True)
    evaluation.add_argument("--index", required=True)
    evaluation.add_argument("--reviewed-data", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--queries", help="JSON object with retrieval and answers lists and independent expectations")
    evaluation.add_argument("--top-k", type=int, default=5)
    return cli


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if hasattr(args, "top_k") and not 1 <= args.top_k <= 50:
        cli.error("top-k must be between 1 and 50")
    encoder = Encoder(getattr(args, "allow_download", False))
    if args.command == "build":
        result = build_index(args.person2_root, args.output, encoder)
    else:
        rag = RAG(args.person2_root, args.index, encoder, args.reviewed_data)
        if args.command == "ask":
            result = rag.ask(args.artifact_id, args.question, args.top_k)
        elif args.command == "evaluate":
            queries = read_json(args.queries) if args.queries else DEVELOPMENT_QUERIES
            result = evaluate(rag, queries, args.top_k)
        else:
            result = {"evidence_status": "unreviewed_candidates", "results": rag.retrieve(
                args.question, args.artifact_id, args.top_k)}
        save_json(output_path(args.output, args.person2_root), result)
    print(json.dumps({"command": args.command, "output": str(args.output),
                      "status": result.get("status", "complete"),
                      "number_of_chunks": result.get("number_of_chunks")}, indent=2))


if __name__ == "__main__":
    main()
