# Notion task: M1-AE. Owner: Adel.
"""Week 1 English RAG on Essam's updated knowledge handoff.

Install: python -m pip install numpy torch "transformers>=4.40,<5" huggingface-hub sentencepiece faiss-cpu
Prepare Essam's knowledge_en.json first, then:
  python rag.py build --person2-root PERSON2 --knowledge RUN/essam/knowledge_en.json --index RUN/rag --allow-download
  python rag.py ask --person2-root PERSON2 --knowledge RUN/essam/knowledge_en.json --index RUN/rag --artifact-id EG_PA_001 --question "What material is this artifact made of?" --output RUN/answer.json
  python rag.py ask --person2-root PERSON2 --knowledge RUN/essam/knowledge_en.json --index RUN/rag --artifact-id EG_PA_001 --question "What is its current price?" --output RUN/fallback.json

The first build downloads the pinned encoder and generator when requested.
Later runs use their cache. Indexes and answers are generated outside the code
and Person 2 directories. No API key, website or classifier is part of Week 1.

This baseline creates its OWN token-bounded chunks from the prepared corpus.
It does not replace Person 2's original 2,333-row RAG release or reuse its old
vectors. Both the corpus and index retain source fingerprints and canonical IDs.
Answers quote supplied English catalog evidence; they are not a new historical
source review. This small model can abstain or misunderstand nuanced questions.

Shared helpers: Doha_Week1/week1_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
import faiss
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Doha_Week1"
if not (_COMMON_FOLDER / "week1_common.py").is_file():
    raise FileNotFoundError(
        "Missing Doha_Week1/week1_common.py. Extract the complete Week 1 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week1_common import fingerprint, load_source, output_path, read_json, resolve_id, sha256, write_json


def model_directory(model_id, revision, allow_download):
    """Resolve a pinned local snapshot before loading the tokenizer or weights."""
    return snapshot_download(repo_id=model_id, revision=revision,
                             local_files_only=not allow_download,
                             allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


MAX_TOKENS = 128


class Encoder:
    def __init__(self, allow_download=False):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        directory = model_directory(EMBEDDING_MODEL, EMBEDDING_REVISION, allow_download)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModel.from_pretrained(directory, use_safetensors=True, local_files_only=True).eval()

    def encode(self, texts):
        batches = []
        for start in range(0, len(texts), 16):
            inputs = self.tokenizer(texts[start:start + 16], padding=True, truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > MAX_TOKENS:
                raise ValueError("Embedding input exceeds 128 tokens; shorten the question or rebuild chunks.")
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                vectors = torch.nn.functional.normalize(pooled, dim=1)
            batches.append(vectors.cpu().numpy().astype(np.float32))
            if (start + 16) % 128 == 0 or start + 16 >= len(texts):
                print(f"Encoded {min(start + 16, len(texts))}/{len(texts)} passages", flush=True)
        return np.concatenate(batches) if batches else np.empty((0, 384), dtype=np.float32)


GENERATION_MODEL = "google/flan-t5-small"


GENERATION_REVISION = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"


class Generator:
    def __init__(self, allow_download=False):
        torch.set_num_threads(min(4, torch.get_num_threads()))
        directory = model_directory(GENERATION_MODEL, GENERATION_REVISION, allow_download)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(directory, use_safetensors=True, local_files_only=True).eval()

    def prompt(self, question, context):
        return ("Answer using only the context. Copy the answer from the context. "
                "If it is not in the context, say 'not enough information'.\n"
                f"Context: {context}\nQuestion: {question}\nAnswer:")

    def fits(self, question, context):
        return len(self.tokenizer(self.prompt(question, context))["input_ids"]) <= 512

    def answer(self, question, context):
        inputs = self.tokenizer(self.prompt(question, context), return_tensors="pt", truncation=False)
        if inputs["input_ids"].shape[1] > 512:
            raise ValueError("Question and context exceed the generator budget.")
        with torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        return self.tokenizer.decode(result[0], skip_special_tokens=True).strip()


FALLBACK = "I do not have enough information in the supplied sources to answer this question."
FIELD_LABELS = {"name": "Title", "period": "Period", "dynasty": "Dynasty", "date": "Date", "material": "Material",
                "objectNumber": "Museum accession", "civilization": "Civilization", "description": "Description"}


def load_knowledge(root, path):
    catalog, mapping = load_source(root)
    payload = read_json(path)
    if payload.get("source_fingerprint") != fingerprint(root):
        raise ValueError("Knowledge belongs to another dataset. Rerun Essam's preparation.")
    records = payload.get("records", [])
    ids = [row["artifact_id"] for row in records]
    if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in catalog}:
        raise ValueError("Knowledge IDs do not match the full canonical catalog.")
    for row in records:
        for field in ("artifact_id", "title", "text", "source_url"):
            if not isinstance(row.get(field), str):
                raise ValueError("Missing knowledge string field: " + field)
        if row.get("language") != "en" or not row["source_url"].startswith(("https://", "http://")):
            raise ValueError("Knowledge requires English records and source URLs.")
        expected = "\n".join(f"{label}: {row['fields'][field]}" for field, label in FIELD_LABELS.items() if row["fields"].get(field))
        if row["text"] != expected:
            raise ValueError("Knowledge text and structured fields disagree.")
    return payload, records, mapping


def chunk_records(records, encoder, overlap=24):
    """Split on tokenizer offsets; retain exact original character spans."""
    if not 0 <= overlap < MAX_TOKENS - 4:
        raise ValueError("Invalid chunk overlap.")
    tokenizer = encoder.tokenizer
    budget = MAX_TOKENS - tokenizer.num_special_tokens_to_add(pair=False)
    chunks = []
    for row in records:
        text = row["text"]
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False)["offset_mapping"]
        offsets = [(a, b) for a, b in offsets if b > a]
        start, number = 0, 0
        while start < len(offsets):
            end = min(start + budget, len(offsets))
            left, right = offsets[start][0], offsets[end - 1][1]
            while len(tokenizer(text[left:right])["input_ids"]) > MAX_TOKENS and end > start + 1:
                end -= 1
                right = offsets[end - 1][1]
            if len(tokenizer(text[left:right])["input_ids"]) > MAX_TOKENS:
                raise ValueError("A source span cannot fit the embedding budget.")
            chunks.append({"chunk_id": f"{row['artifact_id']}:week1:{number:04d}", "artifact_id": row["artifact_id"],
                           "title": row["title"], "source_url": row["source_url"], "text": text[left:right],
                           "start_char": left, "end_char": right})
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap)
            number += 1
    return chunks


def build(root, knowledge, directory, encoder):
    payload, records, _ = load_knowledge(root, knowledge)
    directory = output_path(root, directory)
    directory.mkdir(parents=True, exist_ok=True)
    chunks = chunk_records(records, encoder)
    if not chunks:
        raise ValueError("No knowledge text to index.")
    print(f"Prepared {len(chunks)} token-bounded chunks from {len(records)} artifacts", flush=True)
    vectors = encoder.encode([row["text"] for row in chunks])
    if vectors.shape != (len(chunks), 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid embedding matrix.")
    index = faiss.IndexFlatIP(384)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    faiss.write_index(index, str(directory / "index.faiss"))
    write_json(directory / "chunks.json", chunks)
    config = {"format": 1, "scope": "Independent Week 1 baseline over Essam knowledge, not the original release index",
              "source_fingerprint": payload["source_fingerprint"], "knowledge_sha256": sha256(knowledge),
              "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
              "generation_model": GENERATION_MODEL, "generation_revision": GENERATION_REVISION,
              "max_tokens": MAX_TOKENS, "overlap_tokens": 24, "pooling": "attention_mask_mean", "normalization": "L2",
              "artifact_count": len(records), "chunk_count": len(chunks), "metric": "INNER_PRODUCT",
              "chunks_sha256": sha256(directory / "chunks.json"), "index_sha256": sha256(directory / "index.faiss")}
    if fingerprint(root) != payload["source_fingerprint"]:
        raise RuntimeError("Source changed during indexing.")
    write_json(directory / "config.json", config)
    return config


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def evidence_sentence(candidate, text):
    needle = normalized(candidate).strip(" .")
    if not needle or needle in {"yes", "no", "unknown", "none", "not enough", "not enough information"}:
        return None
    for line in text.splitlines():
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            if re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", normalized(sentence)):
                return sentence.strip()
    return None


def requested_field(question, title):
    """Recognize direct metadata questions without confusing events or objects."""
    words = lambda value: " ".join(re.findall(r"\w+", normalized(value)))
    q = words(question)
    references = {"it", "this", "this artifact", "the artifact", "this object", "the object", "this item", "the item"}
    if title:
        references.update({words(title), "the " + words(title), "this " + words(title)})
    subject = "(?:" + "|".join(re.escape(value) for value in sorted(references, key=len, reverse=True)) + ")"
    patterns = {
        "material": [rf"what (?:material )?is {subject} made (?:of|from)", r"what (?:is|are) its materials?"],
        "date": [rf"when was {subject} (?:made|created)", r"what is its (?:date|creation date)"],
        "period": [rf"(?:what|which) (?:period|era) is {subject} from", r"what is its (?:period|era)"],
        "dynasty": [rf"(?:what|which) dynasty is {subject} from", r"what is its dynasty"],
    }
    for field, options in patterns.items():
        if any(re.fullmatch(pattern, q) for pattern in options):
            return field, True
    metadata_question = bool(re.search(r"\b(material|materials|period|era|dynasty|date|dated|when)\b|made (?:of|from)", q))
    return None, metadata_question


class RAG:
    def __init__(self, root, knowledge, directory, encoder, generator):
        payload, records, self.mapping = load_knowledge(root, knowledge)
        self.records = {row["artifact_id"]: row for row in records}
        directory = Path(directory)
        self.config = read_json(directory / "config.json")
        required = {"format": 1, "source_fingerprint": payload["source_fingerprint"], "knowledge_sha256": sha256(knowledge),
                    "embedding_model": EMBEDDING_MODEL, "embedding_revision": EMBEDDING_REVISION,
                    "generation_model": GENERATION_MODEL, "generation_revision": GENERATION_REVISION,
                    "max_tokens": MAX_TOKENS, "pooling": "attention_mask_mean", "normalization": "L2", "metric": "INNER_PRODUCT"}
        if any(self.config.get(key) != value for key, value in required.items()):
            raise ValueError("The knowledge or model changed. Rebuild this Week 1 index.")
        for name, key in (("chunks.json", "chunks_sha256"), ("index.faiss", "index_sha256")):
            if sha256(directory / name) != self.config.get(key):
                raise ValueError("Index bundle checksum mismatch.")
        self.chunks = read_json(directory / "chunks.json")
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if (self.index.ntotal != len(self.chunks) or self.index.ntotal != self.config.get("chunk_count")
                or len(self.records) != self.config.get("artifact_count")
                or len({row["chunk_id"] for row in self.chunks}) != len(self.chunks)
                or self.index.d != 384 or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise ValueError("Vector and chunk index mismatch.")
        self.vectors = self.index.reconstruct_n(0, self.index.ntotal)
        if not np.isfinite(self.vectors).all() or not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-4):
            raise ValueError("Index must contain normalized finite vectors.")
        self.positions = {}
        for number, chunk in enumerate(self.chunks):
            record = self.records.get(chunk["artifact_id"])
            if not record or record["text"][chunk["start_char"]:chunk["end_char"]] != chunk["text"] or chunk["source_url"] != record["source_url"]:
                raise ValueError("A chunk does not match its source record.")
            self.positions.setdefault(chunk["artifact_id"], []).append(number)
        self.encoder, self.generator = encoder, generator

    def retrieve(self, identifier, question, top_k=3):
        if not question.strip() or len(question) > 600 or not 1 <= top_k <= 10:
            raise ValueError("Provide a question of 1-600 characters and top_k between 1 and 10.")
        aid = resolve_id(identifier, self.mapping)
        positions = self.positions.get(aid, [])
        if not positions:
            return []
        query = self.encoder.encode([question])[0]
        scores = self.vectors[positions] @ query
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [{**self.chunks[positions[i]], "similarity": float(scores[i])} for i in order]

    def ask(self, identifier, question):
        aid = resolve_id(identifier, self.mapping)
        hits = self.retrieve(aid, question)
        result = {"artifact_id": aid, "requested_artifact_id": identifier, "question": question, "language": "en",
                  "status": "insufficient_evidence", "answer": FALLBACK, "sources": [], "retrieved_chunks": hits,
                  "generator_used": False, "generated_candidate": None, "source_fingerprint": self.config["source_fingerprint"],
                  "evidence_note": "Quotes are from the supplied catalog; independent historical verification is pending."}
        if not hits:
            result["reason"] = "Unknown artifact ID or missing text."
            return result
        if re.search(r"\b(today|current|currently|now|price|cost|worth|opening|restored|stolen|display case|artist prefer)\b", question.casefold()):
            result["reason"] = "This baseline does not establish current information, value or those external events."
            return result
        if hits[0]["similarity"] < 0.20:
            result["reason"] = "Retrieval evidence is too weak."
            return result
        record = self.records[aid]
        if re.search(r"\b(height|tall|width|wide|weight|weigh|dimensions)\b", question.casefold()):
            if not re.search(r"\b\d+(?:\.\d+)?\s*(?:cm|mm|m|kg|centimet\w*|metres?|meters?|inches|feet)\b", record["text"].casefold()):
                result["reason"] = "The supplied record does not contain measurements."
                return result
        selected = list(hits)
        while selected:
            ranges = sorted((hit["start_char"], hit["end_char"]) for hit in selected)
            merged = []
            for start, end in ranges:
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            context = "\n".join(record["text"][start:end] for start, end in merged)
            if self.generator.fits(question, context):
                break
            selected.pop()
        if not selected:
            result["reason"] = "Retrieved context exceeds the generation budget."
            return result
        field, metadata_question = requested_field(question, record["title"])
        if metadata_question and field is None:
            result["reason"] = "Ask one direct metadata question about the selected artifact; another object or event needs separate evidence."
            return result
        if field:
            line = FIELD_LABELS[field] + ": " + record["fields"].get(field, "")
            if not record["fields"].get(field) or line not in context:
                result["reason"] = "The requested field is absent from retrieved evidence."
                return result
            # A field-specific question must be grounded in that field's line.
            support = line
        else:
            support = context
        # Use the retrieved field line for a field question, avoiding unrelated
        # description sentences that can distract this small baseline generator.
        generation_context = f"Selected artifact: {record['title']}\n{support}" if field else context
        candidate = self.generator.answer(question, generation_context)
        result.update(generator_used=True, generated_candidate=candidate)
        sentence = evidence_sentence(candidate, support)
        method = "generated_source_quote"
        if not sentence and field:
            # A direct field question can quote its retrieved field even when
            # the small generator abstains. Record this path transparently.
            sentence = support
            method = "retrieved_field"
        if not sentence:
            result["reason"] = "Generated text did not match the retrieved source evidence."
            return result
        result.update(status="answered", answer=sentence + " [1]", answer_method=method, reason="The displayed answer quotes retrieved source text.",
                      sources=[{"label": "[1]", "artifact_id": aid, "title": record["title"], "url": record["source_url"],
                                "evidence": sentence, "chunk_ids": [hit["chunk_id"] for hit in selected]}])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "ask"])
    parser.add_argument("--person2-root", type=Path, required=True)
    parser.add_argument("--knowledge", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--artifact-id")
    parser.add_argument("--question")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "ask" and (not args.artifact_id or not args.question or not args.output):
        parser.error("ask requires --artifact-id, --question and --output.")
    print("Loading the pinned multilingual encoder...", flush=True)
    encoder = Encoder(args.allow_download)
    if args.command == "build":
        if args.allow_download:
            Generator(True)
        result = build(args.person2_root, args.knowledge, args.index, encoder)
        print(json.dumps({key: result[key] for key in ("artifact_count", "chunk_count", "source_fingerprint")}))
    else:
        service = RAG(args.person2_root, args.knowledge, args.index, encoder, Generator(args.allow_download))
        result = service.ask(args.artifact_id, args.question)
        write_json(output_path(args.person2_root, args.output), result)
        print(json.dumps({"status": result["status"], "answer": result["answer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
