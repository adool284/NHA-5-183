"""Week 2 content-based recommendations. Python 3.10+; no extra packages.

Run:
  python recommendations.py recommend --person2-root PATH --output RUN_DIR \
    --interests "gold jewelry" --liked-id MET_2021.41.2 --top-k 5
  python recommendations.py evaluate --person2-root PATH --output RUN_DIR

Catalog data stays in Person 2. Interaction weights describe a profile, not
observed popularity. Evaluation uses metadata proxies, not human relevance labels.

Shared helpers: Essam_Week2/week2_common.py. Keep this week's person folders
side by side; extract the complete weekly archive for the runnable layout.
"""

from __future__ import annotations
import argparse
import collections
import json
import math
import re
from pathlib import Path
import sys


_COMMON_FOLDER = Path(__file__).resolve().parents[1] / "Essam_Week2"
if not (_COMMON_FOLDER / "week2_common.py").is_file():
    raise FileNotFoundError(
        "Missing Essam_Week2/week2_common.py. Extract the complete Week 2 archive, "
        "or extract the person archives into the same folder without renaming their folders."
    )
sys.path.insert(0, str(_COMMON_FOLDER))
from week2_common import load_catalog, load_mapping, resolve_id, safe_output, source_fingerprint, write_json


FIELD_WEIGHTS = {"name": 2.0, "category": 3.0, "period": 2.0,
                 "dynasty": 1.0, "material": 2.0, "description": 0.5}
STOP_WORDS = set("a an the of and or in on to for from with by is was are were this that it its as at be".split())
SIGNAL_WEIGHTS = {"view": 0.3, "like": 1.0, "save": 1.5, "dislike": -1.0}


def tokens(value):
    return [word for word in re.findall(r"[a-z0-9]+", str(value).lower())
            if len(word) > 1 and word not in STOP_WORDS]


def category(record):
    value = str(record.get("category", "")).strip().replace("_", "/").casefold()
    return {"amulets": "amulet", "scarabs": "scarab"}.get(value, value)


def normalized(vector):
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def dot(left, right):
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


class Recommender:
    """Weighted TF-IDF with cosine ranking and a small diversity penalty."""

    def __init__(self, records, mapping=None):
        self.records = {record["id"]: record for record in records}
        if len(self.records) != len(records):
            raise ValueError("Duplicate canonical artifact IDs.")
        self.mapping = mapping or {}
        terms = {}
        frequency = collections.Counter()
        for artifact_id, record in self.records.items():
            counts = collections.Counter()
            for field, weight in FIELD_WEIGHTS.items():
                words = tokens(category(record) if field == "category" else record.get(field, ""))
                for word, count in collections.Counter(words).items():
                    counts[f"{field}:{word}"] += weight * (1.0 + math.log(count))
            terms[artifact_id] = dict(counts)
            frequency.update(counts.keys())
        self.idf = {term: math.log((1 + len(records)) / (1 + count)) + 1
                    for term, count in frequency.items()}
        self.vectors = {artifact_id: normalized({key: value * self.idf[key]
                                                for key, value in counts.items()})
                        for artifact_id, counts in terms.items()}

    def canonical(self, artifact_id):
        return resolve_id(artifact_id, self.mapping)

    def profile(self, signals, interests):
        profile = collections.defaultdict(float)
        seen, ignored = set(), []
        for signal in signals:
            identifier = signal.get("artifact_id")
            action = signal.get("action", "like")
            if not isinstance(identifier, str) or not identifier.strip():
                ignored.append(signal)
                continue
            artifact_id = self.canonical(identifier)
            if artifact_id not in self.records or action not in SIGNAL_WEIGHTS:
                ignored.append(signal)
                continue
            seen.add(artifact_id)
            for term, value in self.vectors[artifact_id].items():
                profile[term] += SIGNAL_WEIGHTS[action] * value
        for word in tokens(interests):
            for field, weight in FIELD_WEIGHTS.items():
                key = f"{field}:{word}"
                if key in self.idf:
                    profile[key] += weight * self.idf[key]
        return normalized(profile), seen, ignored

    def recommend(self, signals=(), interests="", top_k=5, visual_only=False, diversity=0.15):
        if top_k < 1 or not 0 <= diversity <= 1:
            raise ValueError("top_k must be positive and diversity must be between zero and one.")
        profile, seen, ignored = self.profile(signals, interests)
        candidates = [key for key, record in self.records.items()
                      if key not in seen and (not visual_only or record.get("has_visual_reference"))]
        if not candidates:
            return {"mode": "empty", "results": [], "ignored_signals": ignored,
                    "message": "No eligible artifacts remain in this catalog."}
        scores = {key: dot(profile, self.vectors[key]) for key in candidates}
        personalized = any(score > 0 for score in scores.values())
        selected = []
        if personalized:
            remaining = {key for key in candidates if scores[key] > 0}
            while remaining and len(selected) < top_k:
                def rank(key):
                    similarity = max((max(0.0, dot(self.vectors[key], self.vectors[other]))
                                      for other in selected), default=0.0)
                    return (scores[key] - diversity * similarity, scores[key])
                key = min(remaining, key=lambda item: (-rank(item)[0], -rank(item)[1], item))
                selected.append(key)
                remaining.remove(key)
            mode = "personalized"
        else:
            # No popularity is invented: browse one item per category in stable order.
            # A negative-only profile still excludes content matching its dislikes.
            candidates = [key for key in candidates if scores[key] >= -1e-12]
            if not candidates:
                return {"mode": "empty", "results": [], "ignored_signals": ignored,
                        "message": "No candidates remain after applying negative content preferences."}
            groups = collections.defaultdict(list)
            for key in sorted(candidates):
                groups[category(self.records[key])].append(key)
            depth = 0
            while len(selected) < min(top_k, len(candidates)):
                for label in sorted(groups):
                    if depth < len(groups[label]):
                        selected.append(groups[label][depth])
                        if len(selected) == min(top_k, len(candidates)):
                            break
                depth += 1
            mode = "cold_start"
        results = []
        for key in selected:
            overlap = sorted(((term, profile.get(term, 0) * value)
                              for term, value in self.vectors[key].items()
                              if profile.get(term, 0) > 0), key=lambda pair: (-pair[1], pair[0]))
            if personalized:
                labels = list(dict.fromkeys(term.split(":", 1)[1] for term, _ in overlap))[:3]
                reason = "Matches profile terms: " + ", ".join(labels) + "."
            else:
                reason = "Catalog browsing suggestion from " + str(self.records[key].get("category") or "an uncategorized group") + "."
            results.append({"artifact_id": key, "title": self.records[key].get("name", key),
                            "score": round(scores[key], 6), "reason": reason,
                            "source_url": self.records[key].get("sourceUrl", "")})
        return {"mode": mode, "results": results, "ignored_signals": ignored,
                "score_meaning": "Cosine content similarity, not probability or popularity."}


def evaluate(model, top_k=5):
    """Create inspectable metadata scenarios and deterministic baseline metrics."""
    groups = collections.defaultdict(list)
    for key, record in model.records.items():
        groups[category(record)].append(key)
    scenarios = []
    for label, ids in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))[:6]:
        if len(ids) < 2:
            continue
        seed = sorted(ids)[0]
        result = model.recommend([{"artifact_id": seed, "action": "like"}], top_k=top_k)
        recommendations = [row["artifact_id"] for row in result["results"]]
        relevant = set(ids) - {seed}
        pairs = [(a, b) for index, a in enumerate(recommendations) for b in recommendations[index + 1:]]
        scenarios.append({"name": "Liked artifact in " + label, "seed_id": seed,
                          "relevance_proxy": "Same normalized catalog category, excluding the seed.",
                          "category_precision_at_k": sum(key in relevant for key in recommendations) / max(1, len(recommendations)),
                          "category_diversity": len({category(model.records[key]) for key in recommendations}) / max(1, len(recommendations)),
                          "mean_content_distance": sum(1 - dot(model.vectors[a], model.vectors[b]) for a, b in pairs) / max(1, len(pairs)),
                          "seen_items_returned": seed in recommendations, "recommendations": result})
    return {"evaluation_type": "Metadata proxy scenarios; human relevance is not measured.",
            "feature_weights": FIELD_WEIGHTS, "signal_weights": SIGNAL_WEIGHTS,
            "ranking": "L2-normalized field TF-IDF cosine, diversity penalty 0.15",
            "scenarios": scenarios, "cold_start": model.recommend(top_k=top_k),
            "empty_catalog": Recommender([]).recommend(top_k=top_k),
            "unknown_profile": model.recommend([{"artifact_id": "UNSUPPORTED_ID", "action": "like"}], top_k=top_k)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["recommend", "evaluate"])
    parser.add_argument("--person2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interests", default="")
    parser.add_argument("--liked-id", action="append", default=[])
    parser.add_argument("--signals", type=Path, help="JSON list of artifact_id/action entries: view, like, save, dislike.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual-only", action="store_true")
    args = parser.parse_args()
    output = safe_output(args.person2_root, args.output)
    records = load_catalog(args.person2_root)
    model = Recommender(records, load_mapping(args.person2_root))
    if args.command == "evaluate":
        result = evaluate(model, args.top_k)
    else:
        signals = json.loads(args.signals.read_text(encoding="utf-8")) if args.signals else []
        if not isinstance(signals, list) or any(not isinstance(signal, dict) for signal in signals):
            parser.error("--signals must contain a JSON list of objects.")
        signals += [{"artifact_id": artifact_id, "action": "like"} for artifact_id in args.liked_id]
        result = model.recommend(signals, args.interests, args.top_k, args.visual_only)
    result["source_fingerprint"] = source_fingerprint(args.person2_root)
    path = output / ("recommendation_evaluation.json" if args.command == "evaluate" else "recommendations.json")
    write_json(path, result)
    print(str(path.resolve()))


if __name__ == "__main__":
    main()
