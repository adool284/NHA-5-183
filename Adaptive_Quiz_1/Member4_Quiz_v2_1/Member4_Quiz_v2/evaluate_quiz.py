"""
Evaluation for the Member 4 quiz  (run:  python evaluate_quiz.py --llm mock|qwen|none)

Part A  Deterministic audit of EVERY artifact in the dataset (all fields, all levels, EN + AR).
        An independent checker (re-derives answers from the raw CSV and the chronology / material
        tables, does not reuse the serving code's decisions) verifies each served question:
        one correct option, no overlapping options, no uncertain facts, evidence in RAG Text.
Part B  Random visitor sessions (1-8 visited artifacts, both museums, all classes): questions
        must only concern visited artifacts and never repeat.
Part C  LLM-written questions from descriptions (needs --llm). Reports acceptance rate and the
        rejection reasons, and writes a human review sheet (llm_review_sheet.csv).
        With --llm mock this only tests the validators (the mock is not an AI model).
        With --llm qwen it measures the real model - run this on your machine.

Outputs go to ./evaluation/
"""
import argparse
import collections
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

import pandas as pd

import quiz_module as quiz
import quiz_vocab as V

OUT = Path(__file__).resolve().parent / "evaluation"


# ---------------------------------------------------------------------------
# Independent checker (does NOT call quiz.canonicalize / quiz.conflicts)
# ---------------------------------------------------------------------------
_PERIOD_SPAN = {p[0]: (p[1], p[2]) for p in V.PERIODS}
_AR_TO_EN = {}
for _raw, (_en, _ar, _l) in V.CLASS_LABELS.items():
    _AR_TO_EN[_ar] = _en
for _lab, _s, _e, _ar in V.PERIODS:
    _AR_TO_EN[_ar] = _lab
for _pat, _lab, _grp, _ar, _l in V.MATERIAL_RULES:
    _AR_TO_EN[_ar] = _lab
_MATERIAL_GROUP = {lab: grp for _p, lab, grp, _a, _l in V.MATERIAL_RULES}


def _en(label):
    if label in _AR_TO_EN:
        return _AR_TO_EN[label]
    m = re.fullmatch(r"الأسرة (\d+)(?:\u2013(\d+))?", label)
    if m:
        return f"Dynasty {m.group(1)}" + (f"\u2013{m.group(2)}" if m.group(2) else "")
    return label


def _span(field, label_en):
    """Independent 'what does this answer cover' set."""
    if field == "Period":
        return {("p", _PERIOD_SPAN[label_en])}                       # exact year interval
    if field == "Dynasty":
        m = re.fullmatch(r"Dynasty (\d+)(?:\u2013(\d+))?", label_en)
        a = int(m.group(1))
        b = int(m.group(2) or a)
        return {("d", n) for n in range(a, b + 1)}
    if field == "Material":
        return {("g", _MATERIAL_GROUP[label_en])}
    return {("c", label_en)}


def _overlap(x, y):
    """True if two answer-spans can both be true for the same artifact."""
    for kind1, v1 in x:
        for kind2, v2 in y:
            if kind1 != kind2:
                continue
            if kind1 == "p":
                if v1[0] < v2[1] and v2[0] < v1[1]:      # intervals share more than a boundary year
                    return True
            elif v1 == v2:
                return True
    return False


def _raw_uncertain(raw):
    r = (raw or "").casefold()
    outside = re.sub(r"\([^)]*\)", " ", r)
    return ("?" in r or bool(re.search(r"\bor\b", outside))
            or any(w in r for w in ("probabl", "possibl", "perhaps", "likely", "later", "per card")))


def audit_metadata_question(q, row):
    """-> list of failed check names (empty = pass)."""
    fails = []
    field, lang = q["field"], q["language"]
    options = q["options"]
    if len(options) != 4 or len({quiz.normalize_answer(o) for o in options}) != 4:
        fails.append("options_not_4_unique")
    if sum(quiz.normalize_answer(o) == quiz.normalize_answer(q["_correct_answer"]) for o in options) != 1:
        fails.append("not_exactly_one_correct_option")
    labels_en = [_en(o) for o in options]
    try:
        spans = [_span(field, l) for l in labels_en]
    except Exception:
        return fails + ["option_not_in_controlled_vocabulary"]
    for i in range(4):
        for j in range(i + 1, 4):
            if _overlap(spans[i], spans[j]):
                fails.append("overlapping_options")
    raw = q["_source"]["fact_value"]
    if _raw_uncertain(raw):
        fails.append("uncertain_fact_served")
    rag_lines = [ln.strip().casefold() for ln in (row["RAG Text"] or "").splitlines()]
    if f"{field}: {raw}".casefold() not in rag_lines:
        fails.append("fact_not_in_rag_text")
    if _en(q["_correct_answer"]) != q["_source"]["canonical_answer"]:
        fails.append("answer_label_mismatch")
    if field == "Period":
        correct = _span("Period", _en(q["_correct_answer"]))
        date = quiz._parse_bc_range(row.get("Date"))
        if date is not None:
            lo, hi = date
            a, b = _PERIOD_SPAN[_en(q["_correct_answer"])]
            if not (lo - 100 <= b and a <= hi + 100):
                fails.append("period_contradicts_date")
        del correct
    title = (row.get("Title") or "").casefold()
    if _en(q["_correct_answer"]).casefold() in title:
        fails.append("answer_in_title")
    return fails


# ---------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------
def part_a(df, languages=("en", "ar")):
    rows_by_id = {r["Artifact ID"]: r for r in df.to_dict("records")}
    quiz.set_llm_backend(False)       # deterministic: template wording
    stats = collections.Counter()
    fail_counter = collections.Counter()
    failures = []
    skip_reasons = collections.Counter()
    askable_per_artifact = {}
    t0 = time.time()
    for aid, row in rows_by_id.items():
        facts = {}
        for field in quiz.ALL_FIELDS:
            fact, reason = quiz.extract_fact(row, field)
            if fact is None:
                skip_reasons[(field, reason)] += 1
            else:
                facts[field] = fact["label"]
        askable_per_artifact[aid] = len(facts)
        for lang in languages:
            for level in quiz.LEVELS:
                for scope in ("dataset",):
                    s = quiz.start_quiz_session([aid], language=lang, seed=zlib.crc32(f'{aid}|{level}|{scope}'.encode()) % 10**6,
                                                distractor_scope=scope, question_mode="metadata")
                    s["level"] = level
                    served = 0
                    while True:
                        pub = quiz.generate_quiz_question(s)
                        if pub.get("status") != "ok":
                            break
                        q = s["pending"]
                        served += 1
                        stats["questions_audited"] += 1
                        fails = audit_metadata_question(q, row)
                        if pub["artifact_id"] != aid:
                            fails.append("question_about_unvisited_artifact")
                        for f in fails:
                            fail_counter[f] += 1
                        if fails:
                            failures.append({"artifact_id": aid, "field": q["field"], "level": level,
                                             "language": lang, "fails": ";".join(fails),
                                             "options": " | ".join(q["options"])})
                        quiz.submit_answer(s, q["options"][0])
                    if served != len(facts):
                        stats["served_count_differs_from_askable"] += 1
    stats["seconds"] = round(time.time() - t0, 1)
    n_art = len(rows_by_id)
    cover = pd.Series(askable_per_artifact)
    return {
        "artifacts": n_art,
        "stats": dict(stats),
        "failures_by_check": dict(fail_counter),
        "failure_rows": failures,
        "skip_reasons": {f"{k[0]}:{k[1]}": v for k, v in sorted(skip_reasons.items())},
        "artifacts_with_ge1_askable_fact": int((cover >= 1).sum()),
        "artifacts_with_0_askable_fact": int((cover == 0).sum()),
        "askable_facts_total": int(cover.sum()),
    }


# ---------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------
def part_b(df, n_sessions, seed):
    rng = random.Random(seed)
    ids_by_class = {c: g["Artifact ID"].tolist() for c, g in df.groupby("Project Class")}
    all_ids = df["Artifact ID"].tolist()
    problems = collections.Counter()
    served_total = collections.Counter()
    for k in range(n_sessions):
        size = rng.randint(1, 8)
        if k % 3 == 0:                                        # single-class visit
            visited = rng.sample(ids_by_class[rng.choice(list(ids_by_class))], size)
        else:
            visited = rng.sample(all_ids, size)
        s = quiz.start_quiz_session(visited, language=rng.choice(["en", "ar"]), seed=k,
                                    distractor_scope=rng.choice(["dataset", "visited"]),
                                    question_mode=rng.choice(["llm", "metadata"]))
        seen = set()
        for _ in range(14):
            pub = quiz.generate_quiz_question(s)
            if pub.get("status") != "ok":
                break
            q = s["pending"]
            served_total[q["question_source"]] += 1
            if q["artifact_id"] not in visited:
                problems["question_about_unvisited_artifact"] += 1
            key = (q["artifact_id"], q["field"],
                   quiz.norm_text(q["_source"]["evidence"]) if q["field"] == "Description" else "")
            if key in seen:
                problems["repeated_question"] += 1
            seen.add(key)
            if len({quiz.normalize_answer(o) for o in q["options"]}) != 4:
                problems["duplicate_options"] += 1
            if q["question_source"] == "llm_description":
                row = df[df["Artifact ID"] == q["artifact_id"]].iloc[0]
                rag = quiz.norm_text(row["RAG Text"])
                if quiz.norm_text(q["_source"]["evidence"]) not in rag:
                    problems["llm_evidence_not_in_rag_text"] += 1
                if not quiz.contains_phrase(quiz.norm_text(q["_source"]["evidence"]),
                                            quiz.norm_text(q["_correct_answer"])):
                    problems["llm_answer_not_in_evidence"] += 1
            correct = rng.random() < 0.7
            quiz.submit_answer(s, q["_correct_answer"] if correct else
                               next(o for o in q["options"] if o != q["_correct_answer"]))
        quiz.finish_quiz(s)
    return {"sessions": n_sessions, "questions_by_source": dict(served_total), "problems": dict(problems)}


# ---------------------------------------------------------------------------
# Part C
# ---------------------------------------------------------------------------
def part_c(df, n, seed):
    rng = random.Random(seed)
    cand = df[df["Description"].notna()].to_dict("records")
    sample = rng.sample(cand, min(n, len(cand)))
    accepted, rejected = [], []
    reasons = collections.Counter()
    t0 = time.time()
    for i, row in enumerate(sample, 1):
        if i % 5 == 0 or i == 1:
            el = time.time() - t0
            print(f"  [{i}/{len(sample)}] accepted so far: {len(accepted)} | elapsed {el:.0f}s | "
                  f"ETA {el / i * (len(sample) - i):.0f}s", flush=True)
        built, why = quiz.build_llm_question(row, "medium", random.Random(rng.random()), seed=rng.randrange(10**6))
        if built is None:
            reasons.update(why)
            rejected.append({"artifact_id": row["Artifact ID"], "title": row["Title"], "reasons": ";".join(why)})
        else:
            accepted.append({
                "artifact_id": row["Artifact ID"], "title": row["Title"], "question": built["question"],
                "options": " | ".join(built["options"]), "correct_answer": built["answer"],
                "evidence": built["evidence"], "source_url": row["Source URL"],
                "checks_passed": ",".join(built["checks_passed"]),
                "human_ok_Y_N": "", "human_notes": ""})
    return {
        "attempted": len(sample), "accepted": len(accepted), "rejected": len(rejected),
        "acceptance_rate_pct": round(100 * len(accepted) / max(1, len(sample)), 1),
        "rejection_reasons": dict(reasons.most_common()),
        "seconds": round(time.time() - t0, 1),
        "accepted_rows": accepted, "rejected_rows": rejected,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", choices=["none", "mock", "qwen"], default="none")
    ap.add_argument("--n", type=int, default=60, help="artifacts for the LLM part")
    ap.add_argument("--sessions", type=int, default=300)
    ap.add_argument("--skip-audit", action="store_true", help="skip Part A (takes ~2 min, already passed)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    df = quiz.get_dataset()

    if args.skip_audit:
        a = {"artifacts": len(df), "stats": {}, "failures_by_check": {}, "failure_rows": [], "skip_reasons": {}}
    else:
        print("Part A: auditing every artifact ...")
        a = part_a(df)
    if a["failure_rows"]:
        pd.DataFrame(a["failure_rows"]).to_csv(OUT / "audit_failures.csv", index=False)

    import mock_llm_for_tests as mock
    print("Part B: random visitor sessions (session logic only - uses the mock, not the real model) ...")
    quiz.set_llm_backend(mock.extractive_mock)
    b = part_b(df, args.sessions, args.seed)

    c = None
    if args.llm != "none":
        if args.llm == "mock":
            quiz.set_llm_backend(mock.extractive_mock)
        else:
            quiz.set_llm_backend(None)
            print("LLM status:", quiz.get_llm_status())
            if not quiz.get_llm_status()["available"]:
                sys.exit("Qwen is not available (install torch/transformers and allow the model download).")
        print(f"Part C: LLM-written questions from {args.n} artifacts ({args.llm}) ...")
        c = part_c(df, args.n, args.seed)
        pd.DataFrame(c["accepted_rows"]).to_csv(OUT / f"llm_review_sheet_{args.llm}.csv", index=False)
        pd.DataFrame(c["rejected_rows"]).to_csv(OUT / f"llm_rejected_{args.llm}.csv", index=False)

    summary = {"module_version": quiz.MODULE_VERSION, "llm_backend": args.llm,
               "part_a": {k: v for k, v in a.items() if k != "failure_rows"},
               "part_b": b,
               "part_c": None if c is None else {k: v for k, v in c.items() if not k.endswith("_rows")}}
    (OUT / f"evaluation_summary_{args.llm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
