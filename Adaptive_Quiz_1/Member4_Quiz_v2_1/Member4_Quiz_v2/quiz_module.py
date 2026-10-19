"""
Member 4 - Adaptive Museum Quiz (v2.0.0)

What this module does
---------------------
* Builds multiple-choice questions ONLY about artifacts the visitor actually visited.
* Two question sources, both grounded in the artifact's own "RAG Text" record:
    1. llm_description : the LLM (Qwen2.5-0.5B-Instruct, the project model) READS the artifact's
                         description and WRITES the question, the correct answer, 3 wrong answers
                         and the evidence sentence. Nothing is served until it passes the
                         validators in `validate_llm_question` (verbatim evidence, answer inside
                         evidence, no leak, wrong answers absent from the record, no hedged
                         wording) and a second "answer it back" verification pass.
    2. metadata        : Class / Material / Period / Dynasty questions. The fact and answer come
                         from the record, the wording is phrased by the LLM (validated) with a
                         safe template fallback.
* Uncertain facts ("Granite (?)", "Steatite or serpentinite", "Dynasty 18, possibly later",
  "Late Period or later" ...) are never asked.
* Answer options can never overlap: every fact is mapped to one canonical label with a
  conflict key (see quiz_vocab.py), and distractors must have a disjoint key.
  e.g. "Wood" vs "Sycomore wood", "Bronze" vs "Cupreous metal", "New Kingdom" vs
  "New Kingdom, Ramesside", "Dynasty 19-20" vs "Dynasty 20" can never share a question.
* Sessions are plain JSON dicts (FastAPI friendly); events use the field names Member 6 needs.

Public API
----------
    start_quiz_session(visited_artifact_ids, language="en", voice_used=False, seed=None,
                       distractor_scope="dataset", question_mode="llm")
    generate_quiz_question(session)       -> public question dict (no answer inside)
    submit_answer(session, user_answer)   -> result dict (correct answer + evidence)
    get_quiz_summary(session) / finish_quiz(session) / export_events_jsonl(session, path)
    set_llm_backend(fn)                   -> plug another LLM (or False to disable) - used by tests

question_mode: "llm" (default) = try an LLM-written question first, fall back to metadata
questions; "metadata" = never call the LLM for question content.
"""

import json
import random
import re
import uuid
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

import quiz_vocab as V

MODULE_VERSION = "2.0.0"
DATASET_PATH = Path(__file__).resolve().parent / "combined_rag_dataset.csv"

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
QUESTION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # same model Member 1 uses in the RAG
_QUESTION_TOKENIZER = None
_QUESTION_MODEL = None
_QUESTION_MODEL_ERROR = None
_LLM_OVERRIDE = None          # None = default Qwen, False = disabled, callable = custom backend

LLM_MAX_ATTEMPTS = 3          # how many artifacts to try per served question before falling back
LLM_MAX_PER_ARTIFACT = 2      # max LLM-written questions per artifact in one session
LLM_VERIFY = True             # second pass: the LLM must pick the right option from the evidence
LLM_MAX_CONTEXT_CHARS = 1500
LLM_MAX_WORDS_ANSWER = 6

LEVELS = ["easy", "medium", "hard"]
QUESTION_FIELDS = {
    "easy": ["Class", "Material", "Period"],
    "medium": ["Period", "Dynasty", "Material"],
    "hard": ["Dynasty", "Period", "Material"],
}
ALL_FIELDS = ["Class", "Material", "Period", "Dynasty"]
FIELD_COLUMN = {"Class": "Project Class", "Material": "Material",
                "Period": "Period", "Dynasty": "Dynasty"}

TEMPLATES = {
    "en": {
        "Class": "What type of object is '{title}'?",
        "Material": "What is the main material of '{title}'?",
        "Period": "Which historical period does '{title}' belong to?",
        "Dynasty": "Which dynasty does '{title}' belong to?",
    },
    "ar": {
        "Class": "ما نوع القطعة '{title}'؟",
        "Material": "ما المادة الأساسية التي صُنعت منها القطعة '{title}'؟",
        "Period": "إلى أي فترة تاريخية تنتمي القطعة '{title}'؟",
        "Dynasty": "إلى أي أسرة تنتمي القطعة '{title}'؟",
    },
}
SUPPORTED_LANGUAGES = set(TEMPLATES)
_FIELD_KEYWORDS = {   # a phrased question must be recognisably about the requested field
    "en": {"Class": r"type|kind|object|category|what is",
           "Material": r"material|made|crafted|carved|constructed",
           "Period": r"period|era|kingdom|historical",
           "Dynasty": r"dynast"},
    "ar": {"Class": r"نوع|فئة|صنف",
           "Material": r"مادة|صنع|صُنع",
           "Period": r"فترة|عصر|حقبة|دولة",
           "Dynasty": r"أسرة|اسرة|سلالة"},
}


# ----------------------------------------------------------------------------
# Small text helpers
# ----------------------------------------------------------------------------
def clean_value(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    value = str(value).strip()
    if not value or value.casefold() in {"nan", "none", "null", "unknown", "n/a"}:
        return None
    return value


def normalize_answer(value):
    cleaned = clean_value(value)
    return " ".join(cleaned.casefold().split()) if cleaned else ""


def norm_text(value):
    """Loose normalisation for 'is this text inside that text' checks."""
    s = (value or "").casefold()
    s = (s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"')
         .replace("\u201d", '"').replace("\u2013", "-").replace("\u2014", "-"))
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .;:,\"'")


def contains_phrase(haystack_norm, phrase_norm):
    if not phrase_norm:
        return False
    return re.search(r"(?<!\w)" + re.escape(phrase_norm) + r"(?!\w)", haystack_norm) is not None


_HEDGE_VALUE = re.compile(
    r"\?|probabl|possibl|perhaps|likely|uncertain|unknown|per card|presumabl|\bca\.|"
    r"\blater\b|\bearlier\b|\bmodern\b|\breus|\bformerly\b|attribut", re.I)


def _strip_parens(text):
    return re.sub(r"\([^)]*\)", " ", text)


def is_uncertain(value):
    """Museum-marked uncertainty ('Granite (?)', 'A or B', 'possibly later') must not be quizzed."""
    if value is None:
        return True
    if _HEDGE_VALUE.search(value):
        return True
    return re.search(r"\bor\b", _strip_parens(value), re.I) is not None


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
_DF = None


def get_dataset():
    global _DF
    if _DF is None:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
        data = pd.read_csv(DATASET_PATH)
        required = {"Artifact ID", "Title", "Project Class", "Material", "Period",
                    "Dynasty", "Date", "Description", "Source URL", "RAG Text"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
        data["Artifact ID"] = data["Artifact ID"].apply(clean_value)
        _DF = data
    return _DF


# ----------------------------------------------------------------------------
# Canonical facts  (raw museum value -> ONE clear answer label + conflict key)
# ----------------------------------------------------------------------------
def _fact(field, raw, label, label_ar, key, leak=None):
    return {"field": field, "raw": raw, "label": label, "label_ar": label_ar,
            "key": frozenset(key), "leak": leak}


_PERIOD_LEAK = {"Ptolemaic Period": r"ptolemaic", "Roman Period": r"roman",
                "Predynastic Period": r"predynastic", "Early Dynastic Period": r"early dynastic"}


def _canon_class(raw):
    if raw not in V.CLASS_LABELS:
        return None, "ambiguous_value"
    en, ar, leak = V.CLASS_LABELS[raw]
    return _fact("Class", raw, en, ar, {raw}, leak), None


def _canon_period(raw):
    if is_uncertain(raw):
        return None, "uncertain_value"
    head = re.split(r"[,(]", raw)[0].strip().casefold()
    head = V.PERIOD_ALIASES.get(head, head)
    candidates = [head]
    m = re.match(r"(early|late|mid)\s+(.+)", head)
    if m:
        candidates.append(m.group(2))
    for cand in candidates:
        if cand in V.PERIOD_BY_NAME:
            label, _, _, ar = V.PERIOD_BY_NAME[cand]
            return _fact("Period", raw, label, ar, {label},
                         _PERIOD_LEAK.get(label, re.escape(label.casefold()))), None
    return None, "ambiguous_value"        # ranges like "Late Period-Ptolemaic Period"


def _canon_dynasty(raw):
    if is_uncertain(raw):
        return None, "uncertain_value"
    text = _strip_parens(raw).strip().casefold()
    m = re.fullmatch(r"dynasty\s+(\d{1,2})(?:\s*[\u2013-]\s*(\d{1,2}))?", text)
    if not m:
        return None, "ambiguous_value"    # "Dynasty 12, early", "Late Dynasty 12-13", ...
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    if not (1 <= a <= b <= 31) or b - a > 3:
        return None, "ambiguous_value"
    num = f"{a}" if a == b else f"{a}\u2013{b}"
    return _fact("Dynasty", raw, f"Dynasty {num}", f"الأسرة {num}", range(a, b + 1), r"dynast"), None


def _canon_material(raw):
    if is_uncertain(raw):
        return None, "uncertain_value"
    text = _strip_parens(raw).casefold()
    text = re.sub(r"\b(gold|silver|electrum)\s+(leaf|foil|sheet|inlays?|wire)\b", " ", text)
    text = V._DECORATION.sub(" ", text)
    parts = [re.sub(r"[^a-z\- ]", " ", p).strip()
             for p in re.split(r"[,;:/]|\band\b|\bwith\b|\bover\b|\bon\b", text)]
    parts = [p for p in parts if p]
    found = set()
    for part in parts:
        for rx, label, group, ar, leak in V._MATERIAL_RULES:
            if rx.search(part):
                found.add((label, group, ar, leak))
                break
        else:
            return None, "ambiguous_value"     # something we cannot classify -> skip safely
    labels = {f[0] for f in found}
    if len(labels) != 1:
        return None, "ambiguous_value"          # composite object (e.g. limestone + sandstone)
    label, group, ar, leak = next(iter(found))
    return _fact("Material", raw, label, ar, {group}, leak), None


_CANON = {"Class": _canon_class, "Period": _canon_period,
          "Dynasty": _canon_dynasty, "Material": _canon_material}


def canonicalize(field, raw):
    """(fact, None) or (None, reason). Row independent."""
    raw = clean_value(raw)
    if raw is None:
        return None, "missing"
    return _CANON[field](raw)


def validate_fact(row, field, value):
    """True only if the RAG Text record itself states 'Field: value'."""
    rag_text = clean_value(row.get("RAG Text")) or ""
    wanted = f"{field}: {value}".casefold()
    return any(line.strip().casefold() == wanted for line in rag_text.splitlines())


def _parse_bc_range(date):
    s = clean_value(date)
    if not s or "b.c" not in s.casefold() or "a.d" in s.casefold():
        return None
    nums = [int(n) for n in re.findall(r"(?<!\d)\d{2,4}(?!\d)", s)]
    if not nums:
        return None
    return -max(nums), -min(nums)


def period_matches_date(period_label, date):
    """Sanity check museum Period vs Date. Unparseable dates are not judged."""
    span = _parse_bc_range(date)
    if span is None:
        return True
    _, start, end, _ = V.PERIOD_BY_NAME[period_label.casefold()]
    return span[0] - 100 <= end and start <= span[1] + 100


def extract_fact(row, field):
    """The askable fact of `field` for this artifact row -> (fact, None) or (None, skip_reason)."""
    raw = clean_value(row.get(FIELD_COLUMN[field]))
    fact, reason = canonicalize(field, raw)
    if fact is None:
        return None, reason
    if not validate_fact(row, field, raw):
        return None, "not_in_rag_text"
    title = clean_value(row.get("Title")) or ""
    if fact["leak"] and re.search(fact["leak"], title, re.I):
        return None, "answer_in_title"
    if field == "Period" and not period_matches_date(fact["label"], row.get("Date")):
        return None, "date_mismatch"
    return fact, None


def label_for(fact, language):
    return fact["label_ar"] if language == "ar" else fact["label"]


def conflicts(field, correct, other):
    """Compat helper on RAW museum strings: True if the two cannot safely share a question."""
    a, _ = canonicalize(field, correct)
    b, _ = canonicalize(field, other)
    if a is None or b is None:
        return True
    return bool(a["key"] & b["key"])


# ----------------------------------------------------------------------------
# Distractors
# ----------------------------------------------------------------------------
_POOL = {}


def _pool(field):
    """All distinct canonical labels that exist in the data for a field: {label: fact}."""
    if field not in _POOL:
        pool = {}
        if field == "Class":
            for raw in V.CLASS_LABELS:
                f, _ = _canon_class(raw)
                pool[f["label"]] = f
        elif field == "Period":
            for label, *_ in V.PERIODS:
                f, _ = _canon_period(label)
                pool[f["label"]] = f
        else:
            for raw in get_dataset()[FIELD_COLUMN[field]]:
                f, _ = canonicalize(field, raw)
                if f is not None:
                    pool.setdefault(f["label"], f)
        _POOL[field] = pool
    return _POOL[field]


def _distance(field, a, b):
    if field == "Period":
        names = [p[0] for p in V.PERIODS]
        return abs(names.index(a["label"]) - names.index(b["label"]))
    if field == "Dynasty":
        return min(abs(x - y) for x in a["key"] for y in b["key"])
    return 1.0 - SequenceMatcher(None, a["label"].casefold(), b["label"].casefold()).ratio()


def pick_distractors(field, fact, level, rng, visited_labels, scope):
    pool = _pool(field)
    scope_used = "dataset"
    candidates = [f for f in pool.values() if not (f["key"] & fact["key"])]
    if scope == "visited":
        usable = [f for f in candidates if f["label"] in visited_labels]
        if len(usable) >= 3:
            candidates, scope_used = usable, "visited"
    if len(candidates) < 3:
        return None, scope_used
    ranked = sorted(candidates, key=lambda f: _distance(field, fact, f))   # nearest -> farthest
    if level == "easy":
        window = ranked[-max(3, len(ranked) // 2):]        # clearly different values
    elif level == "hard":
        window = ranked[:max(3, min(8, len(ranked)))]       # nearest plausible values
    else:
        window = ranked
    for pool_choice in (window, ranked):                    # widen if the window cannot fit 3
        shuffled = list(pool_choice)
        rng.shuffle(shuffled)
        chosen = []
        for f in shuffled:
            if all(not (f["key"] & c["key"]) for c in chosen):
                chosen.append(f)
            if len(chosen) == 3:
                return chosen, scope_used
    return None, scope_used


# ----------------------------------------------------------------------------
# Events for Member 6 (JSON-friendly)
# ----------------------------------------------------------------------------
def _log(session, event, **fields):
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": session["session_id"],
        "language": session["language"],
        "voice_used": session["voice_used"],
        "module_version": MODULE_VERSION,
    }
    record.update(fields)
    session["events"].append(record)
    return record


def export_events_jsonl(session, path):
    with open(path, "a", encoding="utf-8") as file:
        for record in session["events"]:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# LLM backend (Qwen2.5-0.5B-Instruct by default, replaceable for tests)
# ----------------------------------------------------------------------------
def _load_question_model():
    """Load Qwen lazily so importing the quiz module never loads a large model."""
    global _QUESTION_TOKENIZER, _QUESTION_MODEL, _QUESTION_MODEL_ERROR
    if _QUESTION_MODEL is not None:
        return _QUESTION_TOKENIZER, _QUESTION_MODEL
    if _QUESTION_MODEL_ERROR:
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _QUESTION_TOKENIZER = AutoTokenizer.from_pretrained(QUESTION_MODEL_NAME)
        _QUESTION_MODEL = AutoModelForCausalLM.from_pretrained(
            QUESTION_MODEL_NAME, torch_dtype="auto", device_map="auto")
        _QUESTION_MODEL.eval()
        return _QUESTION_TOKENIZER, _QUESTION_MODEL
    except Exception as exc:       # no internet / no torch -> quiz still works with templates
        _QUESTION_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        return None, None


def _qwen_generate(prompt, max_new_tokens=220, seed=None, greedy=False):
    tokenizer, model = _load_question_model()
    if tokenizer is None:
        raise RuntimeError(_QUESTION_MODEL_ERROR or "question model unavailable")
    import torch
    messages = [{"role": "system", "content": "You are a careful, factual museum quiz writer."},
                {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    if seed is not None:
        torch.manual_seed(int(seed) % (2**31))
    kwargs = {"do_sample": False} if greedy else {"do_sample": True, "temperature": 0.3, "top_p": 0.9}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def set_llm_backend(backend):
    """backend: callable(prompt, max_new_tokens=..., seed=..., greedy=...) -> str,
    False to disable the LLM completely, or None to restore the default Qwen backend."""
    global _LLM_OVERRIDE
    _LLM_OVERRIDE = backend


def _llm():
    """The generate function to use, or None when no LLM is available."""
    if _LLM_OVERRIDE is False:
        return None
    if callable(_LLM_OVERRIDE):
        return _LLM_OVERRIDE
    tokenizer, model = _load_question_model()
    return _qwen_generate if tokenizer is not None else None


def get_llm_status():
    if _LLM_OVERRIDE is False:
        return {"available": False, "reason": "disabled"}
    if callable(_LLM_OVERRIDE):
        return {"available": True, "model": "custom backend"}
    tokenizer, _ = _load_question_model()
    if tokenizer is None:
        return {"available": False, "reason": _QUESTION_MODEL_ERROR}
    return {"available": True, "model": QUESTION_MODEL_NAME}


# ----------------------------------------------------------------------------
# (A) LLM-written questions from the artifact description
# ----------------------------------------------------------------------------
_HEDGE_TEXT = re.compile(
    r"\b(may|might|could|perhaps|presum\w*|apparent\w*|seem\w*|appear\w*|suggest\w*|indicat\w*|"
    r"thought|believ\w*|assum\w*|suppos\w*|uncertain|unclear|unknown|unidentified|speculat\w*|"
    r"hypothes\w*|conjectur\w*|probabl\w*|possibl\w*|likel\w*|doubt\w*|evident\w*|conceivabl\w*|"
    r"tentativ\w*|reportedly|if|whether)\b", re.I)
_STOP = set("a an the of to in on at for and or is are was were be been by with as from that this it its "
            "which what who whom whose how when where why does do did their his her they he she".split())
_YES_NO_START = re.compile(r"^(is|are|was|were|does|do|did|can|could|has|have|had|will|would|should)\b", re.I)
_TEXT_REFERENCE = re.compile(r"\b(the text|this text|the passage|the description|the record|according to|"
                             r"above|below)\b", re.I)
_EXAMPLE_WRONG = {"hathor", "sobek", "khonsu", "one", "three", "four", "hold an offering", "pour water",
                  "hang it on a wall"}
_BAD_OPTION = re.compile(r"\b(all of the above|none of the above|both|all of these|none of these)\b", re.I)

# The small model gets ONE easy job per call. Code picks the evidence sentence, checks the answer,
# and validates everything; the model only writes a question, then three wrong answers.
LLM_QUESTION_PROMPT = """Write ONE quiz question about the sentence. The correct answer to your question must be exactly the given answer.
Rules:
- Start the question with the given opening words.
- Use only facts from the sentence. Do not include the answer in the question.
- Do not say "the sentence" or "the text". Reply with the question only.

Sentence: The bowl was dedicated to the god Khepri at Heliopolis.
Answer: Khepri
Opening words: Which god
Question: Which god was the bowl dedicated to?

Sentence: The bowl was dedicated to the god Khepri at Heliopolis.
Answer: Heliopolis
Opening words: Where
Question: Where was the bowl dedicated to Khepri?

Sentence: The jar has two handles and a painted blue rim.
Answer: two
Opening words: How many
Question: How many handles does the jar have?

Sentence: __SENTENCE__
Answer: __ANSWER__
Opening words: __STEM__
Question: __STEM__"""

LLM_QA_PROMPT = """Read the sentence about a museum artifact and write ONE quiz question about it.
Rules:
- The answer must be 1 to 5 words copied exactly from the sentence.
- Use only information from the sentence. The question must not contain the answer.
- Do not write a yes/no question. Do not say "the sentence" or "the text".
Reply with exactly two lines: Q: <question> and A: <answer>

Example
Sentence: The jar has two handles and a painted blue rim.
Q: How many handles does the jar have?
A: two

Sentence: __SENTENCE__
Q:"""

LLM_DISTRACTOR_PROMPT = """Here is a sentence about a museum artifact, a quiz question about it and its correct answer.
Write exactly 3 WRONG answers: the same kind of thing as the correct answer, similar length, and NOT mentioned in the sentence.
One per line, no numbers, no explanations.

Sentence: The offering table is dedicated to the god Thoth.
Question: Which god is the offering table dedicated to?
Correct answer: Thoth
Wrong answers:
Hathor
Sobek
Khonsu

Sentence: __SENTENCE__
Question: __QUESTION__
Correct answer: __ANSWER__
Wrong answers:"""

LLM_VERIFY_PROMPT = """Answer the question using ONLY the text. Reply with one letter (A, B, C or D).
Text: __EVIDENCE__
Question: __QUESTION__
__OPTIONS__
Answer:"""


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text or "").strip())
    return [p.strip() for p in parts if p.strip()]


_CREDIT = re.compile(r"\u00a9|\b(image|photo|photograph\w*|copyright|link to|courtesy|credit)\b", re.I)


def certain_sentences(description):
    """Sentences of the description without hedging language, long enough to carry a fact."""
    return [s for s in _split_sentences(description)
            if len(s) >= 25 and not _HEDGE_TEXT.search(s) and not _CREDIT.search(s) and "?" not in s]


def _excerpt(sentences, max_chars=LLM_MAX_CONTEXT_CHARS):
    out, size = [], 0
    for s in sentences:
        if size + len(s) > max_chars and out:
            break
        out.append(s)
        size += len(s) + 1
    return out


_NUMBER_WORDS = set("first second third fourth fifth sixth seventh eighth ninth tenth "
                    "two three four five six seven eight nine ten".split())


def unanchored_details(question, evidence, title=""):
    """Numbers, ordinals and capitalised names in the question must appear in the evidence/title.
    Catches questions that quietly change a detail ('second register' vs 'third register')."""
    haystack = norm_text(evidence) + " " + norm_text(title)
    missing = []
    for i, tok in enumerate(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", question)):
        low = tok.lower().strip("'-")
        if re.fullmatch(r"\d+", low) or low in _NUMBER_WORDS or (i > 0 and tok[:1].isupper()):
            if not contains_phrase(haystack, low):
                missing.append(tok)
    return missing


def _parse_llm_json(raw):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        obj = json.loads(blob)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_option(text):
    return re.sub(r"\s+", " ", str(text)).strip().strip("\"'").rstrip(".").strip()


def _content_tokens(text):
    return {t for t in re.findall(r"[a-z0-9]+", norm_text(text)) if t not in _STOP and len(t) > 2}


def _has_number(text):
    return bool(re.search(r"\d", text)) or any(w in _NUMBER_WORDS or w == "one"
                                               for w in re.findall(r"[a-z]+", text.casefold()))


def validate_llm_question(obj, sentences, full_description_norm, rag_text_norm, title):
    """Return (clean_question_dict | None, [reasons]). Pure function - no model needed."""
    reasons = []
    if not isinstance(obj, dict):
        return None, ["not_json"]
    question = re.sub(r"\s+", " ", str(obj.get("question", ""))).strip()
    answer = _clean_option(obj.get("correct_answer", ""))
    wrong = obj.get("wrong_answers")
    evidence = re.sub(r"\s+", " ", str(obj.get("evidence", ""))).strip()
    if not question or not answer or not isinstance(wrong, list) or not evidence:
        return None, ["missing_fields"]
    wrong = [_clean_option(w) for w in wrong]

    # ---- shape of question / options
    if not (20 <= len(question) <= 220) or not question.endswith("?"):
        reasons.append("bad_question_shape")
    if _YES_NO_START.match(question):
        reasons.append("yes_no_question")
    if _TEXT_REFERENCE.search(question):
        reasons.append("question_refers_to_text")
    if len(wrong) != 3 or any(not w for w in wrong):
        reasons.append("need_exactly_3_wrong_answers")
    options = [answer] + [w for w in wrong if w]
    n_answer = len(answer.split())
    if not (1 <= n_answer <= LLM_MAX_WORDS_ANSWER) or any(not (1 <= len(o.split()) <= LLM_MAX_WORDS_ANSWER) for o in options):
        reasons.append("option_length")
    normed = [norm_text(o) for o in options]
    if len(set(normed)) != len(normed):
        reasons.append("duplicate_options")
    for i in range(len(normed)):
        for j in range(len(normed)):
            if i != j and normed[i] and contains_phrase(normed[j], normed[i]):
                reasons.append("option_contains_option")
                break
        else:
            continue
        break
    if any(_BAD_OPTION.search(o) for o in options):
        reasons.append("generic_option")
    if len(wrong) == 3 and len({norm_text(w) for w in wrong} & _EXAMPLE_WRONG) >= 2:
        reasons.append("copied_prompt_example")
    if not _has_number(answer) and any(_has_number(w) for w in wrong):
        reasons.append("numeric_distractor_for_non_numeric_answer")
    if any(abs(len(w.split()) - n_answer) > 2 for w in wrong if w):
        reasons.append("distractor_length_mismatch")      # e.g. 2-word wrong answers for a 6-word answer

    # ---- grounding in the knowledge base record
    ev_norm = norm_text(evidence)
    sent_norms = [norm_text(s) for s in sentences]
    if len(ev_norm) < 20 or not any(ev_norm in s for s in sent_norms):
        reasons.append("evidence_not_verbatim_in_certain_sentences")
    elif ev_norm not in rag_text_norm:
        reasons.append("evidence_not_in_rag_text")
    if not contains_phrase(ev_norm, normed[0]):
        reasons.append("answer_not_in_evidence")
    if _HEDGE_TEXT.search(evidence):
        reasons.append("hedged_evidence")

    # ---- uniqueness of the correct answer
    for w in normed[1:]:
        if contains_phrase(full_description_norm, w):
            reasons.append("wrong_answer_appears_in_record")
            break
    q_norm = norm_text(question)
    if any(contains_phrase(q_norm, o) for o in normed):
        reasons.append("question_contains_option")
    if not (_content_tokens(question) & _content_tokens(evidence)):
        reasons.append("question_unrelated_to_evidence")
    if unanchored_details(question, evidence, title):
        reasons.append("question_detail_not_in_evidence")
    if re.search(r"[\u0600-\u06FF]", question + " ".join(options)):
        reasons.append("not_english")

    if reasons:
        return None, sorted(set(reasons))
    return {"question": question, "answer": answer, "wrong": wrong, "evidence": evidence}, []


ALLOW_LLM_DISTRACTORS = False      # LLM-written wrong answers were too easy to spot ("The garment hugs his legs")
USE_CORPUS_DISTRACTORS = True      # code-made distractors first, LLM-made only as a fallback
_CORPUS_INDEX = None
_SPAN_OK = re.compile(r"^[A-Za-z][A-Za-z'\-]*$")


def _tok(text):
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*|[.,;:()]", text)


def _corpus_index():
    """(word-2, word-1, span_length) -> [(artifact_id, span text)] over every certain description sentence."""
    global _CORPUS_INDEX
    if _CORPUS_INDEX is None:
        idx = {}
        for r in get_dataset().to_dict("records"):
            desc = clean_value(r.get("Description"))
            if not desc:
                continue
            for sent in certain_sentences(desc):
                t = _tok(sent)
                low = [x.lower() for x in t]
                for j in range(2, len(t)):
                    for k in range(1, 5):
                        if j + k > len(t) or not _SPAN_OK.match(t[j + k - 1]):
                            break
                        if k == 1 and not _SPAN_OK.match(t[j]):
                            break
                        idx.setdefault((low[j - 2], low[j - 1], k), []).append((r["Artifact ID"], " ".join(t[j:j + k])))
        _CORPUS_INDEX = idx
    return _CORPUS_INDEX


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def numeric_distractors(answer, desc_norm, rng, n=3):
    """'11th Dynasty' -> '9th Dynasty', '13th Dynasty', '14th Dynasty' (never a value found in the record)."""
    m = re.search(r"\b(\d+)(st|nd|rd|th)?\b", answer)
    if not m:
        return None
    val = int(m.group(1))
    steps = [1, 2, 3, 4, 5, 6] if val < 100 else [10, 25, 50, 100, 150, 200]
    values = sorted({val + sgn * d for d in steps for sgn in (1, -1) if val + sgn * d >= 1},
                    key=lambda v: abs(v - val))[:8]
    rng.shuffle(values)
    out = []
    for v in values:
        text = answer[:m.start()] + f"{v}{_ordinal(v) if m.group(2) else ''}" + answer[m.end():]
        if not contains_phrase(desc_norm, norm_text(text)):
            out.append(text)
        if len(out) == n:
            return out
    return None


def corpus_distractors(sentence, answer, aid, desc_norm, rng, n=3):
    """Wrong answers that other artifacts' descriptions use in the SAME context
    ('...dedicated to the god X' -> Osiris, Thoth, Horus), absent from this artifact's record."""
    st = _tok(sentence)
    a = [x.lower() for x in _tok(answer)]
    k = len(a)
    if not 1 <= k <= 4:
        return None
    pos = next((i for i in range(2, len(st) - k + 1) if [x.lower() for x in st[i:i + k]] == a), None)
    if pos is None:
        return None
    capital = st[pos][:1].isupper()
    counts = {}
    for other_id, span in _corpus_index().get((st[pos - 2].lower(), st[pos - 1].lower(), k), []):
        words = span.split()
        if other_id == aid or words[0].lower() in _STOP or words[-1].lower() in _STOP:
            continue
        if words[0][:1].isupper() != capital or norm_text(span) == norm_text(answer):
            continue
        if capital and not all(_CAP_TOKEN.match(w) for w in words):
            continue
        if _generic_span(words) or _too_similar(span, answer):
            continue
        if contains_phrase(desc_norm, norm_text(span)):
            continue
        counts[span] = counts.get(span, 0) + 1
    top = sorted(counts, key=lambda x: (-counts[x], x))[:10]
    rng.shuffle(top)
    chosen = []
    for span in top:
        if all(not contains_phrase(norm_text(c), norm_text(span)) and not contains_phrase(norm_text(span), norm_text(c))
               and not _too_similar(c, span) for c in chosen):
            chosen.append(span)
        if len(chosen) == n:
            return chosen
    return None


def _make_distractors(llm, sentence, question, answer, aid, desc_norm, rng, seed):
    """-> (wrong_answers, source). Code first; the LLM only as a fallback."""
    if USE_CORPUS_DISTRACTORS:
        w, src = code_distractors(sentence, answer, aid, desc_norm, rng)
        if w:
            return w, src
    if not ALLOW_LLM_DISTRACTORS:
        return [], "none_available"
    raw = llm(LLM_DISTRACTOR_PROMPT.replace("__SENTENCE__", sentence).replace("__QUESTION__", question)
              .replace("__ANSWER__", answer), max_new_tokens=45, seed=seed, greedy=False)
    return _parse_lines(raw, 3), "llm"


ANSWER_FIRST = True                # code chooses the answer span, the LLM only writes the question
_ORDINAL_WORDS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
                  "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth", "Seventeenth",
                  "Eighteenth", "Nineteenth", "Twentieth"]
_GENERIC_ANSWERS = set("lady house egyptian egyptians egypt king queen kingdom period dynasty book dead god goddess "
                       "museum metropolitan new old upper lower north south east west first second third fourth "
                       "pyramid kings step peabody delta valley egyptians "
                       "fifth sixth late early middle gallery figure statue coffin".split())
_CAP_TOKEN = re.compile(r"^(?:[A-Z][A-Za-z'\-]*|(?:el|al)-[A-Z][A-Za-z'\-]*|\d+(?:st|nd|rd|th))$")


def _generic_span(words):
    """True if a capitalised span is made of generic words (Lady, Kings, New ...) and is not a period name."""
    if norm_text(" ".join(words)) in V.PERIOD_BY_NAME:
        return False
    return any(w.lower() in _GENERIC_ANSWERS for w in words)


def _too_similar(a, b):
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio() >= 0.8


def candidate_answer_spans(sentence, title_norm=""):
    """Names, places, gods, periods and dynasty numbers: maximal capitalised runs inside the sentence."""
    t = _tok(sentence)
    out, i = [], 2
    while i < len(t):
        if not _CAP_TOKEN.match(t[i]) or t[i].endswith("'s"):
            i += 1
            continue
        j = i
        while j < len(t) and j - i < 4 and _CAP_TOKEN.match(t[j]) and not t[j].endswith("'s"):
            j += 1
        words = t[i:j]
        nxt = [x.lower() for x in t[j:j + 2]]
        part_of_longer_title = bool(nxt) and nxt[0] in {"of", "the"} and j + 1 < len(t) and t[j + 1][:1].isupper()
        low = [w.lower() for w in words]
        bad = (low[0] in _STOP or low[-1] in _STOP or part_of_longer_title
               or _generic_span(words)
               or (len(words) == 1 and re.match(r"^\d", words[0]))
               or contains_phrase(title_norm, norm_text(" ".join(words))))
        if not bad:
            out.append(" ".join(words))
        i = j + 1
    return out


def period_distractors(answer, desc_norm, rng, n=3):
    if norm_text(answer) not in V.PERIOD_BY_NAME:
        return None
    pool = [p[0] for p in V.PERIODS if p[0].casefold() != norm_text(answer)
            and not contains_phrase(desc_norm, p[0].casefold())]
    if len(pool) < n:
        return None
    names = [p[0] for p in V.PERIODS]
    pool.sort(key=lambda x: abs(names.index(x) - names.index(next(p[0] for p in V.PERIODS
                                                               if p[0].casefold() == norm_text(answer)))))
    window = pool[:6]
    rng.shuffle(window)
    return window[:n]


def ordinal_dynasty_distractors(answer, desc_norm, rng, n=3):
    m = re.fullmatch(r"([A-Z][a-z]+) Dynasty", answer)
    if not m or m.group(1) not in _ORDINAL_WORDS:
        return None
    k = _ORDINAL_WORDS.index(m.group(1))
    near = [x for x in sorted(range(len(_ORDINAL_WORDS)), key=lambda v: abs(v - k))[1:9]]
    rng.shuffle(near)
    out = [f"{_ORDINAL_WORDS[v]} Dynasty" for v in near
           if not contains_phrase(desc_norm, norm_text(f"{_ORDINAL_WORDS[v]} Dynasty"))]
    return out[:n] if len(out) >= n else None


_GODS = {g.lower() for g in """Amun Amun-Re Amen Amen-Ra Ra Re Re-Harakhty Ra-Harakhty Harakhty Osiris Isis Horus Hathor Thoth
Ptah Sakhmet Sekhmet Bastet Bes Anubis Nephthys Sobek Khonsu Mut Min Montu Nut Geb Shu Tefnut Atum Khnum Neith
Nefertum Wadjet Taweret Sokar Sokar-Osiris Maat Meretseger Selket Serqet Apis Mnevis Harpokrates Imhotep Khepri
Seth Set Hapy Wepwawet Anukis Satet Heqet Renenutet Meret Neit Ptah-Sokar-Osiris Amduat""".split()}
_PLACES = {g.lower() for g in """Thebes Memphis Karnak Abydos Giza Saqqara Dendera Amarna Luxor Heliopolis Bubastis Edfu Aswan
Elephantine Hermopolis Asyut Meir Lahun Lisht Dahshur Hierakonpolis Naqada Tanis Sais Buto Kerma Nubia Faiyum Fayum
Deir-el-Bahri Medinet-Habu Giza Akhmim Esna Kom-Ombo Philae Beni-Hasan Hawara Gurob Tuna-el-Gebel Zawiyet-el-Aryan""".split()}
_PERSON_TRIGGER = re.compile(
    r"\b(?:named|inscribed for|dedicated by|father|mother|wife|husband|daughter|son|brother|sister|"
    r"reign of|king|queen|pharaoh|priest|scribe|steward|overseer|owner),?\s+(?:the\s+)?"
    r"((?:[A-Z][A-Za-z'\-]*|(?:el|al)-[A-Z][A-Za-z'\-]*)(?:\s+(?:[A-Z][A-Za-z'\-]*|[IVX]+)){0,2})")
_PERSON_POOL = None
# names that denote the SAME god, or places inside one another -> never answer + distractor together
_ALIAS_GROUPS = [
    {"amun", "amen", "amun-re", "amen-ra", "amun-ra", "amen-re"},
    {"ra", "re", "re-harakhty", "ra-harakhty", "harakhty"},
    {"sakhmet", "sekhmet"}, {"neith", "neit"}, {"sokar", "sokar-osiris", "ptah-sokar-osiris", "osiris"},
    {"ptah", "ptah-sokar-osiris"}, {"harpokrates", "horus"}, {"serqet", "selket"}, {"maat", "meret"},
    {"thebes", "luxor", "karnak", "deir-el-bahri", "medinet-habu"},
    {"memphis", "saqqara", "giza", "dahshur"},
    {"faiyum", "fayum", "lahun", "hawara", "gurob"},
    {"aswan", "elephantine", "philae"}, {"nubia", "kerma"}, {"amarna", "hermopolis"},
]


def _name_parts(name):
    norm = {"re": "ra", "amen": "amun", "el": "", "al": ""}
    return {norm.get(x, x) for x in name.lower().split("-")} - {""}


def names_conflict(a, b):
    """Same god / nested place / one name contained in the other ('Amen-Ra' contains 'Ra')."""
    return bool(_alias_group(a) & _alias_group(b)) or bool(_name_parts(a) & _name_parts(b))


def _alias_group(name):
    low = name.lower()
    group = {low}
    for g in _ALIAS_GROUPS:
        if low in g:
            group |= g
    return group


def _person_pool():
    """Personal names used after a 'named / his wife / reign of ...' context somewhere in the museum texts."""
    global _PERSON_POOL
    if _PERSON_POOL is None:
        pool = {}
        for r in get_dataset().to_dict("records"):
            desc = clean_value(r.get("Description"))
            for sent in (certain_sentences(desc) if desc else []):
                for m in _PERSON_TRIGGER.finditer(sent):
                    span = m.group(1).strip()
                    words = span.split()
                    if (re.search(r"(wig|crown|dress|kilt|collar|style|type|form|garment)$", span.lower())
                            or span.lower() in _GODS or span.lower() in _PLACES or _generic_span(words)
                            or words[0].lower() in _STOP or span.endswith("'s")):
                        continue
                    pool[span] = pool.get(span, 0) + 1
        _PERSON_POOL = pool
    return _PERSON_POOL


def typed_distractors(answer, desc_norm, rng, n=3):
    """Same-TYPE wrong answers (god / place / person) from curated or corpus-derived pools."""
    low = answer.lower()
    if low in _GODS:
        kind, pool = "god", sorted(g for g in _GODS)
    elif low in _PLACES:
        kind, pool = "place", sorted(_PLACES)
    elif answer in _person_pool():
        kind, pool = "person", [x for x in _person_pool() if abs(len(x.split()) - len(answer.split())) <= 1]
    else:
        return None, None
    mine = _alias_group(answer)
    pool = [x for x in pool if not names_conflict(x, answer)
            and not any(contains_phrase(desc_norm, norm_text(a)) for a in _alias_group(x))
            and not _too_similar(x, answer)]
    if kind != "person":                 # curated lists are lowercase -> restore capitalisation
        pool = [x.title() if "-" not in x else "-".join(w.capitalize() for w in x.split("-")) for x in pool]
    rng.shuffle(pool)
    chosen = []
    for x in pool:
        if all(not _too_similar(x, c) and not names_conflict(x, c)
               and not contains_phrase(norm_text(c), norm_text(x))
               and not contains_phrase(norm_text(x), norm_text(c)) for c in chosen):
            chosen.append(x)
        if len(chosen) == n:
            return chosen, f"typed_{kind}"
    return None, None


def typed_code_distractors(answer, desc_norm, rng):
    """Guaranteed same-type wrong answers (period / dynasty / number / god / place / person) or (None, None)."""
    for fn, name in ((period_distractors, "period_vocabulary"), (ordinal_dynasty_distractors, "ordinal_variants"),
                     (numeric_distractors, "numeric_variants")):
        w = fn(answer, desc_norm, rng)
        if w:
            return w, name
    return typed_distractors(answer, desc_norm, rng)


def code_distractors(sentence, answer, aid, desc_norm, rng):
    """Wrong answers made by code only -> (list, source) or (None, None). Typed first, then same-context."""
    w, src = typed_code_distractors(answer, desc_norm, rng)
    if w:
        return w, src
    w = corpus_distractors(sentence, answer, aid, desc_norm, rng)
    return (w, "corpus_same_context") if w else (None, None)


def _parse_qa(raw):
    """'Q: ...\nA: ...' (the prompt already ends with 'Q:') -> (question, answer) or (None, None)."""
    text = "Q:" + raw if raw and not re.match(r"\s*(Q|Question)\s*:", raw, re.I) else (raw or "")
    q = re.search(r"(?:Q|Question)\s*:\s*(.+)", text, re.I)
    a = re.search(r"(?:A|Answer)\s*:\s*(.+)", text, re.I)
    if not q or not a:
        return None, None
    return q.group(1).strip(), a.group(1).strip()


def _parse_lines(raw, n=3):
    lines = [re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", ln).strip() for ln in (raw or "").splitlines()]
    return [ln for ln in lines if ln][:n]


def _sentences_for_question(description, skip_evidence, rng):
    certain = certain_sentences(description)
    pool = [s for s in certain if 6 <= len(s.split()) <= 40 and norm_text(s) not in skip_evidence]
    rng.shuffle(pool)
    return certain, pool


_STEMS = {"typed_god": "Which god", "typed_place": "Where", "typed_person": "Who",
          "period_vocabulary": "Which period", "ordinal_variants": "Which dynasty"}
_KIND = {"typed_god": "god", "typed_place": "place", "typed_person": "person", "period_vocabulary": "period",
         "ordinal_variants": "dynasty", "numeric_variants": "number"}


def question_stem(dsource, answer):
    if dsource == "numeric_variants":
        return "Which dynasty" if "dynasty" in answer.lower() else "What"
    return _STEMS.get(dsource, "What")


def question_type_ok(question, dsource, answer):
    """The question word must fit the kind of answer ('How many ...' can never have the answer 'Seth')."""
    q = question.lower()
    kind = _KIND.get(dsource)
    if re.search(r"\bhow (many|much|long|tall|old|big|far|high|large)\b", q) and not _has_number(answer):
        return False
    if kind is None:
        return True
    if re.search(r"\bhow (did|does|do|is|are|was|were|can|could|would)\b", q):
        return False
    if re.search(r"\bhow (many|much|long|tall|old|big|far|high|large)\b", q) and kind != "number":
        return False
    if re.search(r"\bwhere\b", q) and kind != "place":
        return False
    if re.search(r"\bwho(se)?\b", q) and kind not in ("person", "god"):
        return False
    if re.search(r"\bwhen\b", q) and kind not in ("period", "dynasty", "number"):
        return False
    return True


def _first_question_line(raw):
    for line in (raw or "").splitlines():
        line = re.sub(r"^\s*(?:Question|Q)\s*:\s*", "", line.strip(), flags=re.I).strip().strip('"')
        if line:
            return line
    return ""


def build_llm_question(row, level, rng, seed=None, skip_evidence=(), sentences_to_try=3):
    """Pick a hedge-free evidence sentence in code; get a question + 3 wrong answers; validate.
    answer-first (default): CODE picks the answer span + wrong answers, the LLM only writes the question.
    fallback: the LLM proposes question + answer from the sentence (answer must be copied from it).
    -> (internal_question_dict | None, reasons)"""
    llm = _llm()
    if llm is None:
        return None, ["llm_unavailable"]
    description = clean_value(row.get("Description"))
    if not description:
        return None, ["no_description"]
    certain, pool = _sentences_for_question(description, set(skip_evidence), rng)
    if not pool:
        return None, ["no_certain_sentences"]
    aid = row["Artifact ID"]
    title = clean_value(row.get("Title")) or "this artifact"
    desc_norm = norm_text(description)
    rag_norm = norm_text(clean_value(row.get("RAG Text")) or "")
    title_norm = norm_text(title)

    # sentences that contain a code-checkable answer go first
    plans = []
    for sentence in pool[:10]:
        options = []
        if ANSWER_FIRST and USE_CORPUS_DISTRACTORS:
            for span in candidate_answer_spans(sentence, title_norm):
                w, src = typed_code_distractors(span, desc_norm, rng)      # only guaranteed same-type answers
                if w:
                    options.append((span, w, src))
        plans.append((sentence, options))
    plans.sort(key=lambda p: 0 if p[1] else 1)

    def finish(sentence, question, answer, wrong, dsource, extra_checks):
        parsed, reasons = validate_llm_question(
            {"question": question, "correct_answer": answer, "wrong_answers": wrong, "evidence": sentence},
            certain, desc_norm, rag_norm, title)
        if parsed is None:
            return None, reasons
        if not question_type_ok(parsed["question"], dsource, parsed["answer"]):
            return None, ["question_type_mismatch"]
        if extra_checks and len(_content_tokens(parsed["question"]) & (_content_tokens(sentence)
                                                                     - _content_tokens(answer))) < 2:
            return None, ["question_not_about_the_sentence"]
        options = [parsed["answer"]] + parsed["wrong"]
        rng.shuffle(options)
        checks = ["evidence_chosen_by_code", "verbatim_evidence", "answer_in_evidence", "no_leak",
                  "details_anchored_in_evidence", "distractors_absent_from_record", "no_hedging",
                  "question_type_fits_answer", f"distractors_{dsource}"]
        if extra_checks:
            checks.append("answer_chosen_by_code")
        if LLM_VERIFY:
            letters = "ABCD"
            vprompt = (LLM_VERIFY_PROMPT.replace("__EVIDENCE__", sentence)
                       .replace("__QUESTION__", parsed["question"])
                       .replace("__OPTIONS__", "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options))))
            vraw = llm(vprompt, max_new_tokens=4, seed=seed, greedy=True)
            m = re.search(r"\b([ABCD])\b", vraw or "")
            if not m or options[letters.index(m.group(1))] != parsed["answer"]:
                return None, ["verification_failed"]
            checks.append("llm_verification_pass")
        return {"question": parsed["question"], "options": options, "answer": parsed["answer"],
                "evidence": sentence, "checks_passed": checks}, []

    all_reasons = []
    for sentence, options in plans[:sentences_to_try]:
        try:
            if options:                                           # ---- answer-first
                answer, wrong, dsource = rng.choice(options)
                stem = question_stem(dsource, answer)
                raw = llm(LLM_QUESTION_PROMPT.replace("__SENTENCE__", sentence).replace("__ANSWER__", answer)
                          .replace("__STEM__", stem), max_new_tokens=50, seed=seed, greedy=False)
                question = _first_question_line(raw)
                if question and not question.lower().startswith(stem.lower()):
                    if question[0] in "'\u2019":                 # "Which god" + "'s role ...?"
                        question = stem + question
                    elif question[0].islower():                  # normal case: the model continues the opening
                        question = f"{stem} {question}"
                    else:                                        # it ignored the opening words
                        all_reasons.append("question_ignored_opening")
                        continue
                if not question:
                    all_reasons.append("question_parse_failed")
                    continue
                built, reasons = finish(sentence, question, answer, wrong, dsource, True)
            else:                                                 # ---- model proposes answer
                raw = llm(LLM_QA_PROMPT.replace("__SENTENCE__", sentence), max_new_tokens=70, seed=seed,
                          greedy=False)
                question, answer = _parse_qa(raw)
                answer = _clean_option(answer) if answer else answer
                if not question or not answer:
                    all_reasons.append("qa_parse_failed")
                    continue
                if not contains_phrase(norm_text(sentence), norm_text(answer)):
                    all_reasons.append("answer_not_in_evidence")
                    continue
                if contains_phrase(norm_text(question), norm_text(answer)):
                    all_reasons.append("question_contains_option")
                    continue
                wrong, dsource = _make_distractors(llm, sentence, question, answer, aid, desc_norm, rng, seed)
                if not wrong:
                    all_reasons.append("no_code_distractors")
                    continue
                built, reasons = finish(sentence, question, answer, wrong, dsource, False)
        except Exception as exc:
            return None, [f"llm_error:{type(exc).__name__}"]
        if built is not None:
            return built, []
        all_reasons.extend(reasons)
    return None, all_reasons or ["no_valid_question"]


# ----------------------------------------------------------------------------
# (B) Metadata questions - wording phrased by the LLM (validated) or template
# ----------------------------------------------------------------------------
def _fallback_question(title, field, language):
    return TEMPLATES[language][field].format(title=title)


def generate_question_text(title, field, language="en", forbidden=()):
    """Phrase a metadata question. Returns (text, generator).
    The LLM never sees the answer; its wording is rejected unless it mentions the exact title,
    is clearly about `field`, and contains none of the `forbidden` strings (answer + options)."""
    llm = _llm()
    if llm is None:
        return _fallback_question(title, field, language), "template_fallback"
    try:
        language_name = "Arabic" if language == "ar" else "English"
        field_names = {"Class": "what type of object it is", "Material": "its main material",
                       "Period": "its historical period", "Dynasty": "its dynasty"}
        prompt = (
            "Write exactly one short quiz question. Do not answer it, do not list options.\n"
            f"Language: {language_name}.\nCopy the artifact title exactly inside single quotes: '{title}'.\n"
            f"The question must ask about {field_names[field]}.\nReturn only the question.")
        generated = (llm(prompt, max_new_tokens=80, seed=None, greedy=True) or "").strip()
        generated = generated.splitlines()[0].strip().strip('"') if generated else ""
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", generated))
        if not generated or len(generated) > 240 or (language == "ar") != has_arabic:
            return _fallback_question(title, field, language), "template_fallback_invalid_output"
        if norm_text(title) not in norm_text(generated):
            return _fallback_question(title, field, language), "template_fallback_title_missing"
        if not re.search(_FIELD_KEYWORDS[language][field], generated, re.I):
            return _fallback_question(title, field, language), "template_fallback_off_topic"
        rest = norm_text(generated).replace(norm_text(title), " ")
        if any(contains_phrase(rest, norm_text(f)) for f in forbidden if f):
            return _fallback_question(title, field, language), "template_fallback_leaks_option"
        if not generated.endswith(("?", "؟")):
            generated += "؟" if language == "ar" else "?"
        return generated, "qwen_phrasing"
    except Exception:
        return _fallback_question(title, field, language), "template_fallback_generation_error"


# ----------------------------------------------------------------------------
# Session / question flow
# ----------------------------------------------------------------------------
def start_quiz_session(visited_artifact_ids, language="en", voice_used=False, seed=None,
                       distractor_scope="dataset", question_mode="llm"):
    """Start a quiz. `visited_artifact_ids` must be the visitor's real visited IDs."""
    if not visited_artifact_ids:
        raise ValueError("Actual visited artifact IDs must be provided.")
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Use one of {sorted(SUPPORTED_LANGUAGES)}.")
    if distractor_scope not in {"dataset", "visited"}:
        raise ValueError("distractor_scope must be 'dataset' or 'visited'.")
    if question_mode not in {"llm", "metadata"}:
        raise ValueError("question_mode must be 'llm' or 'metadata'.")

    df = get_dataset()
    requested = list(dict.fromkeys(
        clean_value(item) for item in visited_artifact_ids if clean_value(item)))
    found = set(df["Artifact ID"].dropna())
    artifact_ids = [item for item in requested if item in found]
    missing = [item for item in requested if item not in found]
    if not artifact_ids:
        raise ValueError("None of the supplied artifact IDs exist in the dataset.")

    session = {
        "session_id": uuid.uuid4().hex[:12],
        "language": language,
        "voice_used": bool(voice_used),
        "seed": seed if seed is not None else random.SystemRandom().randrange(2**31),
        "distractor_scope": distractor_scope,
        "question_mode": question_mode,
        "artifact_ids": artifact_ids,
        "missing_ids": missing,
        "level": "easy",
        "questions_answered": 0,
        "correct_answers": 0,
        "consecutive_correct": 0,
        "asked": [],            # [[artifact_id, field], ...]; LLM questions use field "Description"
        "skipped_facts": [],    # [artifact_id, field, reason] facts never asked (audit trail)
        "llm_rejections": [],   # [artifact_id, [reasons]] LLM questions that failed validation
        "asked_evidence": [],   # normalised evidence sentences already used (no repeated LLM question)
        "pending": None,
        "finished": False,
        "events": [],
    }
    _log(session, "quiz_started", artifact_count=len(artifact_ids), missing_ids=missing,
         difficulty=session["level"], question_mode=question_mode)
    return session


def _rng(session):
    return random.Random(f"{session['seed']}:{len(session['asked'])}")


def _public(question):
    return {k: v for k, v in question.items() if not k.startswith("_")}


def _candidate_facts(session):
    """All (row, field, fact) that are certain, unambiguous, validated and not asked yet."""
    df = get_dataset()
    asked = {tuple(item) for item in session["asked"]}
    rows = df[df["Artifact ID"].isin(session["artifact_ids"])]
    facts = []
    for row in rows.to_dict("records"):
        for field in ALL_FIELDS:
            if (row["Artifact ID"], field) in asked:
                continue
            fact, reason = extract_fact(row, field)
            if fact is None:
                if reason != "missing":
                    note = [row["Artifact ID"], field, reason]
                    if note not in session["skipped_facts"]:
                        session["skipped_facts"].append(note)
                continue
            if session["language"] == "ar" and not fact["label_ar"]:
                continue
            facts.append((row, field, fact))
    return facts


def _serve(session, question, field):
    session["pending"] = question
    session["asked"].append([question["artifact_id"], field])
    _log(session, "question_served", question_id=question["question_id"],
         artifact_id=question["artifact_id"], field=question["field"],
         difficulty=question["difficulty"], question_source=question["question_source"],
         question_generator=question["question_generator"],
         distractor_scope_used=question["distractor_scope_used"])
    return _public(question)


def _try_llm_question(session, rng, df):
    """Try up to LLM_MAX_ATTEMPTS visited artifacts. Returns a public question or None."""
    if session["language"] != "en":
        return None                      # descriptions are English; Arabic uses metadata questions
    if _llm() is None:
        return None
    counts = {}
    for aid, field in session["asked"]:
        if field == "Description":
            counts[aid] = counts.get(aid, 0) + 1
    rows = [r for r in df[df["Artifact ID"].isin(session["artifact_ids"])].to_dict("records")
            if clean_value(r.get("Description")) and counts.get(r["Artifact ID"], 0) < LLM_MAX_PER_ARTIFACT]
    rng.shuffle(rows)
    rows.sort(key=lambda r: counts.get(r["Artifact ID"], 0))     # spread over artifacts first
    for row in rows[:LLM_MAX_ATTEMPTS]:
        built, reasons = build_llm_question(row, session["level"], rng, seed=rng.randrange(2**31),
                                            skip_evidence=session["asked_evidence"])
        if built is not None and norm_text(built["evidence"]) in session["asked_evidence"]:
            built, reasons = None, ["duplicate_evidence"]
        if built is None:
            session["llm_rejections"].append([row["Artifact ID"], reasons])
            _log(session, "llm_question_rejected", artifact_id=row["Artifact ID"], reasons=reasons)
            continue
        title = clean_value(row.get("Title")) or "this artifact"
        question = {
            "question_id": f"{session['session_id']}-{len(session['asked']) + 1}",
            "status": "ok",
            "artifact_id": row["Artifact ID"],
            "title": title,
            "field": "Description",
            "question": built["question"],
            "question_source": "llm_description",
            "question_generator": "qwen_description",
            "options": built["options"],
            "difficulty": session["level"],
            "language": "en",
            "distractor_scope_used": "llm",
            "_correct_answer": built["answer"],
            "_source": {
                "artifact_id": row["Artifact ID"],
                "source_url": clean_value(row.get("Source URL")),
                "evidence": built["evidence"],
                "fact_field": "Description",
                "fact_value": built["answer"],
                "validated_against": "RAG Text (Description)",
                "checks_passed": built["checks_passed"],
            },
        }
        session["asked_evidence"].append(norm_text(built["evidence"]))
        return _serve(session, question, "Description")
    return None


def _metadata_question(session, rng, df):
    level = session["level"]
    language = session["language"]
    facts = _candidate_facts(session)
    if not facts:
        return None
    preferred = QUESTION_FIELDS[level]
    rank = lambda f: preferred.index(f[1]) if f[1] in preferred else len(preferred)   # noqa: E731
    facts.sort(key=rank)
    best = rank(facts[0])
    tier = [f for f in facts if rank(f) == best]
    rng.shuffle(tier)
    rest = [f for f in facts if all(f is not t for t in tier)]
    visited_rows = df[df["Artifact ID"].isin(session["artifact_ids"])].to_dict("records")

    for row, field, fact in tier + rest:
        visited_labels = set()
        for r in visited_rows:
            f, _ = canonicalize(field, r.get(FIELD_COLUMN[field]))
            if f:
                visited_labels.add(f["label"])
        distractors, scope_used = pick_distractors(
            field, fact, level, rng, visited_labels, session["distractor_scope"])
        if distractors is None:
            continue
        options = [label_for(f, language) for f in distractors] + [label_for(fact, language)]
        if len({normalize_answer(o) for o in options}) != 4:
            continue
        rng.shuffle(options)
        title = clean_value(row.get("Title")) or "this artifact"
        text, generator = (generate_question_text(title, field, language, forbidden=options)
                           if session["question_mode"] == "llm"
                           else (_fallback_question(title, field, language), "template"))
        question = {
            "question_id": f"{session['session_id']}-{len(session['asked']) + 1}",
            "status": "ok",
            "artifact_id": row["Artifact ID"],
            "title": title,
            "field": field,
            "question": text,
            "question_source": "metadata",
            "question_generator": generator,
            "options": options,
            "difficulty": level,
            "language": language,
            "distractor_scope_used": scope_used,
            "_correct_answer": label_for(fact, language),
            "_source": {
                "artifact_id": row["Artifact ID"],
                "source_url": clean_value(row.get("Source URL")),
                "evidence": f"{field}: {fact['raw']}",
                "fact_field": field,
                "fact_value": fact["raw"],
                "canonical_answer": fact["label"],
                "validated_against": "RAG Text",
            },
        }
        return _serve(session, question, field)
    return None


def generate_quiz_question(session):
    """Next question (public view, without the answer) or {"status": "no_more_questions"}."""
    if session["finished"]:
        return {"status": "quiz_finished"}
    if session["pending"] is not None:
        return _public(session["pending"])       # an unanswered question stays the same
    df = get_dataset()
    rng = _rng(session)
    question = None
    if session["question_mode"] == "llm":
        question = _try_llm_question(session, rng, df)
    if question is None:
        question = _metadata_question(session, rng, df)
    if question is None:
        return {"status": "no_more_questions", "summary": get_quiz_summary(session)}
    return question


def check_answer(question, user_answer):
    """Server-side check against a stored question (includes `_correct_answer`)."""
    if question is None:
        return False
    options = [normalize_answer(o) for o in question["options"]]
    value = normalize_answer(user_answer)
    return (len(options) == len(set(options)) and value in options
            and value == normalize_answer(question["_correct_answer"]))


def submit_answer(session, user_answer):
    question = session["pending"]
    if question is None:
        return {"valid_answer": False, "message": "No pending question. Call generate_quiz_question first."}
    options = {normalize_answer(o) for o in question["options"]}
    if normalize_answer(user_answer) not in options:
        return {"valid_answer": False, "message": "Invalid answer. Choose one of the options."}

    is_correct = check_answer(question, user_answer)
    session["questions_answered"] += 1
    previous_level = session["level"]
    index = LEVELS.index(previous_level)
    if is_correct:
        session["correct_answers"] += 1
        session["consecutive_correct"] += 1
        if session["consecutive_correct"] >= 2:
            session["level"] = LEVELS[min(index + 1, len(LEVELS) - 1)]
            session["consecutive_correct"] = 0
    else:
        session["consecutive_correct"] = 0
        session["level"] = LEVELS[max(index - 1, 0)]

    session["pending"] = None
    _log(session, "answer_submitted", question_id=question["question_id"],
         artifact_id=question["artifact_id"], field=question["field"],
         question_source=question["question_source"],
         is_correct=is_correct, difficulty=previous_level,
         score=session["correct_answers"], answered=session["questions_answered"])
    return {
        "valid_answer": True,
        "is_correct": is_correct,
        "correct_answer": question["_correct_answer"],
        "source": question["_source"],
        "previous_level": previous_level,
        "current_level": session["level"],
    }


def get_quiz_summary(session):
    total = session["questions_answered"]
    correct = session["correct_answers"]
    by_source = {}
    for e in session["events"]:
        if e["event"] == "question_served":
            by_source[e["question_source"]] = by_source.get(e["question_source"], 0) + 1
    return {
        "session_id": session["session_id"],
        "questions_answered": total,
        "correct_answers": correct,
        "accuracy_percentage": round(correct / total * 100, 2) if total else 0.0,
        "current_level": session["level"],
        "finished": session["finished"],
        "questions_by_source": by_source,
        "llm_questions_rejected": len(session["llm_rejections"]),
    }


def finish_quiz(session):
    if not session["finished"]:
        session["finished"] = True
        session["pending"] = None
        summary = get_quiz_summary(session)
        _log(session, "quiz_completed", score=summary["correct_answers"],
             answered=summary["questions_answered"],
             accuracy_percentage=summary["accuracy_percentage"])
    return get_quiz_summary(session)


def display_question(question):
    print(f"\nDifficulty: {question['difficulty']}   Source: {question['question_source']}")
    print(f"Artifact: {question['title']} ({question['artifact_id']})")
    print(question["question"])
    for index, option in enumerate(question["options"], start=1):
        print(f"{index}. {option}")


if __name__ == "__main__":
    import sys
    if "--no-llm" in sys.argv:
        set_llm_backend(False)
    print("LLM status:", get_llm_status())
    # Example visited list - replace with the real visited IDs from Member 3 / the app.
    demo_ids = ["MET_2021.41.2", "MET_2021.41.15", "MET_2021.41.18", "MET_2021.41.106",
                "MET_2021.41.98", "CMA_1914.561"]
    demo = start_quiz_session(demo_ids, language="en", seed=7)
    print("Unknown IDs (not in dataset):", demo["missing_ids"])
    for _ in range(4):
        q = generate_quiz_question(demo)
        if q.get("status") != "ok":
            print(q)
            break
        display_question(q)
        result = submit_answer(demo, q["options"][0])
        print("-> correct:", result["is_correct"], "| answer:", result["correct_answer"],
              "| evidence:", result["source"]["evidence"][:90])
    print("Summary:", finish_quiz(demo))
    print("LLM questions rejected (artifact, reasons):")
    for aid, why in demo["llm_rejections"]:
        print("  ", aid, why)
