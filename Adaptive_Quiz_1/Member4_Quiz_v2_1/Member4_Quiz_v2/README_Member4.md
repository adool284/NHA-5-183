# Member 4 — Adaptive Quiz (v2.0.0)

Questions are generated **only from artifacts the visitor visited**, and every fact is checked against
that artifact's `RAG Text` record (same trusted record Member 1's RAG uses).

## What changed (feedback → fix)

| Feedback | What was done |
|---|---|
| Options with overlapping historical periods | Every period / dynasty / material is mapped to ONE canonical label with a conflict key (`quiz_vocab.py`). Distractors must have a disjoint key, so `New Kingdom` vs `New Kingdom, Ramesside`, `Dynasty 19–20` vs `Dynasty 20`, `Wood` vs `Sycomore wood`, `Bronze` vs `Cupreous metal`, `Granite` vs `Granodiorite` can never sit in the same question. Periods use disjoint year spans. |
| Drop uncertain info (materials etc.) | Anything with `?`, `or`, `probably/possibly/likely`, `later/earlier`, `per card`, composite objects ("limestone table; greywacke statuette"), or values that cannot be mapped to one clear label is skipped. Skipped facts are recorded in `session["skipped_facts"]` with the reason. Period is also cross-checked against the `Date` column. Questions whose answer is written in the title ("Statuette of…") are skipped. |
| Test on more than 4 artifacts | `evaluate_quiz.py` audits **all 1,418 artifacts** with an independent checker, plus 400 random visitor sessions. 41 unit tests in `test_quiz_module.py`. |
| The model should write the questions | New `llm_description` question type: the LLM (Qwen2.5-0.5B-Instruct) writes the question from a sentence of the description of a visited artifact. Code picks the answer / evidence and builds the wrong answers. See below. |
| Make sure Q&A are correct | Code validators + a second LLM "answer it back" pass + a human review sheet (below). Real-model results are in `evaluation/EVALUATION_REPORT.md`. |

## How an LLM-written question is made and accepted

Real-model testing showed Qwen-0.5B cannot write question + answer + 3 wrong answers + evidence in one JSON
(3% accepted), and that it writes "How many…" questions whatever the answer is. So the work is split into small
steps and **code does everything that can be checked**:

1. **Code** picks one hedge-free sentence of the description as the evidence (credit lines like "photo by" are never used).
2. **Answer-first (default):** **code** picks the answer from types it can guarantee: a god, place, person, period,
   dynasty or number found in the sentence, and builds the 3 wrong answers of the SAME type from curated /
   corpus-derived pools (aliases such as Amun/Amen-Ra or Thebes/Karnak are never mixed; nothing mentioned in this
   artifact's record is used). The **LLM only writes the question** for that answer, and its opening words are
   fixed by the answer type ("Which god…", "Where…", "Who…", "Which period…", "Which dynasty…"); a question whose
   wording does not fit the answer (e.g. "How many…" for a god) is rejected.
3. **Fallback** (sentence has no such answer): the LLM proposes question + answer (copied from the sentence);
   wrong answers come from typed pools or same-context spans in other descriptions. LLM-written wrong answers are
   OFF by default (`ALLOW_LLM_DISTRACTORS`) because they copied the question's wording and were easy to spot.
4. **Code validates** (`validate_llm_question`): answer inside the sentence; no answer/option inside the question;
   not yes/no; no "the text"; options unique, similar length, none contains another; no number-like wrong answers
   for a non-numeric answer; **no wrong answer appears anywhere in the artifact record**; numbers / ordinals /
   names in the question must appear in the evidence (stops "second register" vs "third register").
5. **LLM verification:** given only the evidence sentence, the model must pick the correct option.

Anything that fails is dropped (logged as `llm_question_rejected`) and a validated metadata question is served.
The quiz never serves an unvalidated model question, but the checks cannot prove a question is *good* —
hence the human review sheet.

Metadata questions (Class / Material / Period / Dynasty) keep the answer from the record; the LLM only phrases
the wording, and the wording is rejected unless it contains the exact title, is about the right field and
leaks no option.

## Run
```bash
pip install -r requirements.txt
python -m unittest -v test_quiz_module.py        # 41 tests, no model needed
python quiz_module.py                            # demo (downloads Qwen2.5-0.5B on first run)
python quiz_module.py --no-llm                   # demo without the model
python evaluate_quiz.py --llm none               # deterministic audit of all 1,418 artifacts (~2 min)
python evaluate_quiz.py --llm qwen --n 30 --sessions 100 --skip-audit   # REAL model, ~10 min on CPU
```
`evaluate_quiz.py --llm qwen` writes `evaluation/llm_review_sheet_qwen.csv` (one row per accepted question with
evidence + source URL and empty `human_ok_Y_N` / `human_notes` columns) and `llm_rejected_qwen.csv`.
Fill in the Y/N column by hand: that is the real "question correctness" number. (Part B of the script checks
session logic with the mock; only Part C uses the real model.)

## Usage (for Member 5)
```python
import quiz_module as quiz
s = quiz.start_quiz_session(visited_ids, language="en", voice_used=False, seed=None,
                            distractor_scope="dataset", question_mode="llm")   # or "metadata"
q = quiz.generate_quiz_question(s)     # public dict, no answer inside
r = quiz.submit_answer(s, q["options"][0])   # {is_correct, correct_answer, source{evidence, source_url,...}}
quiz.finish_quiz(s)                    # summary + quiz_completed event
```
Question dict fields: `question_id, artifact_id, title, field, question, options, difficulty, language,
question_source (llm_description|metadata), question_generator, distractor_scope_used`.
Event fields for Member 6 (`s["events"]`): `quiz_started, question_served, llm_question_rejected,
answer_submitted, quiz_completed` with `session_id, language, voice_used, module_version, question_source`.
`get_llm_status()` reports whether Qwen loaded.

## Known limits (be upfront in the demo)
* **Model-written questions are a minority.** With Qwen2.5-0.5B only about 20% of attempted artifacts yield an
  accepted question (n=30 run). In a manual check of 6 accepted questions, 4 were sound; one had the roles
  swapped ("which god united with his father" -> Osiris, while the sentence says it is Re) — code checks and the
  model's own verification pass cannot catch that kind of error. Everything else falls back to metadata
  questions, which are fully rule-checked. For a zero-risk demo use `question_mode="metadata"`.
* Model-written questions are limited to answers code can type-check: gods, places, people, periods, dynasties,
  numbers. Wrong answers are always code-made (LLM-written wrong answers are off: `ALLOW_LLM_DISTRACTORS`).
* Description-based questions are **English only** (descriptions are English; Qwen-0.5B Arabic is unreliable).
  Arabic sessions get metadata questions with Arabic option labels (vocabulary translated in `quiz_vocab.py` —
  have an Arabic speaker review them).
* Only 662 / 1,418 artifacts have a description; the rest get metadata questions only.
* "Wrong answer is absent from the record" does not prove it is false; the person judging the review sheet is the
  final check.
* A larger model (`QUESTION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"`) was **not tested**; it may raise the
  acceptance rate. Failures only cost fallbacks, never unvalidated questions.
* Material answers are canonical families (e.g. "Copper alloy (bronze)"), Period answers are the main period
  ("New Kingdom" for "New Kingdom, Ramesside"). Period spans in `quiz_vocab.py` are the conventional chronology.
* Member 3's `member3_handoff.json` `recommended_artifact_ids` are `EG_PA_*`, which do not exist in the shared
  dataset (`MET_*`/`CMA_*`). They are reported in `missing_ids` and ignored. Ask Member 3 for real IDs.

## Files
`quiz_module.py` (API) · `quiz_vocab.py` (periods, materials, Arabic labels) · `combined_rag_dataset.csv` ·
`test_quiz_module.py` · `evaluate_quiz.py` · `mock_llm_for_tests.py` (test double, NOT an AI model) ·
`evaluation/` (results)
