# Quiz evaluation (module v2.0.0)

Produced by `python evaluate_quiz.py --llm mock --n 200 --sessions 400`.

## A. Every artifact, deterministic questions (real result)
* 1,418 artifacts × 3 difficulty levels × EN/AR → **20,130 questions served and audited**.
* Independent checker: exactly 4 unique options, exactly 1 correct, **no overlapping options**
  (year intervals / dynasty numbers / material groups recomputed separately), no uncertain fact, fact present in
  `RAG Text`, answer not in title, period consistent with `Date`.
* **Failures: 0.**
* Askable facts: 3,355 across 1,385 artifacts; 33 artifacts have no askable metadata fact.

| Skipped fact (field:reason) | Count |
|---|---:|
| Class: answer in title | 1146 |
| Dynasty: missing | 372 |
| Period: ambiguous (ranges such as "Late Period–Ptolemaic Period") | 250 |
| Dynasty: ambiguous (e.g. "Dynasty 12, early") | 184 |
| Material: answer in title | 93 |
| Material: uncertain | 72 |
| Period: missing | 67 |
| Material: ambiguous / composite | 65 |
| Dynasty / Period: uncertain | 32 / 32 |
| Period: answer in title | 4 |

For comparison, a sample of 300 questions from the previous version contained 4 questions with two
overlapping options (3 with several "wood" options, 1 with several copper/bronze options).

## B. Random visitor sessions (real result)
400 sessions, 1–8 visited artifacts (both museums, all classes, EN/AR): 3,886 questions,
**0** questions about unvisited artifacts, **0** repeats, **0** duplicate options.

## C. LLM-written questions
**Not measured on the real model yet.** The run above used `mock_llm_for_tests.py` (a rule-based test
double, not an AI), so its numbers (72% accepted; rejections: question too long 45, refers to text 6,
wrong answer in record 4, no certain sentence 4) only show that the validators work.
The 27 unit tests also confirm every bad-output case is rejected (invented evidence, answer leak, wrong option
present in the record, hedged text, garbage JSON, failed verification, yes/no).

**To do on a machine with Hugging Face access:** `python evaluate_quiz.py --llm qwen --n 100`, then fill in
`evaluation/llm_review_sheet_qwen.csv` by hand and report acceptance rate + % judged correct.
