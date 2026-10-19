# Quiz evaluation (module v2.0.0)

Parts A and B: `python evaluate_quiz.py` (Part B uses the mock to test session logic). Part C: real Qwen2.5-0.5B-Instruct on the author's machine.

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

## C. LLM-written questions (real model: Qwen2.5-0.5B-Instruct, 30 random artifacts with a description)

Design iterations measured on the same 30 artifacts: one-shot JSON 3.3% -> split steps 23.3% -> code-made wrong
answers 10% -> forced answer-type opening 33% (but manual review showed most accepted questions were wrong:
"How many gods…" with answer "Seth") -> **final version 20%**.

Final run: 30 attempted, **6 accepted (20%)**, 24 rejected. Main rejection reasons: answer not copied from the
sentence 19, no code-made wrong answers available 15, question contains an option 10, Q/A format not followed 8,
question ignored the required opening 7.

Manual review of the 6 accepted questions (first-pass review by the assistant; Member 4 to confirm in
`llm_review_sheet_qwen.csv`):

| Question | Answer | Verdict |
|---|---|---|
| Which god is associated with the union of Lower and Upper Egypt? | Horus | OK (loose wording) |
| What is the name of the woman's mummy? | Nefer | OK |
| Which god was dedicated to by Hatshepsut? | Amun-Re | **Not OK** (awkward; the sanctuary was dedicated to Amun-Re) |
| Which god is shown on one side of her stela? | Anubis | OK |
| Which god united with his father to ensure his rebirth? | Osiris | **Not OK** (the sentence says Re) |
| Where was this large Osiris dedicated? | Memphis | OK |

**4 / 6 sound.** Sample is small; treat it as a rough indication, not a measured accuracy.
The unit tests (41) confirm that bad outputs are rejected: answer leak, invented evidence, wrong option present in
the record, hedged text, garbage output, failed verification, yes/no, wrong question type for the answer,
ignored opening words, same-god aliases (Amun / Amen-Ra, Re / Amen-Ra) and nested places (Thebes / Karnak).

Not tested: Qwen2.5-1.5B-Instruct.
