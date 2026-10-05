# Phase 1 — task generation and calibration (2026-10-04)

**Result: baseline accuracy 50/100 (50%) at k=0, greedy, on the first calibration round. In the
25–60% target, so no adjustment was made and the generator is frozen.**

- Model: V4 Flash via OpenRouter (Parasail fp8), thinking off, 10-shot, max_tokens=10.
- Command: `python scripts/phase1.py` (defaults). Cost: $0.01 (cumulative $0.05).
- Parse rate 100/100; 0 calls with reasoning; 1 of the 50 wrong answers was a binding-error rival.
- T=0 is not deterministic on this route (Phase 0), so 50% carries a few points of run-to-run noise
  on top of the ±10 pp binomial interval at n=100.

## Final generator settings (frozen)
Brauer's `generate_varbind_dataset.py` defaults, seed 20261004:
5 variables per item (1 base literal x, 1 queried variable y derived from x, 3 distractors),
coefficients {2, 3}, constants 1–50, literals 10–99, chain length 1, all values non-negative.

- `data/tasks/eval.jsonl`: N = 600 items. `data/tasks/fewshot.jsonl`: 10 fixed few-shot items,
  generated in the same de-duplicated stream and excluded from eval.
- Each item stores the prompt pieces, queried variable, all variable values, the chain
  `{x, c1x, y, c2y, answer}`, distractor values, and rival answers (the question's operation applied
  to each other variable).
- Answers span 0–1010 (425 unique values over 600 items).

## Prompt format
System prompt (same for every condition):

> You will be given a list of variable definitions followed by a question. Each variable equals either a number or an expression that refers to an earlier variable (for example 'twice the number for X plus 3'). Resolve the references to work out the value the question asks for, then answer immediately with just the number, nothing else. No explanation, no words, no reasoning, just the number. Some filler text may follow the question on the 'Filler:' line before the answer.

Then 10 few-shot user/assistant turn pairs (assistant turn is the bare number), each with the same
filler as the test item, then the test turn. The k=0 baseline keeps the bare `Filler:` line.
Below are the test turns of 3 example prompts.

**Example 1: item 0, dots k=0** (truth 264; chain {'x': 48, 'c1x': 144, 'y': 114, 'c2y': 228, 'answer': 264})

```
zif = 48
yaj = three times the number for zif minus 30
rer = twice the number for zif plus 29
rav = 67
cib = twice the number for rav minus 6
Question: What is twice the number for yaj plus 36?

Filler:

Answer:
```

**Example 2: item 1, dots k=10** (truth 353; chain {'x': 77, 'c1x': 231, 'y': 192, 'c2y': 384, 'answer': 353})

```
nex = 77
faj = three times the number for nex plus 6
woz = three times the number for nex minus 43
ziq = three times the number for nex minus 39
xok = three times the number for nex minus 26
Question: What is twice the number for ziq minus 31?

Filler: . . . . . . . . . .

Answer:
```

**Example 3: item 2, counting k=25** (truth 290; chain {'x': 28, 'c1x': 84, 'y': 94, 'c2y': 282, 'answer': 290})

```
zus = 12
laj = 14
qiv = 69
zuq = 28
coq = three times the number for zuq plus 10
Question: What is three times the number for coq plus 8?

Filler: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25

Answer:
```

## STOP 1 — recommendation
Proceed to Phase 2 (uplift screen pilot: 150 items x 6 conditions, about $0.15).
