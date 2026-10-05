# Phase 4 — filler-type ladder, behavioural (2026-10-04)

**STOP 4.** Run with explicit human OK. This tests the partner's freed-capacity hypothesis, not Neel's.

Commands: `python scripts/phase4.py pilot` (100 items), then `python scripts/phase4.py full`.
V4 Flash via OpenRouter (Parasail fp8), thinking off, 10-shot, greedy, N=600, paired.
Cost: $0.88 (cumulative $5.99 of $30).
Figure: `scripts/phase4_ladder_full.png`. Table: `data/results/phase4_full_table.csv`.
Filler strings and token counts: `data/tasks/phase4_fillers.json`.

- Every filler is exactly 100 tokens by the V4 Flash tokenizer (downloaded from the HF repo); the
  API's prompt-token counts agree (100–101 per turn).
- The same filler string appears in all 10 few-shot examples and the test item.
- "No filler" and "dots" are reused from Phase 2.
- **Format violations: 1 of 6,600 responses** (in the no-filler condition), so accuracy counting
  violations as wrong and accuracy among format-following responses are the same to one decimal.
- **Surprisal: deferred to white-box.** Phase 0 found no route to prompt-token logprobs.

| Filler (100 tokens) | Accuracy | Gain vs no filler | wrong->right | right->wrong | p vs no filler | p vs dots |
|---|---|---|---|---|---|---|
| Memorized text (Gettysburg Address) | 64.8% | +11.3 pp | 87 | 19 | 1e-11 | 0.37 |
| Repeated word (`the the ...`) | 64.5% | +11.0 pp | 93 | 27 | 1e-9 | 0.49 |
| Prose, two paragraphs (P+Q) | 64.0% | +10.5 pp | 92 | 29 | 8e-9 | 0.80 |
| Counting in digits | 64.0% | +10.5 pp | 90 | 27 | 4e-9 | 0.79 |
| Dots | 63.5% | +10.0 pp | 89 | 29 | 3e-8 | — |
| Prose, one paragraph twice (P+P) | 63.3% | +9.8 pp | 84 | 25 | 1e-8 | 1.0 |
| Shuffled prose (P+Q words shuffled) | 62.0% | +8.5 pp | 83 | 32 | 2e-6 | 0.31 |
| Counting in words | 57.5% | +4.0 pp | 67 | 43 | 0.028 | 3e-5 |
| Short cycle (`red blue green ...`) | 57.2% | +3.7 pp | 69 | 47 | 0.051 | 1e-5 |
| Random vocabulary tokens | 55.8% | +2.3 pp | 57 | 43 | 0.19 | 2e-7 |
| No filler | 53.5% | — | — | — | — | — |

Matched pairs (McNemar):
- Prose vs shuffled prose: 64.0% vs 62.0%, p = 0.13. No detectable difference.
- P+Q vs P+P: 64.0% vs 63.3%, p = 0.64. No difference.
- Counting in digits vs in words: 64.0% vs 57.5%, p = 5e-7. Digits are better.

Reading:
- Seven very different fillers give the same uplift as dots (62–65%). Content-free tokens are not
  special; ordinary prose, a memorized passage and even shuffled prose work as well.
- Three fillers give little or nothing: counting in words, a three-word cycle, random vocabulary.
- **A simple "more predictable filler frees more capacity" ordering does not fit.** Two highly
  predictable fillers (word counting, the cycle) are among the worst, while novel prose and
  shuffled prose are among the best. Without measured surprisal this rests on informal judgements
  of predictability.
- Random vocabulary being weak agrees with the disruption runs, where all-letter filler kept only
  part of the benefit.
- Not investigated: why word counting and the cycle underperform when digit counting does not.

Caveats: one fixed string per filler type (one prose pair, one shuffle, one random sample), so
"type" effects are confounded with the particular string. Greedy on this route is not deterministic.

Recommendation: no further black-box runs planned. Human decides what follows.
