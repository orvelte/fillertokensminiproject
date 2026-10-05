# Phase 2 — uplift screen (2026-10-04)

**GATE: PASS.** At N=600, dots k=100 gives +10.0 pp (53.5% -> 63.5%, McNemar p = 3e-8) and
counting k=25 gives +8.7 pp (p = 1e-6). The bar was +5 pp with p < 0.01.

Model: V4 Flash via OpenRouter (Parasail fp8), thinking off, 10-shot, greedy, max_tokens=10.
Cost: $0.26 for this phase (cumulative $0.31). No retry or 5-shot diagnostic was needed.

## Pilot (150 items, paired) — `python scripts/phase2.py pilot`
| Condition | Filler tokens | Accuracy | Gain vs k=0 | wrong->right | right->wrong | McNemar p |
|---|---|---|---|---|---|---|
| k=0 | 0 | 51.3% | — | — | — | — |
| dots 10 | 10 | 54.7% | +3.3 pp | 14 | 9 | 0.40 |
| dots 25 | 25 | 61.3% | +10.0 pp | 23 | 8 | 0.011 |
| dots 50 | 50 | 60.0% | +8.7 pp | 19 | 6 | 0.015 |
| dots 100 | 100 | 62.7% | +11.3 pp | 24 | 7 | 0.003 |
| counting 25 | 51 | 63.3% | +12.0 pp | 25 | 7 | 0.002 |

Scaling rule met (best gain >= 5 pp, p < 0.05). Best two: counting 25 and dots 100.
Figure: `scripts/phase2_uplift_pilot.png`. Table: `data/results/phase2_pilot_table.csv`.

## Full run (N=600, paired) — `python scripts/phase2.py full --conds counting_25 dots_100`
| Condition | Filler tokens | Accuracy | Gain vs k=0 | wrong->right | right->wrong | McNemar p |
|---|---|---|---|---|---|---|
| k=0 | 0 | 53.5% | — | — | — | — |
| counting 25 | 51 | 62.2% | +8.7 pp | 83 | 31 | 1.2e-6 |
| dots 100 | 100 | 63.5% | +10.0 pp | 89 | 29 | 2.8e-8 |

Figure: `scripts/phase2_uplift_full.png`. Table: `data/results/phase2_full_table.csv`.

## Notes
- Filler tokens per turn were measured from `usage.prompt_tokens` (difference from k=0 divided by
  the 11 turns): dots are 1 token per unit, counting to 25 is 51 tokens.
- Filler also breaks answers: about 5% of items go right->wrong (29–31 of 600). Some of that is
  T=0 non-determinism rather than a filler effect; Phase 0 found greedy is not repeatable.
- Parse rate 99.8–100% in every condition; 0 calls with reasoning tokens.
- The gain is smaller than Brauer reports for DeepSeek V3 on this task (31% -> 61%), with a
  stronger prompt here (10-shot) and a higher baseline.

## F* for Phase 3: dots k=100
Largest and most significant gain at N=600, and dots carry no content of their own (counting adds
digits to the context). Cost consequence: about 2,020 prompt tokens per call instead of 920.
Alternative if cost matters: dots k=25 showed a similar gain in the pilot (+10 pp at n=150) at
roughly 60% of the per-call cost, but was not confirmed at N=600.

## STOP 2 — recommendation
Proceed to Phase 3 pilot (M=60) with F* = dots k=100. Phase 3 projected cost for the full M=300:
$2.30 expected, $3.22 upper bound.
