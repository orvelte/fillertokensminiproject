# Phase 3 — E3 black-box (2026-10-04)

## STOP 3 — full run, M=300

**Verdict: (a), with a caveat.** Filler does more than sharpening or majority voting: items where
the correct answer was *not* the modal no-filler answer improve by +22.9 pp [+17.0, +29.2], while
voting over the no-filler samples predicts about 0 and lower temperature gives at most +2.2 pp
[−0.7, +5.1]. The caveat: this also rules out the *simplest* form of Neel's hypothesis (plain
majority vote), and black-box data cannot say whether the extra effect comes from "parallel
attempts + smarter selection" or from filler enabling a different, better computation.

Commands: `python scripts/phase3.py --M 300`, then `python scripts/phase3_analysis.py --M 300`.
F* = dots k=100; K=16 samples per item per condition; V4 Flash via OpenRouter (Parasail fp8).
Cost: $1.85 for Phase 3 (cumulative $2.16 of $30). Parse rate 99.7%; 0 calls with reasoning.
Figure: `scripts/phase3_delta_vs_p0_M300.png`. Numbers: `data/results/phase3_M300_summary.json`,
per item: `data/results/phase3_M300_items.csv`.

Half A (samples 0–7 of the k=0, T=1 set) classifies each item; half B gives p0 and maj@k.

| Group (from half A) | n | p0 | pf | Δ filler [95% CI] | maj@15 Δ | T=0.7 Δ [CI] | T=0.4 Δ [CI] |
|---|---|---|---|---|---|---|---|
| Correct-modal | 153 | 0.82 | 0.90 | +8.3 pp [+4.5, +12.0] | +8.9 | +5.5 [+3.1, +7.9] | +8.1 [+5.6, +10.9] |
| Correct present, non-modal | 45 | 0.17 | 0.50 | +33.6 pp [+22.5, +45.3] | −1.4 | +4.0 [−1.1, +9.2] | +6.4 [−0.6, +14.0] |
| Correct absent | 91 | 0.05 | 0.23 | +17.7 pp [+10.9, +24.7] | −0.1 | +0.1 [−2.2, +2.5] | +0.1 [−2.2, +2.7] |
| Wrong-modal (two rows above) | 136 | 0.09 | 0.32 | +22.9 pp [+17.0, +29.2] | −0.5 | +1.4 [−0.9, +3.8] | +2.2 [−0.7, +5.1] |
| Tie | 11 | 0.24 | 0.52 | +28.4 pp [+1.1, +55.1] | +10.3 | +11.4 | +10.2 |
| All | 300 | 0.47 | 0.62 | +15.6 pp [+12.1, +19.3] | +4.7 | +3.9 [+2.2, +5.6] | +5.5 [+3.4, +7.5] |

### Which hypothesis pattern holds
- **H_vote (simple majority vote): rejected.** It predicts wrong-modal items stay flat or get worse;
  they improve by +23 pp. Only 12.5% of wrong-modal items get worse with filler; 51% get better.
- **H_sharpen: rejected as the full story.** T=0.7 and T=0.4 leave wrong-modal items unchanged.
- **H_verify pattern: present.** 67% of present-but-non-modal items improve (11% get worse), mean
  +34 pp.
- **Beyond H_verify:** items where the correct answer was *absent* from half A also improve (+18 pp).
  Of 42 items with 0 correct in all 48 no-filler samples (three temperatures), 14 get at least one
  correct answer with filler and 4 get a majority correct. Filler produces answers the no-filler
  model essentially never samples, which "selecting among existing attempts" does not explain.
- **H_nudge (uniform small boost): does not fit either.** The effect is concentrated: 48 of 300
  items rise by >= 50 points in pass rate and 3 fall by that much (T=0.7: 2 up, 1 down; T=0.4:
  10 up, 0 down). 128 items do not move at all.
- **Correct-modal items look like sharpening:** their gain (+8.3 pp) matches maj@15 (+8.9) and
  T=0.4 (+8.1). So a sharpening-like component may exist alongside the larger wrong-modal effect.
- **maj@k fit:** no k matches pf. Best on average is maj@15, still 11 pp short of mean pf; per item,
  maj@1 (no change) is the best match for 205 of 300 items. Temperature fits per-item pf slightly
  better than any maj@k (RMSE 0.32 vs 0.35) but both are poor.

### Testability checks
- **T1 power: FAIL (marginal).** Items with p0 in [0.1, 0.9]: 75 correct-modal, 52 wrong-modal
  (target about 60 each). p0 is strongly bimodal (88 items at 0, 76 at 1). The wrong-modal effect is
  nonetheless detected with a CI far from zero, so the shortfall did not prevent a conclusion.
  The pivot table's fix (raise difficulty once, re-run the pilot) looks unnecessary; human decides.
- **T2 uplift survives sampling: PASS.** Mean Δ = +15.6 pp [+12.1, +19.3] at T=1.
- **T3 discriminability: FAIL on the 5 pp guide, but the predictions are separated.** Half-widths:
  correct-modal 3.8 pp, wrong-modal 6.1 pp, absent 6.9 pp, present-non-modal 11.4 pp. M needed for
  5 pp: about 445 (wrong-modal) and about 1,560 (present-non-modal). Not increased. The wrong-modal
  interval [+17.0, +29.2] excludes both the voting prediction (0) and the temperature intervals.

### Caveats
- One task, one model, one filler condition (dots k=100), one host (fp8).
- Unparsed responses (0.3%) are counted as wrong; they are more common without filler (0.4%) than
  with it (0.06%), which cannot account for effects of this size.
- p0 rests on 8 samples per item, so individual items are noisy; group means are unbiased.
- Mean accuracy under filler barely depends on sampling (62.1% at T=1 vs 63.5% greedy in Phase 2),
  while no-filler accuracy does (46.5% at T=1 vs 53.5% greedy). Not investigated.

### Recommendation
Verdict (a) meets the Phase 5 entry condition. The black-box result cannot separate "attempts +
selection" from "different computation"; that needs the white-box spike. Phase 4 is optional and
tests the partner's hypothesis; it needs an explicit OK.

---
## Pilot, M=60 (this is NOT STOP 3; full M=300 not yet run)
Commands: `python scripts/phase3.py --M 60`, then `python scripts/phase3_analysis.py --M 60`.
F* = dots k=100. K=16 samples per item per condition. Cost: $0.37 (cumulative $0.68).
Parse rate 99.9%; 0 calls with reasoning tokens.
Figure: `scripts/phase3_delta_vs_p0_M60.png`. Numbers: `data/results/phase3_M60_summary.json`.

Half A (8 of the 16 k=0, T=1 samples) classifies each item; half B gives p0 and maj@k.

| Group (from half A) | n | p0 | pf | Δ filler [95% CI] | maj@15 Δ | T=0.7 Δ | T=0.4 Δ |
|---|---|---|---|---|---|---|---|
| Correct-modal | 28 | 0.79 | 0.88 | +0.09 [+0.01, +0.20] | +0.10 | +0.05 | +0.07 |
| Correct present, non-modal | 10 | 0.11 | 0.29 | +0.18 [0.00, +0.38] | −0.01 | +0.01 | +0.01 |
| Correct absent | 21 | 0.08 | 0.27 | +0.19 [+0.02, +0.37] | +0.02 | −0.05 | −0.02 |
| Wrong-modal (the two rows above) | 31 | 0.09 | 0.28 | +0.19 [+0.06, +0.33] | +0.01 | −0.03 | −0.01 |
| All (incl. 1 tie) | 60 | 0.43 | 0.57 | +0.15 [+0.07, +0.23] | +0.06 | +0.01 | +0.03 |

Pilot-level reading (wide CIs, do not treat as the verdict):
- Wrong-modal items improve by about as much as or more than correct-modal ones. Majority voting over
  the no-filler samples and lower temperature both predict about zero change for them.
- The effect is concentrated: 10 of 60 items move up by >= 0.5 in pass rate and none move down by
  that much. Comparing k=0 at T=1 with k=0 at T=0.7, no item moves by >= 0.5.
- Several "absent" items (correct answer never seen in 8 no-filler samples) reach ~100% with filler.
- No maj@k or temperature matches pf on average (best is maj@15, still 9 pp short), and none
  predicts per-item pf better than p0 itself (RMSE about 0.36 for all).
- Headline fractions: 19% of wrong-modal items get worse with filler, 48% get better; 40% of
  present-non-modal items improve (n=10).

Testability checks at M=60:
- **T1 power: PASS (scaled).** 14 correct-modal and 13 wrong-modal items have p0 in [0.1, 0.9];
  that projects to about 70 and 65 at M=300 (need about 60 each). p0 is strongly bimodal: 19 items
  at 0 and 13 at 1.
- **T2 uplift survives sampling: PASS.** Mean Δ = +14.6 pp [+6.5, +23.3] at T=1.
- **T3 discriminability: FAIL at M=60, as expected for a pilot.** CI half-widths: correct-modal
  9.5 pp, wrong-modal 13.4 pp. Projected at M=300: about 4.2 pp and 6.0 pp (non-modal 8.5 pp,
  absent 7.9 pp). So T3 will probably miss the 5 pp guide for the subgroups at M=300, although an
  effect of the pilot's size (+19 pp) would still be clearly separated from zero.

Recommendation: run the full M=300 (about $1.50 more; 60 items are already cached).

---

## T1 fix attempt: harder task, pilot only (M=60) — 2026-10-04
Human chose to raise difficulty once and re-run the pilot. Change: coefficients {2, 3, 4} instead
of {2, 3} (`--max-coef 4`); everything else unchanged. New set written alongside the frozen one as
`data/tasks/{eval,fewshot,settings}_hard.*` (the original set is untouched).

Commands: `python scripts/phase1.py --max-coef 4 --suffix _hard`;
`python scripts/phase3.py --M 60 --suffix _hard`; `python scripts/phase3_analysis.py --M 60 --suffix _hard`.
Cost: $0.36 (cumulative $2.52). Greedy k=0 baseline on 100 items: 35% (was 50%).
Figure: `scripts/phase3_delta_vs_p0_M60_hard.png`.

**T1 still FAILS, slightly worse than before.** Items with p0 in [0.1, 0.9]: 10 correct-modal and
11 wrong-modal of 60 (need 12 each), projecting to about 50 and 55 at M=300. On the original task
the full run had 75 and 52. Harder items mostly moved to the floor (26 of 60 have p0 = 0) rather
than into the mid range.

| Group (from half A) | n | p0 | pf | Δ filler [95% CI] | maj@15 Δ | T=0.7 Δ | T=0.4 Δ |
|---|---|---|---|---|---|---|---|
| Correct-modal | 22 | 0.71 | 0.87 | +15.6 pp [+3.7, +29.0] | +2.9 | +10.8 | +17.3 |
| Correct present, non-modal | 8 | 0.22 | 0.27 | +4.7 pp [−7.8, +18.8] | −1.0 | +0.8 | +7.8 |
| Correct absent | 27 | 0.03 | 0.18 | +15.0 pp [+6.7, +24.8] | +0.3 | +2.1 | −0.2 |
| Wrong-modal (two rows above) | 35 | 0.08 | 0.20 | +12.7 pp [+5.4, +20.9] | 0.0 | +1.8 | +1.6 |
| All (incl. 3 ties) | 60 | 0.32 | 0.46 | +13.8 pp [+7.5, +20.6] | +1.3 | +5.4 | +7.8 |

- T2 PASS: mean Δ = +13.8 pp [+7.5, +20.6]. T3 FAIL at pilot size (wrong-modal half-width 7.8 pp).
- The main pattern replicates on a second task variant: wrong-modal items improve under filler
  while voting and temperature predict about zero. 7 of 60 items rise by >= 50 points, none fall.
- The present-but-non-modal group shows no clear gain here (n=8, CI spans zero); the wrong-modal
  gain comes from the "absent" group.
- 22 of 3,840 responses were empty strings (counted as wrong), mostly without filler.

Recommendation: the one permitted difficulty raise did not fix T1, so stop adjusting. T1's purpose
(enough items to detect a wrong-modal effect) is already met in practice by the M=300 result on
the original task. Verdict stays (a) with the stated caveat.
