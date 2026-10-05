# Follow-up analyses on existing data (2026-10-04)

Human asked for cheaper alternatives to the white-box spike and approved the two free ones.
Command: `python scripts/followup_existing_data.py` (no API calls, $0).
Numbers: `data/results/followup_existing_data.json`. Data: Phase 3 M=300 samples, Phase 2 N=600.

## 1. Error-step analysis (which step fails, and what filler fixes)
Each wrong answer is matched to the first simple story that reproduces it.

| Error type | % of samples, no filler (T=1) | % of samples, T=0.4 | % of samples, filler | Relative change with filler |
|---|---|---|---|---|
| Wrong value of y, final step done correctly | 45.7 | 41.0 | 33.2 | −27% |
| Final-step structural (sign / coefficient / dropped step) | 3.0 | 3.3 | 2.1 | −31% |
| Final step itself wrong (not reachable from any integer y) | 1.6 | 1.0 | 0.1 | −94% |
| Hop-1 structural (sign / coefficient / dropped constant / wrong source) | 1.5 | 1.2 | 1.1 | −24% |
| Binding: question applied to another variable | 1.4 | 1.3 | 1.3 | −5% |
| Unparsed | 0.4 | 0.3 | 0.1 | −85% |
| Total wrong | 53.5 | 48.0 | 37.9 | −29% |

- **Where the model fails:** 85% of wrong answers equal the correct final step applied to a wrong y.
  The share consistent with a correct final step is 96–97% without filler, against a chance level
  of 50% (c2=2) or 33% (c2=3). So the failure is almost always in computing y, not in binding or in
  the final step. Implied y errors are mostly small (half are within 10) and 64% are undershoots.
  Caveat for c2=2: a carry slip while doubling also yields an even number, so some "wrong y" cases
  could be final-multiplication slips; the c2=3 rows do not have this ambiguity and show the same.
- **What filler fixes:** most of the gain (12.5 of 15.7 points) is fewer wrong-y errors. Final-step
  errors nearly vanish. Binding errors do not change.
- **Error type does not predict which items flip:** items that jump >= 50 points with filler and
  items that stay wrong have nearly the same no-filler error mix (83% vs 88% wrong-y).
- **Answer concentration (16 samples per item):**

| Item kind | Distinct answers, no filler | Distinct answers, filler | Top wrong answer's share, no filler | Top wrong answer's share, filler |
|---|---|---|---|---|
| Flips up (n=48) | 7.5 | 1.8 | 36% | 6% |
| Stays wrong (n=91) | 8.1 | 4.4 | 38% | 60% |

  Without filler, hard items give many different wrong answers. With filler the model settles on
  one answer: the correct one for items that flip, and a single wrong one for items that stay wrong.
  For only 38% of stay-wrong items is that the same wrong answer that was most common without filler.

## 2. Cross-filler agreement (dots k=100 vs counting k=25, greedy, N=600)
- Of 279 items wrong at k=0, dots fix 89 and counting fixes 83; **61 are fixed by both** (26.5
  expected if independent; odds ratio 16.6, Fisher p = 1e-21).
- Of 321 items right at k=0, dots break 29 and counting breaks 31; 18 are broken by both (2.8
  expected; odds ratio 35, p = 4e-14).
- Independent check on the first 300 items, restricted to items with no-filler pass rate <= 25%:
  where counting-25 is right (greedy), the sampled dots-100 pass rate is 0.74 (n=48); where it is
  wrong, 0.10 (n=97).
- Caveat: some agreement reflects item difficulty that an 8-sample pass rate does not capture.
  Splitting the same low-pass-rate items by the *k=0* greedy outcome also separates dots pass
  rates (0.63 vs 0.26, n=21 vs 124), though less sharply.

## Reading
- Filler effects are item-specific and shared across two different fillers, so they are not random
  re-rolls of borderline items.
- The picture is "noisy computation of y without filler, a settled value with filler", where the
  settled value is right for some items and consistently wrong (often a *new* wrong answer) for
  others. A vote or selector over the no-filler candidates would favour the existing plurality
  answer; a new consistent wrong answer fits "filler changes the computation" better.
- This tilts toward the different-computation explanation but does not exclude parallel attempts
  whose candidates differ from what the no-filler model samples.

---

# Per-item dose-response (2026-10-04)

Command: `python scripts/followup_dose.py`. Cost $1.05 (cumulative $3.56 of $30).
Dots at k = 10, 25, 50 added to the Phase 3 data (k = 0, 100); T=1, 16 samples per item per k.
219 of the 300 items were run; the 81 skipped were at ceiling (>= 15/16) at both k=0 and k=100.
Figure: `scripts/followup_dose.png`. Per item: `data/results/followup_dose_items.csv`.

Item kinds are defined from k=0 and k=100 (as before), so those two columns are selected values;
k = 10, 25, 50 are fresh, unselected samples.

| Item kind | n | k=0 | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| Flips up: pass rate | 48 | 0.16 | 0.42 | 0.64 | 0.70 | 0.92 |
| Other: pass rate | 80 | 0.60 | 0.58 | 0.60 | 0.67 | 0.72 |
| Stays wrong: pass rate | 91 | 0.05 | 0.06 | 0.08 | 0.07 | 0.04 |
| Flips up: distinct answers in 16 | 48 | 7.5 | 5.7 | 3.7 | 3.2 | 1.8 |
| Other: distinct answers in 16 | 80 | 4.6 | 4.3 | 3.9 | 3.0 | 2.5 |
| Stays wrong: distinct answers in 16 | 91 | 8.1 | 7.4 | 6.5 | 5.5 | 4.4 |

Flip items, per item:
- The average rises gradually, but individual items mostly move in one jump: the largest rise
  between adjacent k values is >= 50 points for 73% of them (median 0.59).
- The jump happens at different lengths for different items. Smallest k with pass rate >= 75%:
  k=10 for 9 items, k=25 for 18, k=50 for 7, k=100 for 11, never for 3.
- At k=10, 44% of flip items sit at an intermediate pass rate (25–75%); at k=25 and k=50 about 22%.
- 23% are not monotone in k (a drop of more than 2/16 somewhere from k=10 up). Across all 219
  items, 7 do better by >= 50 points at some intermediate k than at k=100.

Items that stay wrong:
- They stay wrong at every length; there is no hidden good length.
- The answer they settle on depends on k: the most common answer is the same at k=50 and k=100
  for 58% of items, and at k=25 and k=100 for 42%. (With 16 samples and several distinct answers,
  some of this is sampling noise in the mode.)

Reading:
- Answer diversity falls smoothly with filler length for every kind of item, right or wrong. Filler
  acts as a gradual "settling" of the answer; whether it settles on the right value is item-specific.
- Neither clean prediction fits. "More positions = more attempts" predicts monotone, gradual gains
  per item; most items jump, and a quarter are non-monotone. "Enough room to finish a fixed-length
  chain" predicts one common threshold; thresholds range from 10 to 100 across items.
- What fits: longer filler progressively reduces noise in the computation of y, and each item tips
  into a consistent answer at its own length. That is compatible with averaging many noisy
  computations inside the model (a latent form of the parallel idea that an output-level vote test
  cannot see) and with a more reliable serial computation. Black-box data cannot separate these.

---

# Binding error / arithmetic slip / other (2026-10-04)

Human-requested. Command: `python scripts/followup_error_mix.py` ($0, existing data).
Figure: `scripts/followup_error_mix.png`. Numbers: `data/results/followup_error_mix.json`.

Definitions, applied in order: **binding error** = equals a recorded rival answer, or the whole
chain run from the wrong source variable; **arithmetic slip** = not binding and within 60 of the
correct answer; **other** = the rest, including unparsed.

Sampled data (300 items x 16 samples, T=1):

| Error type | No filler, % of samples | Dots k=100, % of samples | Relative change [95% CI over items] | Share of errors, no filler -> filler |
|---|---|---|---|---|
| Binding error | 1.7 | 1.4 | −16% [−65%, +44%] | 3.2% -> 3.8% |
| Arithmetic slip | 42.5 | 28.7 | −32% [−40%, −25%] | 79.3% -> 75.8% |
| Other | 9.4 | 7.7 | −18% [−38%, +4%] | 17.5% -> 20.4% |
| All wrong | 53.5 | 37.9 | −29% | |

- **The task has essentially one error type.** Binding errors are 3% of errors, and about a third
  of those are coincidence (1.0% of wrong answers match *another* item's rivals).
- **The effect lives in arithmetic on the right variables**: 660 of the 821 errors removed by filler
  (80%) are arithmetic slips, the same as their share of errors (79%).
- **The proportionality test is not discriminating here.** The error mix barely changes, which is
  what "all types reduced in proportion" predicts, but it is also what "one bottleneck" predicts
  when one type is 80% of errors. The binding interval is far too wide to say whether binding
  errors are spared.
- Greedy replication (600 items): binding 8 -> 9 (dots) and 8 -> 8 (counting); arithmetic slips
  −26% [−35%, −17%] (dots) and −17% [−26%, −7%] (counting).
- Threshold sensitivity: binding −16% at every threshold; slips vs other are −44% / −27% at 10,
  −30% / −30% at 30, −32% / −18% at 60, −30% / −24% at 100. No robust difference between the two.
- By filler length (219 items not at ceiling): slips fall steadily (57.7%, 53.4, 47.6, 44.2, 39.3
  at k = 0, 10, 25, 50, 100); binding stays near 2% and "other" near 11–13%.
- Temperature 0.4 reduces slips by 12% [7%, 17%], well short of filler.

Combined with the finer breakdown earlier in this file (96–97% of unexplained errors are a correct
final step applied to a wrong y), the component is the arithmetic that produces y, with the c2=2
caveat noted there.

---

# Binding-heavy variant: does filler remove binding errors too? (2026-10-04)

Human-approved ("proceed with the black-box, partial"). Cost $0.31 (pilots $0.07, run $0.24;
cumulative $3.87 of $30).

## Finding a variant (two pilot rounds, 100 items each, k=0, greedy)
`python scripts/phase1.py <args> --suffix <name>`; binding = recorded rivals + wrong source variable.

| Variant | Accuracy | Binding errors (of wrong answers) | Constant from another line |
|---|---|---|---|
| Original (5 variables) — Phase 3 samples | 46% | 3% | 7% (about half coincidental) |
| 10 variables, similar names (`_t10sim`) | 31% | 7% | 13% |
| 10 variables, similar names, easy arithmetic (`_easy10sim`) | 31% | 12% | 25% |
| 15 variables, similar names, easy arithmetic (`_easy15sim`) | 27% | 15% | 26% |

"Easy arithmetic" = coefficient fixed to 2, constants 1–9. Similar names = all variables in an item
share a 2-letter prefix. More distractors alone barely raise binding errors; they mostly produce
more near-miss answers. Surprise: accuracy stays near 30% even with easy arithmetic, and half the
errors are still near misses that match no binding story I checked.

## Main run on `_easy15sim` — `python scripts/followup_binding.py`
Greedy, 600 items, k=0 vs dots k=100. Figure: `scripts/followup_binding.png`.

**Uplift is larger here: 23.2% -> 44.8% (+21.7 pp; wrong->right 155, right->wrong 25; McNemar p = 4e-24).**

| Error type | No filler | Dots k=100 | Relative change [95% CI] | Share of errors before -> after | Fixed by filler (same item) |
|---|---|---|---|---|---|
| Variable binding | 42 | 35 | −17% [−45%, +25%] | 9.1% -> 10.6% | 21 of 42 (50%) |
| Constant binding | 127 | 96 | −24% [−38%, −8%] | 27.5% -> 29.0% | 53 of 127 (42%) |
| Arithmetic slip | 242 | 178 | −26% [−35%, −17%] | 52.5% -> 53.8% | 73 of 242 (30%) |
| Other | 50 | 22 | −56% [−72%, −37%] | 10.8% -> 6.6% | 8 of 50 (16%) |
| All wrong | 461 | 331 | −28% | | |

- **Filler does not spare binding errors.** Item by item, binding errors become correct at least
  as often as arithmetic slips (50% and 42% vs 30%; Fisher p = 0.02 and 0.03 against slips).
- **The net reduction is roughly proportional across types**, and the error mix is nearly unchanged.
  Net binding counts fall less than the per-item fix rate because some items that had other errors
  (or were right) show a binding error under filler.
- Caveats: 42 variable-binding errors is a small base (interval −45% to +25%). About 54 of the 127
  "constant binding" matches are expected by coincidence (45 of 96 with filler). Greedy on this
  route is not deterministic: 25 of 139 right items went wrong under filler, so some churn is noise.
  This is a different task variant from the main one.

## Reading
"Filler relieves one specific step (the arithmetic for y)" predicts binding errors are spared. They
are not: filler helps across error types, and helps most on a variant whose difficulty comes from
clutter, not arithmetic. That fits a general noise-reduction mechanism (hypothesis A, latent
aggregation) better than a single relieved bottleneck. It does not exclude a version of B in which
extra positions relieve a general capacity limit that affects every step.

---

# Filler disruption by location (2026-10-04)

Human-approved ("run 3"). Design and predictions below were written BEFORE any data was collected.

## Design
- Items: the 48 items that flipped up by >= 50 points between k=0 and dots k=100 in Phase 3.
  Because they were selected on those samples, fresh k=0 and intact dots-100 samples are drawn here.
- Six conditions, T=1, 16 fresh samples per item each:
  no filler; intact (100 dots); early (units 1–20 replaced); middle (units 41–60 replaced);
  late (units 81–100 replaced); all 100 units replaced.
- Disruptive tokens: independent random lowercase letters, space-separated (one per unit), so the
  filler stays 100 units long. Letters carry no numbers. Each few-shot example shows the same
  condition with its own fixed random letters; the test item gets item-specific letters.

## Pre-registered criteria
Let loss(loc) = pass rate intact − pass rate with the block at loc, and benefit = intact − no filler.
All intervals are 95% bootstrap over items, paired.
- **Location-uniform (fits independent attempts):** every pairwise difference between the three
  location losses is under 10 pp in absolute value with an interval that includes 0.
- **Location-specific (fits one spread-out chain):** some pairwise difference is >= 10 pp with an
  interval that excludes 0.
- **Graceful vs fragile:** graceful if every location's loss is <= 30% of the benefit (20% of the
  positions are damaged); fragile if any location's loss is >= 50% of the benefit.
- **Known ambiguity, stated now:** a late block sits directly before `Answer:` and could hurt under
  either story. "Late is worst" is therefore weak evidence; "early or middle is worst" is the
  diagnostic outcome for a chain. If intact filler does not beat no filler by >= 25 pp on these
  fresh samples, the test is uninformative.

## Results — `python scripts/followup_disrupt.py`
Cost $0.68 (cumulative $4.55 of $30). 48 items, 16 fresh samples each, parse rate >= 99%.
Figure: `scripts/followup_disrupt.png`. Numbers: `data/results/followup_disrupt.json`.

| Condition | Pass rate [95% CI] | Loss vs intact [95% CI] | Share of filler benefit lost | Distinct answers in 16 |
|---|---|---|---|---|
| No filler | 22.0% [16.5, 27.9] | — | — | 7.4 |
| Intact, 100 dots | 87.2% [82.4, 91.7] | — | — | 2.2 |
| Letters at units 1–20 (early) | 82.0% [74.1, 89.3] | +5.2 pp [−0.8, +11.6] | 8% [−1%, 18%] | 2.4 |
| Letters at units 41–60 (middle) | 91.7% [87.2, 95.6] | −4.4 pp [−7.8, −1.3] | −7% | 1.8 |
| Letters at units 81–100 (late) | 90.2% [85.2, 94.8] | −3.0 pp [−7.9, +2.6] | −5% | 1.7 |
| All 100 units letters | 50.7% [41.1, 60.5] | +36.6 pp [+27.9, +45.2] | 56% [41%, 72%] | 5.1 |

Pairwise loss differences: early vs middle +9.6 pp [+3.9, +16.1]; early vs late +8.2 pp
[+2.1, +15.1]; middle vs late −1.4 pp [−6.6, +3.3].

**Pre-registered verdicts**
- Informative: yes. Filler benefit on fresh samples is +65 pp [+59, +72], so the flips replicate.
- Graceful vs fragile: **graceful.** No location loses more than 8% of the benefit.
- Location: **neither criterion met.** Early hurts more than middle or late with intervals that
  exclude zero, but the differences (9.6 and 8.2 pp) are just under the 10 pp bar set in advance.

Other observations (not pre-registered):
- Middle and late blocks do not hurt at all; the middle block's pass rate is slightly *higher* than
  intact (interval excludes zero). Not investigated.
- Random letters throughout still give +29 pp over no filler, about 44% of the dots' benefit, so
  filler content matters but positions alone help.
- Items losing >= 50 points: early 4, middle 0, late 2, all-replaced 20.

**Limitation found after the fact:** the dose-response showed most flip items need only 25–50
dots. With 100 units, any single 20-unit block leaves 40–80 clean contiguous dots, enough for a
chain to run in what remains. So "graceful" is weaker evidence against a spread-out chain than the
design intended. A sharper version would disrupt a block of a 25-unit filler, where there is no
slack. Logged as an idea; not run.

Reading: graceful degradation fits redundant computation across positions (hypothesis A) and does
not show the location-specific fragility a single spread-out chain would predict, subject to the
slack limitation. The small early-specific cost points in the direction Brauer's ordering would
predict but did not reach the pre-set threshold.

---

# Disruption with no slack: 25-unit filler (2026-10-04)

Human-approved ("go"). Design and criteria below were written BEFORE any data was collected.

## Design
- Items: the same 48 flip items. T=1, 16 fresh samples per item per condition (sample indices
  100–115; the no-filler condition reuses the fresh samples from the 100-unit disruption run).
- Conditions: no filler; intact 25 dots; letters at units 1–8 (early); letters at units 10–17
  (middle); letters at units 18–25 (late); **17 intact dots** (same number of clean positions as
  the disrupted conditions); all 25 units letters.
- Letters as before: independent random lowercase letters, fixed per few-shot example,
  item-specific for the test item, identical across block locations.
- **Primary analysis set, fixed now:** flip items whose pass rate at 25 dots was >= 50% in the
  earlier dose-response run (independent samples), i.e. items for which 25 dots is enough.
  All 48 items are reported as a secondary analysis.

## Pre-registered criteria (primary set; 95% paired bootstrap over items)
Let gap(loc) = pass rate with 17 intact dots − pass rate with the block at loc.
- **Informative** only if intact 25 dots beats no filler by >= 25 pp.
- **Redundant positions (fits attempts):** every gap(loc) is under 10 pp in absolute value with an
  interval including 0, and every pairwise difference between locations likewise.
- **Location-specific (fits one spread-out chain):** some gap(loc) is >= 10 pp with an interval
  excluding 0, and some pairwise difference between locations is >= 10 pp with an interval
  excluding 0.
- **General disruption:** every gap(loc) is >= 10 pp with an interval excluding 0 but no pairwise
  difference meets the bar. This would mean the letters themselves do harm; it fits neither story.
- Anything else: "no criterion met".
- Stated ambiguity, as before: the late block sits directly before `Answer:`, so "only late is
  worse" is weak evidence for a chain; early or middle being worse is the diagnostic outcome.

## Results — `python scripts/followup_disrupt25.py`
Cost $0.47 (cumulative $5.02 of $30). Parse rate >= 99% in every condition.
Figure: `scripts/followup_disrupt25.png`. Numbers: `data/results/followup_disrupt25.json`.

Primary set (33 items for which 25 dots was enough in the earlier run):

| Condition | Pass rate [95% CI] | Gap vs 17 intact dots [95% CI] |
|---|---|---|
| No filler | 26.1% [19.1, 33.7] | — |
| Intact 25 dots | 85.4% [78.4, 91.9] | −7.2 pp [−13.8, −0.6] (25 dots are better) |
| 17 intact dots (control) | 78.2% [70.1, 85.8] | — |
| Letters at units 1–8 (early) | 71.8% [61.4, 81.4] | +6.4 pp [−5.7, +18.4] |
| Letters at units 10–17 (middle) | 72.7% [63.6, 81.4] | +5.5 pp [−2.3, +13.6] |
| Letters at units 18–25 (late) | 78.6% [70.5, 86.2] | −0.4 pp [−11.2, +10.0] |
| All 25 units letters | 62.5% [52.5, 72.2] | +15.7 pp [+2.3, +29.2] |

Pairwise location differences: early vs middle +0.9 pp [−9.9, +11.6]; early vs late +6.8 pp
[−4.0, +18.4]; middle vs late +5.9 pp [−4.2, +16.5].

**Pre-registered verdict: "redundant positions"** (primary set, and the same on all 48 items).
- Informative: intact 25 dots beat no filler by +59 pp [+49, +69].
- Every disrupted condition is within 10 pp of the 17-dot control with an interval including 0,
  and so is every pairwise location difference.
- A block of letters costs about what removing those 8 positions costs (25 dots vs 17 dots:
  7 pp), wherever the block sits.

Caveats:
- With 33 items the intervals are about ±10–12 pp, so a location effect of up to roughly 18 pp
  (early) cannot be excluded; the verdict is "no location effect detected at this size".
- Point estimates for early and middle are 5–6 pp below the control, late is level with it.
- Random letters throughout still reach 62.5%, +36 pp over no filler.
- One task, one model, one kind of disruption.

Reading: with no spare room, damaging a third of the filler anywhere behaves like having a third
fewer positions. That is what interchangeable, redundant positions predict (hypothesis A) and not
what a single chain laid out across specific positions predicts.

---

# Where the settled answer sits among the no-filler answers (2026-10-04)

Human-approved. `python scripts/followup_center.py` ($0, Phase 3 M=300 data).
Figure: `scripts/followup_center.png`. Numbers: `data/results/followup_center.json`.

Idea: if filler averages many noisy estimates, the answer it settles on should be central among
the item's 16 no-filler answers. Position 0.5 = dead centre; a random no-filler answer falls in the
middle half 50% of the time.

| Answer examined | Items | In the middle half | Median distance to the no-filler median | Same for a typical no-filler answer |
|---|---|---|---|---|
| Flip items: the correct answer (what filler settles on) | 48 | 46% | 15 | 21 (Wilcoxon p = 0.41) |
| Stay-wrong items: the wrong answer filler settles on | 58 | 60% | 9 | 23 (p = 0.035) |
| Stay-wrong items: the correct answer | 58 | 40% | 23 | 23 (p = 0.63) |

- **No clear support for numeric averaging.** On flip items the answer filler settles on is no more
  central than a random no-filler answer.
- On stay-wrong items the settled wrong answer is somewhat central, but in 45% of those items it is
  simply the most common no-filler answer and in 83% it already appears among the 16 samples, which
  makes it central by construction. It is closer to the no-filler median than the truth is in 57%
  of items (p = 0.16).
- As stated before running: a null here says little, because averaging inside the model need not
  show up as numeric closeness of answers.

---

# Cross-model item overlap: V4.1 Flash (2026-10-04)

Human-approved (adds a model: DeepSeek V4.1 Flash via Fireworks, thinking disabled).
`python scripts/followup_crossmodel.py`. Cost $0.09 (cumulative $5.11).
Same 600 items, same 10-shot prompt, greedy, k=0 vs dots k=100. 0 calls with reasoning; parse 100%.
Figure: `scripts/followup_crossmodel.png`.

| Model | No filler | Dots k=100 | Gain | wrong->right | right->wrong | McNemar p |
|---|---|---|---|---|---|---|
| V4 Flash (OpenRouter, Phase 2) | 53.5% | 63.5% | +10.0 pp | 89 | 29 | 3e-8 |
| V4.1 Flash (Fireworks) | 43.5% | 57.2% | +13.7 pp | 121 | 39 | 6e-11 |

- **Uplift replicates on a second model** with a different architecture.
- **Difficulty is partly shared:** 202 items are wrong on both without filler (158 expected if
  independent; odds ratio 3.5, p = 2e-13).
- **Which items filler fixes is mostly model-specific:** of those 202, V4 fixes 60 and V4.1 fixes
  55; 22 are fixed on both (16.3 expected; odds ratio 1.9, p = 0.058). For comparison, two fillers
  on the *same* model overlapped with odds ratio 16.6.
- Broken items: 2 broken on both of 184 right on both (1.1 expected; p = 0.3).
- Finer check on the first 300 items: among items V4 rarely solves and V4.1 gets wrong at k=0,
  V4's filler pass rate is 0.34 where filler fixes the item on V4.1 (n=30) and 0.24 where it does
  not (n=81).
- Caveat: greedy on both routes is not deterministic, which dilutes overlap.

---

# Brauer artifact reanalysis (2026-10-04)

Human-requested. `python scripts/followup_brauer.py` ($0; no model runs).
Source: public repo `github.com/kaleybrauer/filler-token-reasoning`, commit `a701927` (2026-09-25),
folder `release/top_tokens/`. Figure: `scripts/followup_brauer.png`.

**Scope limits, all important**
- The release has **no system-of-equations decodes**, so y cannot be examined. The computed value
  studied is the **2-fact sum A1 + A2**; A1 and A2 are retrieved facts.
- Models are DeepSeek V3 and Kimi K2, not V4 Flash.
- Correct examples only.
- Each example's top-50 list is **aggregated over all layers and all filler positions**, so
  "co-present" cannot be split into "at different positions" vs "at different layers".

Control: for each offset d, how often (value + d) is in the item's own decode vs in other items'
decodes from the same file. Offsets landing on the item's other true values are excluded.

Dots filler, share of examples with value + d among the decoded numbers (own vs other items):

| Offset from the sum | DeepSeek V3 (735 examples) | Kimi K2 (1,011 examples) |
|---|---|---|
| ±1 | 18.4% vs 6.8% | 21.3% vs 6.8% |
| ±2 | 32.5% vs 6.4% | 29.8% vs 7.0% |
| ±3–9 | 14.3% vs 6.5% | 14.0% vs 6.9% |
| ±10 | 17.1% vs 6.3% | 16.5% vs 7.0% |
| ±11–19 | 7.0% vs 6.4% | 8.5% vs 6.7% |
| ±20 | 9.4% vs 6.6% | 8.7% vs 6.9% |
| ±21–30 | 4.6% vs 6.5% | 4.9% vs 6.4% |

- **Wrong values of the computed sum are co-present with the correct one, far above the control.**
  94% of examples have at least one number within ±30 of the sum that is not a true value
  (5.5 such numbers per example vs 3.9 expected from other items' decodes).
- **The near misses are structured, not a smooth blur.** Even offsets within ±9 appear in 24% of
  examples vs 11% for odd offsets (DeepSeek; Kimi 22% vs 12%; control about 7% for both). ±10
  stands out from its neighbours: excess +10.9 pp [+8.9, +12.6] vs +4.2 pp [+3.5, +5.0] for
  ±8, 9, 11, 12 (DeepSeek; Kimi +9.5 vs +4.3). Counting filler shows the same pattern.
- **Retrieved addends have a much weaker halo:** ±3–9 around A1 or A2 is at control level
  (6.6–7.3% vs 6.4%), against 14% around the sum. ±1 and ±2 are raised for addends too (10–18%).
- **Robustness check:** near-sum offsets can coincide with numbers near an addend when an addend is
  small. Restricting to examples with both addends above 40 (81 DeepSeek, 140 Kimi examples), the
  excess survives (even offsets 27.5% vs 12.3% control on DeepSeek; 15.5% vs 8.2% on Kimi) and so
  does ±10 (24.1% vs 13.6%; 11.8% vs 8.0%). An apparent excess of undershoots does **not** survive
  this check and is not claimed.

Reading:
- "One chain: the correct value plus adjacent chain steps" is not what the decodes show for the
  computed value; a cloud of specific wrong sums sits alongside the correct one, including
  tens-digit slips (±10).
- This is what "many candidate computations" predicts, but two other readings remain open and the
  release cannot separate them: (i) a single computation that refines an approximate sum across
  layers, with the aggregate pooling early and late layers; (ii) digit-wise number features, where
  the lens surfaces numbers sharing the units digit or parity with the true value.
- So: supports "filler positions carry a distribution of candidate values for the computed
  quantity"; does not establish that these are independent parallel attempts.

---

# Brauer decodes: offset spectrum (2026-10-04)

Human-requested. `python scripts/followup_brauer_spectrum.py` ($0). Same source and limits as the
section above (2-fact task, DeepSeek V3 / Kimi K2, correct examples, decodes pooled over layers and
positions). Figure: `scripts/followup_brauer_spectrum.png`.
Method change from the first pass: (item, offset) pairs within 2 of another true value of the item
are dropped, so one quantity's halo cannot leak into another's spectrum.

**Correction to the previous section:** the "±10 stands out from its neighbours" claim compared ±10
with a neighbour set that included odd offsets. Against even neighbours (±8, ±12) it does not
stand out. The ±10 bump around the sum was a parity effect.

## 1. Spectrum around the computed sum (excess = own decode − other items' decodes, pp)

| \|d\| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek V3 | +11.1 | +25.0 | +5.1 | +16.9 | +3.1 | +11.6 | +0.3 | +9.4 | +0.8 | +9.4 | +5.5 | −2.2 | +2.0 | −4.4 | −0.8 |
| Kimi K2 | +13.5 | +21.8 | +4.5 | +14.8 | +3.8 | +11.4 | +1.1 | +8.2 | +0.8 | +7.7 | +4.4 | −1.1 | +1.3 | −2.9 | −1.2 |

Regression of the 30 excess values on the three indicators (pp, 95% bootstrap CI over examples):

| Term | DeepSeek V3 | Kimi K2 |
|---|---|---|
| Same mod 2 | +7.0 [+6.1, +7.8] | +5.0 [+4.4, +5.7] |
| Same mod 5 | −0.5 [−1.2, +0.3] | −0.4 [−1.2, +0.2] |
| Same mod 10 | −2.2 [−3.5, −0.8] | −2.3 [−3.4, −1.2] |

Adding a closeness term (1/|d|) leaves mod 2 at +8.1 / +6.1, moves mod 5 to +1.1 [+0.5, +1.9] in
both models, and leaves mod 10 negative.

Key contrasts (DeepSeek; Kimi in brackets):
- ±5 vs ±3, ±7: +0.4 pp [−1.2, +2.4] (Kimi +1.0 [−0.6, +2.5]). **No ±5 peak.**
- ±15 vs ±13, ±17: +0.1 pp (Kimi +0.4). None.
- ±10 vs ±8, ±12: +2.0 pp [0.0, +4.1] (Kimi +1.4 [−0.3, +3.0]). **Not special beyond being even.**
- ±20 vs ±18, ±22: +1.2 pp; ±30 vs ±28: −0.2 pp. None.
- ±1 vs ±3: +6.1 pp [+3.9, +8.2] (Kimi +9.0). Closeness.
- Even vs odd among non-multiples of 5 within ±9: +11.4 pp [+9.7, +13.0] (Kimi +9.1).

**Neither account fits as stated.** The halo is parity plus closeness.
- Representation geometry predicted peaks at multiples of 5 and 10 and flat odd offsets: there is
  no mod 5 or mod 10 structure, and close odd offsets (±1, ±3) are raised.
- Slips from independent attempts predicted ±1 and ±10 with no parity preference: ±10 is not
  special and parity is the dominant effect. The absence of a ±5 peak is the one prediction of
  this account that holds.

## 2. Shape of the addend halo
- Around the retrieved addends the excess is confined to ±1 (+6 to +9 pp) and ±2 (+3 to +4 pp);
  from ±3 outward it is at or below the control.
- **No parity structure:** even vs odd is −0.5 / +0.6 pp for A1 / A2 on DeepSeek and −1.5 / −2.4 pp
  on Kimi.
- **A mod 10 term instead:** +2.4 pp [+1.3, +3.5] for A1 on DeepSeek, +2.5 [+1.7, +3.3] on Kimi
  (A2: +1.6 and +1.3); ±10 vs ±8, ±12 is +4.1 pp [+2.3, +5.9] for A1 on DeepSeek.
- The two spectra correlate (r = 0.82 DeepSeek, 0.73 Kimi), but that reflects the shared decline
  with distance. The specific structure differs: parity for the computed sum, tens for retrieved
  numbers.
- So this is not "same shape, smaller". The parity structure is specific to the computed value.

## 3. No-filler baseline
The release contains none: `top_tokens/` and `logit_lens_heatmaps/` cover filler conditions only.
The nearest available comparison is filler length, which does not answer the question:

| Condition | DeepSeek: sum present / mean excess within ±10 | Kimi: sum present / mean excess within ±10 |
|---|---|---|
| dots 10 | 63% / +9.7 pp | 55% / +6.1 pp |
| dots 25 | 64% / +7.8 pp | 67% / +11.1 pp |
| dots 50 | 66% / +10.4 pp | 67% / +8.8 pp |

The halo does not grow or shrink systematically from 10 to 50 dots.

## Reading
- A halo that keeps the parity of the true sum, exists only for the computed value, and has no
  tens-digit structure is not what independent attempts with arithmetic slips would produce.
- One explanation that fits (my interpretation, not tested): the sum is held as separate features,
  with its parity resolved and its magnitude still approximate, so the lens surfaces same-parity
  numbers near the true value. That is partial state of one computation, not multiple candidates.
- This weakens the support the first-pass reanalysis gave to hypothesis A. It does not bear on the
  black-box evidence for A (proportional error reduction, graceful disruption), which comes from
  V4 Flash on a different task.

---

# V4 Flash slips: parity, size, spectrum (2026-10-04)

Human-requested. `python scripts/followup_slips.py` ($0; Phase 3 M=300 samples and the dose run).
Figure: `scripts/followup_slips.png`. Numbers: `data/results/followup_slips.json`.
Task: y = c1·x ± k1, answer z = c2·y ± k2. Implied y error dy = (z_wrong ∓ k2)/c2 − y where the
division is exact. Binding errors excluded; structural errors (sign flips etc.) excluded in 1b.

**Correction to an earlier claim in this file.** I wrote that 96–97% of unexplained errors are "a
correct final step applied to a wrong y", inferred from the answer error being divisible by c2.
That inference is not safe: z's residue mod c2 is fixed by k2 whatever y is, so a model that only
knows "c2·anything is a multiple of c2" produces the same divisibility. The supported statement is
weaker: wrong answers keep the answer's residue mod c2.

## 1. Do slips keep parity?
Parity of the **final answer** is kept almost always:

| | No filler | Dots k=100 |
|---|---|---|
| c2 = 2: answer error even | 95.6% (n=951) | 99.8% (n=608) |
| c2 = 3: answer error even | 95.6% (n=1,320) | 97.9% (n=991) |
| c2 = 3: answer error divisible by 3 | 97.3% | 99.6% |

For c2 = 2 this is forced by the task (2y is even). For c2 = 3 it is not forced: z's parity
depends on y's parity, so the model must track parity through the chain to get it right.

Parity of the **implied y error** (share even; chance 50%):

| Cell | No filler | Dots k=100 | Note |
|---|---|---|---|
| c1=3, c2=3 | 95.2% [91.7, 97.8] | 97.9% [94.2, 99.9] | Equals answer parity (odd multiplier), so not a separate test |
| c1=2, c2=3 | 96.8% [93.6, 99.2] | too few | Same confound; y parity also forced by k1 |
| c1=3, c2=2 | 63.8% [55.2, 72.2] | 73.9% [63.3, 85.1] | The only cell where y parity is free and separable |
| c1=2, c2=2 | 44.4% [32.0, 57.3] | 31.4% [13.7, 56.5] | y parity forced by k1, yet not kept |

- **Slips keep the parity of the answer, well above chance, including where the task does not
  force it (c2 = 3).**
- **Whether they keep the parity of the intermediate is not established.** Where it can be
  separated from answer parity (c2 = 2), it is 64% in one cell and 44% in the other. With c2 = 2,
  a tens slip in the final multiplication appears as dy = ±5, which contaminates these cells;
  excluding multiples of 5 gives 68.5% and 37.5%.

## 2. Do slips shrink, or become rarer?
**Rarer, not smaller.**

| | No filler | Dots k=100 |
|---|---|---|
| Slips, share of samples | 50.1% | 36.4% |
| Median \|dy\| | 10 | 10 |
| Quartiles of \|dy\| | 5–20 | 6–21 |
| Mean \|dy\| | 15.4 | 15.9 |
| Share with \|dy\| <= 2 / <= 10 / > 30 | 12% / 51% / 10% | 11% / 51% / 9% |

- Difference in medians: 0.0 [−2.0, +2.0] (bootstrap over items).
- Within item (125 items with >= 4 slips in both conditions): median 11 vs 10; smaller with
  filler in 34% of items, larger in 34% (Wilcoxon p = 0.95).
- By filler length (219 items): slip rate falls 68% -> 65% -> 59% -> 55% -> 50% at k = 0, 10, 25,
  50, 100, while median |dy| stays at 10–12 and the share within 10 stays at 49–51%.
- Undershoots are 64% of slips without filler and 62% with.

## 3. Offset spectrum of implied y errors vs the Brauer decode halo
Share of slips at each |dy| (no filler), cleanest cell c1=3, c2=3:
1: 0.5%, 2: 8.3%, 3: 0.4%, 4: 13.2%, 5: 1.1%, 6: 8.1%, 7: 0.1%, 8: 7.4%, 9: 0.4%, 10: 10.3%,
12: 5.2%, 14: 4.5%, 15: 0.5%, 20: 7.5%.

| Regression term (pp) | Cleanest cell, no filler | Cleanest cell, filler | All slips, no filler |
|---|---|---|---|
| Same mod 2 | +5.3 [+4.8, +5.9] | +5.4 [+4.5, +6.2] | +3.7 [+3.2, +4.2] |
| Same mod 5 | +0.6 [+0.2, +1.0] | +0.1 [−0.2, +0.4] | +1.4 [+0.7, +2.3] |
| Same mod 10 | +1.0 [−0.8, +2.9] | +1.6 [−1.0, +4.6] | +0.6 [−1.0, +2.1] |

- **Shared with the decode halo:** parity dominates and there is no ±5 peak (in the cleanest
  cell). Correlation of the two spectra over |d| = 1..30 is r = 0.69–0.85, mostly from the
  even/odd alternation.
- **Not shared:** the decode halo is tight (peaks at ±1 and ±2, gone by about ±12, no tens
  structure). The output slips are broad (median 10, peak at ±4, ±1 almost absent, ±10 and ±20
  raised: ±10 vs ±8, ±12 is +4.0 pp without filler and +7.4 pp with).
- Caveat: in the cleanest cell dy parity is tied to answer parity (see 1), and the comparison is
  across different models and tasks. In Brauer's task the sum is itself the final answer, so the
  like-for-like statement is "near misses keep the parity of the final answer" in both.

## Reading
- Test 1 gives parity-keeping errors, which looks like one imprecise computation with a reliable
  low-order pathway. Test 2 gives "rarer, not smaller", which is what attempts plus selection
  predicted and not what a gradually sharpening computation predicted.
- Both fit: each sample either lands exactly or makes a slip drawn from the same structured
  distribution, and filler raises the chance of landing exactly without changing what a slip looks
  like. That matches the earlier "settling" result (filler makes the model converge on one answer).
- It does not settle attempts vs one computation: all-or-nothing success with unchanged slips is
  also what a single computation with a fixed failure mode would give if filler only made that
  failure less likely.

---

# Independence test: err_filler vs err_0^N across items (2026-10-04)

Human-requested. `python scripts/followup_independence.py` ($0; Phase 3 M=300 and the dose run).
Figure: `scripts/followup_independence.png`. Numbers: `data/results/followup_independence.json`.

Was this already run as maj@k? No. maj@k tested a majority vote over output samples, which pushes
items below 50% further down. err_0^N (any one of N attempts right, and recognised) improves every
item, so it is a different prediction and worth running.

## Model-free view (bins from half A of the no-filler samples; no-filler error read from half B)

| No-filler errors in half A | Items | No-filler error (half B) | Filler error | Ratio | Implied N | Ends <= 25% error | Ends >= 75% error |
|---|---|---|---|---|---|---|---|
| 0 of 8 | 80 | 3.4% | 1.5% | 0.43 | 1.2 | 99% | 1% |
| 1–2 of 8 | 38 | 19.1% | 5.4% | 0.28 | 1.8 | 92% | 3% |
| 3–5 of 8 | 38 | 52.6% | 34.5% | 0.66 | 1.7 | 55% | 24% |
| 6–7 of 8 | 53 | 82.8% | 50.2% | 0.61 | 3.6 | 43% | 43% |
| 8 of 8 | 91 | 95.2% | 77.5% | 0.81 | 5.2 | 15% | 74% |

- **Slope of the mean relation: 1.18 [0.83, 1.89]** (log-log through the bin means, bootstrap over
  items). That is in the range named in advance as ambiguous, and its interval includes 1.
- **No single N fits:** the N implied by each bin runs from 1.2 for easy items to 5.2 for hard ones.
  A proportional reduction does not fit either: the ratio runs from 0.28 to 0.81.
- Split-half IV slope on per-item log rates: 1.05 [0.96, 1.16] on all items with smoothed logs;
  0.75 [0.45, 1.15] on the 68 items not at 0% or 100% in either condition (naive OLS 0.62, showing
  the attenuation the IV corrects).

## Latent-variable binomial fit (do not read the slope at face value)
General fit: N = 6.2 [3.7, 25], c = 2.1. Pure attempts: N = 3.6 [2.7, 4.9]. Slope fixed at 1:
c = 0.95. All three fit badly. The large N is an artefact: outcomes within a bin are all-or-nothing
(last two columns above), and a steep power law is the closest a smooth curve can get to a step.
Fitted N also rises with filler length (1.5, 2.2, 3.0 at k = 10, 25, 50 on the 219 dose items),
which tracks the number of items that have flipped, not a number of attempts.

## Predictive check: can either model produce all-or-nothing items?
Share of items ending mostly right (<= 25% error) / in between / mostly wrong (>= 75%) with filler:

| No-filler error | Items | Observed | Independent attempts (N = 3.6) | One computation (slope 1) |
|---|---|---|---|---|
| <= 25% | 116 | 97% / 2% / 2% | 100% / 0% / 0% | 95% / 5% / 0% |
| 26–74% | 40 | 68% / 12% / 20% | 84% / 16% / 1% | 22% / 64% / 14% |
| >= 75% | 144 | 23% / 14% / 63% | 4% / 31% / 65% | 0% / 11% / 89% |

- Neither model reproduces the key fact: of 144 items that are wrong at least 75% of the time
  without filler, 23% become right at least 75% of the time with filler, while 63% stay wrong.
  Independent attempts predict 4% and put the rest in between; the slope-1 model predicts 0%.

## Reading
- On the test as posed (the slope), the result is ambiguous: about 1.2, interval including 1.
- The more informative result is that the relation is not a smooth function of the no-filler
  error rate at all. For a given item, filler either fixes it almost completely or leaves it
  almost completely wrong.
- That contradicts independent attempts **of the kind the model makes without filler**: N draws
  from an item's no-filler success rate cannot turn a 5% item into a 95% item while leaving a
  similar item at 5%. An attempts account survives only if each attempt under filler is itself a
  different, more accurate computation on some items, which is no longer distinguishable from
  "filler changes the computation".
- It equally contradicts a uniform proportional reduction.

---

# Tipping point vs arithmetic difficulty; repeated wrong answers (2026-10-04)

Human-requested. `python scripts/followup_tipping.py` ($0; Phase 3 M=300 plus the dose run).
Figure: `scripts/followup_tipping.png`. Numbers: `data/results/followup_tipping.json`.
Tipping point = smallest k in (0, 10, 25, 50, 100) with pass rate >= 75% (16 samples, T=1).
Chain: x -> c1·x -> y = c1·x ± k1 -> c2·y -> z = c2·y ± k2.

Items by tipping point: right without filler 116; tips at 10: 20; at 25: 22; at 50: 13;
at 100: 11; never: 118.

## 1. Does the tipping point track arithmetic difficulty?

**One coarse feature dominates: the first-hop multiplier.**

| First-hop multiplier | Right without filler | Flips with filler | Never |
|---|---|---|---|
| c1 = 2 ("twice"), 142 items | 102 (72%) | 36 (25%) | 4 (3%) |
| c1 = 3 ("three times"), 158 items | 14 (9%) | 30 (19%) | 114 (72%) |

Mean pass rate by multipliers and filler length:

| c1, c2 | Items | k=0 | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|---|---|
| 2, 2 | 71 | 0.75 | 0.79 | 0.87 | 0.87 | 0.92 |
| 2, 3 | 71 | 0.77 | 0.82 | 0.89 | 0.94 | 0.99 |
| 3, 2 | 69 | 0.23 | 0.35 | 0.39 | 0.41 | 0.45 |
| 3, 3 | 89 | 0.18 | 0.17 | 0.17 | 0.20 | 0.22 |

- 97% of never-flip items have c1 = 3 (Spearman of c1 = 3 with tipping rank: +0.75).
- When the first hop is a doubling, filler fixes almost every failure (36 of 40 items wrong without
  filler flip; pass rate reaches 0.92–0.99).
- When the first hop is a tripling, filler helps only if the second hop is a doubling (0.23 -> 0.45)
  and does almost nothing when both are triplings (0.18 -> 0.22).
- The second-hop multiplier barely matters when the first hop is a doubling (0.75 vs 0.77 at k=0).
  So it is tripling in the *first* hop that is costly, not tripling as such.
- Greedy, 600 items: c1 = 2 goes 83.9% -> 94.7% with dots k=100; c1 = 3 goes 26.0% -> 35.2%.
  V4.1 Flash shows the same split (58.9% -> 75.4% vs 29.5% -> 40.6%).

**Finer difficulty features add little or nothing.**
- Within c1 = 3 (flips vs never): carries 3.6 vs 4.0 (p = 0.14), digits in y p = 0.2, first product
  p = 0.1. Only the second hop matters (c2 = 3 in 33% of flippers vs 65% of never; p = 0.002).
- Within c1 = 2 (right vs flips): carries 3.5 vs 3.5 (p = 0.91), products and digits p > 0.4.
- Across all 300 items carries correlate weakly with tipping rank (+0.14), which disappears within
  multiplier classes.

**Among items that do flip, the tipping length is unrelated to any feature** (66 items; adjusted
R² of all features about 0; carries +0.13, p = 0.32; c1 = 3 +0.15, p = 0.24). One nominal hit
(digits in the answer, p = 0.046) among twelve tests.

Against the two readings offered:
- "Harder items tip later, monotonically, and never-flippers are the hardest": the second half
  holds at the level of the multipliers (never-flippers are almost all first-hop triplings, mostly
  with a second tripling). The first half does not: tipping length is not ordered by difficulty.
- "Unrelated to arithmetic difficulty": false for *whether* an item flips, true for *when*.

## 2. For items that stay wrong, is it the same wrong answer every time?
91 items (pass rate <= 25% at both k=0 and k=100).

| Filler length | Mean share of samples taken by the top wrong answer | Items where it is >= 75% | Items where it is <= 25% |
|---|---|---|---|
| k=0 | 38% | 8% | 38% |
| k=10 | 39% | 8% | 35% |
| k=25 | 43% | 11% | 27% |
| k=50 | 52% | 22% | 18% |
| k=100 | 60% | 35% | 15% |

- **Partly, and increasingly with filler.** Without filler only 8% of these items repeat one wrong
  answer in at least three-quarters of samples; with 100 dots 35% do (32 items), and 65% repeat one
  in at least half.
- For those 32 items the repeated answer is a near miss with the same parity as the truth in 25,
  a structural error (sign, coefficient, dropped step) in 4, a binding error in 2. Median distance
  from the truth 33; below the truth in 66%.
- It was already the most common wrong answer without filler in 56% of the 32 (mean share 34% of
  no-filler samples), and at k=50 in 81%.
- So for about a third of stay-wrong items filler produces a deterministic miscomputation, usually
  not a recognisable structural mistake. The rest remain scattered.

## Reading
- Whether filler can fix an item is governed by how much work the first hop is. That is a
  capacity-like pattern and the first result in this project that points at a specific stage.
  Interpretation (untested): the first hop must finish early enough for the second hop to use it;
  a tripling does not, and filler gives it room, unless the second hop is also expensive.
- The pattern is a threshold on the multipliers, not a smooth dependence on carries or size, and it
  does not explain when a fixable item tips.
- Growing repetition of one wrong answer supports "filler changes which computation runs" over
  pure noise reduction for a third of stay-wrong items.
- Caveat: "twice" and "three times" also differ as words and in the size of y, and c1 = 2 fixes
  y's parity; the data cannot separate these from the amount of arithmetic.

---

# Item-level checks within each (c1, c2) cell; earlier aggregates re-split by c1 (2026-10-04)

Human-requested. `python scripts/followup_by_cell.py` ($0; Phase 3 M=300, greedy N=600).
Figure: `scripts/followup_by_cell.png`. Numbers: `data/results/followup_by_cell.json`.

## A. Within each cell (dots k=100 vs no filler, 16 samples per item)

| Cell (c1, c2) | Items | Pass rate | Right / between / wrong, no filler | Right / between / wrong, filler | Items rising >= 50 pts | Items within ±25 pts |
|---|---|---|---|---|---|---|
| 2, 2 | 71 | 0.75 -> 0.92 | 69% / 17% / 14% | 92% / 3% / 6% | 17% | 69% |
| 2, 3 | 71 | 0.77 -> 0.99 | 75% / 7% / 18% | 100% / 0% / 0% | 21% | 75% |
| 3, 2 | 69 | 0.23 -> 0.45 | 13% / 13% / 74% | 33% / 22% / 45% | 23% | 51% |
| 3, 3 | 89 | 0.18 -> 0.22 | 6% / 16% / 79% | 15% / 11% / 74% | 6% | 79% |

("right" = at most 25% error, "wrong" = at least 75% error.)

**The deciding cell, c1 = 3, c2 = 2: heterogeneous, not a uniform shift.**
- Per item: 23% rise by >= 50 points, 17% rise by 25–49, 51% stay within ±25, 9% fall by >= 25.
- SD of per-item change: 0.364 observed vs 0.134 [0.113, 0.156] expected from a uniform +21-point
  shift plus sampling noise.
- Of the 51 items wrong at least 75% of the time without filler: 22% end mostly right, 27% in
  between, 51% stay mostly wrong. Independent attempts predict 4% / 35% / 61%; a uniform shift
  0% / 49% / 50%; proportional error 0% / 34% / 66%.
- So the item-level switch is real inside this cell, though less clean than the pooled picture:
  about a quarter of items land in between.

Other cells:
- c1 = 2 (both c2): nearly every failing item flips fully (of items wrong >= 75% without filler,
  70% and 100% end mostly right). Spread of per-item change is 2.5–2.8 times the uniform-shift value.
- c1 = 3, c2 = 3: almost nothing moves (6% rise >= 50 points).

Independence slope within cells (split-half IV on log error; > 1 would favour attempts):
(2,2) 0.47 [0.22, 0.76]; (2,3) 0.11 [−0.01, 0.24]; (3,2) 0.98 [0.46, 1.65]; (3,3) 1.47 [1.17, 2.19].
- In the c1 = 2 cells the slope is well **below** 1: the items with the highest error improve the
  most, the opposite of what attempts predict.
- The pooled slope of about 1.1–1.2 was a blend of these. The (3,3) slope rests on a cell where
  little moves at all.
- As before, the fitted "attempts N" per cell (3.7, 17.8, 4.1, 1.7) reflects all-or-nothing items,
  not a number of attempts.

**How much of the pooled bimodality was composition?** A good part: most "flipped" items are
c1 = 2 and most "stuck" items are c1 = 3 with c2 = 3. But it is not only composition: the spread
within every cell is 2–3 times what a uniform shift would give.

## B. Earlier aggregates re-split by c1

Error mix (sampled; share of all samples, relative change with 95% CI over items):

| | Binding error | Arithmetic slip | Other | All wrong |
|---|---|---|---|---|
| c1 = 2, no filler -> filler | 0.7% -> 0.2% (−73% [−100, −25]) | 20.0% -> 4.2% (−79% [−91, −65]) | 3.5% -> 0.1% (−97% [−100, −91]) | 24.2% -> 4.4% (−82%) |
| c1 = 3, no filler -> filler | 2.7% -> 2.6% (−3% [−60, +68]) | 62.6% -> 50.8% (−19% [−27, −11]) | 14.7% -> 14.6% (−1% [−24, +25]) | 79.9% -> 67.9% (−15%) |

- **The earlier "all error types fall by a similar proportion" was a blend.** With a first-hop
  doubling, filler removes about four-fifths of every error type, binding errors included (15 -> 4).
  With a first-hop tripling, only arithmetic slips fall, by about a fifth; binding and "other"
  errors do not move.
- Greedy 600-item run agrees: c1 = 2 slips 37 -> 12, other 9 -> 2; c1 = 3 slips 190 -> 155,
  binding 8 -> 8, other 35 -> 41.
- The binding-heavy variant cannot be re-split: its coefficient was fixed at 2, so every item
  there has c1 = 2. Its result (binding errors are fixed too) agrees with the c1 = 2 row here.

Parity retention (share of non-binding wrong answers with the answer's parity):
(2,2) 97.1% -> 100%; (2,3) 97.2% -> 88.9% (9 answers); (3,2) 95.5% -> 99.8%; (3,3) 95.0% -> 97.6%.
Stable across cells; nothing changes in the earlier conclusion.

Slip size (median |answer error|, non-binding wrong answers):
- c1 = 2: 24 -> 8 (quartiles 11–44 -> 2–22; only 97 slips remain, from few items).
- c1 = 3: 30 -> 30 (quartiles 12–57 -> 12–60).
- So "rarer, not smaller" holds for c1 = 3, which supplied most of the slips. For c1 = 2 the few
  slips that survive filler are smaller.

## Revised headline
Filler almost completely fixes items whose first hop is a doubling (errors 24% -> 4%, every error
type). When the first hop is a tripling and the second a doubling, it fully fixes about a fifth of
items and partly helps another fifth. With two triplings it does almost nothing.

---

# Hop-cost experiment (2026-10-04)

Human-requested (new task variants and multipliers). Design and criteria written BEFORE data collection.

## Design
V4 Flash via OpenRouter (Parasail fp8), thinking off, 10-shot, greedy, max_tokens=10. New generator in
`scripts/followup_hopcost.py` (same layout as the main task: 5 variables, literals 10–99, constants
1–50, non-negative values, nonsense names), seed 20261004.
1. **One-hop cost.** Question asks for c·x ± k of a literal variable x. c in {2, 3, 4, 5, 10, 11},
   100 items each, no filler. One-hop accuracy is the cost measure for that multiplier.
2. **Two-hop.** y = c1·x ± k1, answer = c2·y ± k2, c1 in {2, 3, 4, 5, 10, 11} crossed with c2 in
   {2, 3}, 100 items per cell, no filler vs 100 dots.
3. **Wording control.** Same items written symbolically ("xoc = 3·zab + 14", "What is 2·xoc - 35?"),
   c1 in {2, 3, 4, 10} x c2 in {2, 3}, 50 items per cell, no filler vs 100 dots; plus one-hop
   symbolic, 50 items per multiplier.
Few-shot examples come from the same variant (same wording, same hop count, mixed multipliers) and
show the same filler condition.

## Pre-registered criteria
Fixability of a cell = share of no-filler errors removed by filler = 1 − err_filler / err_0.
- **Cost predicts fixability** if, across the six first-hop multipliers, Spearman correlation between
  one-hop error and two-hop fixability is <= −0.8 for both c2 values. **Not supported** if it is
  above −0.5 for either.
- **Cost predicts baseline** if the same correlation between one-hop error and two-hop no-filler
  error is >= +0.8 for both c2 values.
- **Wording:** the twice / three-times gap is lexical if, in symbolic wording, no-filler two-hop
  accuracy for c1 = 3 is within 15 points of c1 = 2 (averaged over c2). It is arithmetic if the gap
  stays >= 30 points.
- **Parity:** if a forced parity of y is what makes c1 = 2 easy, c1 = 4 and c1 = 10 should match
  c1 = 2 (within 15 points, no filler). If cost is what matters, x10 should be at least as easy as
  x2 and x4 should be clearly harder (>= 15 points lower).
- Stated limit: larger multipliers also make y and the answer larger, which raises second-hop cost;
  one-hop accuracy does not capture that.

---

# Single hops in isolation for the first-hop-tripling items (2026-10-04)

Human-requested add-on. Design and criteria written BEFORE data collection; run only after the
hop-cost experiment finishes, without changing it.

## Design
- Items: the 158 items with c1 = 3 among the 300 Phase 3 items (69 with c2 = 2, 89 with c2 = 3).
- **First hop alone:** the item's own definitions, unchanged, with the question "What is the number
  for <y's name>?" Correct answer: y.
- **Second hop alone:** y's definition replaced by its correct literal value ("yaj = 114"),
  everything else and the original question unchanged. Correct answer: the original answer.
- No filler (bare `Filler:` line), T=1, 16 samples per item per hop, 10-shot with the fixed few-shot
  items transformed the same way. 158 x 2 x 16 = 5,056 calls.
- p1, p2 = per-item pass rates of the two single hops; p0, pf = existing two-hop pass rates without
  filler and with 100 dots.

## Pre-registered criteria
The "first hop finishes too late" account says each hop is doable alone, composition in one pass is
what fails, and filler restores composition up to what the single hops allow.
- **Composition deficit:** supported if mean p1·p2 exceeds mean p0 by >= 25 points.
- **Single hops set the ceiling under filler:** supported if Spearman(pf, p1·p2) across items is
  >= 0.5, and items with p1·p2 >= 0.75 have mean pf >= 0.60 while items with p1·p2 <= 0.25 have
  mean pf <= 0.15. Not supported if the correlation is < 0.25.
- **Stuck items explained by a hard single hop:** supported if >= 70% of items that stay wrong with
  filler (pf <= 0.25) have p1·p2 < 0.5. If instead most stuck items have p1·p2 >= 0.75, single-hop
  ability does not explain them and composition fails even with filler.
- Also reported: which hop is the weaker one for stuck items, and the same split by c2.

## Hop-cost experiment: results — `python scripts/followup_hopcost.py`
Cost $0.42 (cumulative $8.02 of $30). 2,400 items, 4,000 calls, parse rate >= 99.9%, no reasoning tokens.
Figure: `scripts/followup_hopcost.png`. Numbers: `data/results/followup_hopcost.json`.
Items: `data/tasks/hopcost_items.jsonl`.

**The pre-registered cost measure failed: one-hop accuracy is 100% for every multiplier**
(x2, x3, x4, x5, x10, x11 in words, 100 items each; x2, x3, x4, x10 symbolic, 50 each). With no
variance the two correlation criteria cannot be computed. (The script printed "in between" and
"not supported" for them; those labels are artefacts of the undefined correlation and should be
read as "not computable".)

What that result does show: **every hop is easy on its own; all the difficulty is in composing
two hops in one pass.**

Two-hop accuracy, words wording, 100 items per cell (no filler -> 100 dots):

| First hop | Second hop x2 | Second hop x3 | Averaged | Share of errors removed (x2 / x3) |
|---|---|---|---|---|
| x2 | 82% -> 94% | 87% -> 100% | 84% -> 97% | 67% / 100% |
| x3 | 48% -> 59% | 26% -> 25% | 37% -> 42% | 21% / −1% |
| x4 | 16% -> 23% | 20% -> 26% | 18% -> 24% | 8% / 8% |
| x5 | 30% -> 31% | 26% -> 24% | 28% -> 28% | 1% / −3% |
| x10 | 37% -> 55% | 24% -> 38% | 30% -> 46% | 29% / 18% |
| x11 | 18% -> 21% | 11% -> 15% | 14% -> 18% | 4% / 4% |

Pre-registered verdicts:
- **Cost predicts fixability / baseline: not computable** (see above).
- **Wording: arithmetic, not lexical.** The x2 vs x3 gap without filler is 48 points in words and
  59 points in symbolic wording (80% vs 21%). Symbolic x4 is 26% and x10 22%.
- **Parity: neither criterion met.** x4 and x10 force y's parity exactly as x2 does, yet they are
  66 and 54 points below x2 without filler. Forced parity is not what makes x2 easy. The "cost"
  criterion also fails as written, because x10 is not as easy as x2.

Post-hoc observations (not pre-registered):
- **A first-hop doubling is in a class of its own**: 84% without filler against 14–37% for every
  other multiplier, in both wordings.
- **Size of y matters too, but does not explain that.** At matched y (100–199): x2 80%, x10 53%,
  x3 44%, x5 29%, x11 26%, x4 12% without filler. Within a multiplier accuracy falls as y grows
  (x3: 42% for y < 100 down to 9% for y of 300–499; x10: 53–62% for y of 100–299, 23% for y >= 500).
- **Filler helps in roughly the order of intuitive arithmetic cost at matched y**: almost fully for
  x2, substantially for x10 (76% at y of 100–299 with filler), partly for x3 when the second hop is a
  doubling, and little or nothing for x4, x5, x11.
- Wrong answers keep the answer's parity for every multiplier (94–100%); median error size is
  18–24 throughout.

Reading:
- The single-hop result rules out "the first hop is hard" in the plain sense. What differs between
  multipliers is how well a hop composes with a second one inside one pass.
- The ordering x2 > x10 > x3 > x4, x5, x11 and the y-size effect fit "the first hop has to produce
  a usable y early", with cheaper first hops and smaller intermediates leaving more room. That
  remains an interpretation: this experiment's planned test of it could not be run.

## Single hops in isolation: results — `python scripts/followup_singlehop.py`
Cost $0.34 (cumulative $8.36 of $30). 158 items, 5,056 calls, parse rate 100%, no reasoning tokens.
Figure: `scripts/followup_singlehop.png`. Per item: `data/results/followup_singlehop_items.csv`.

**Every single hop is answered correctly in every sample: 2,528 of 2,528 for the first hop alone and
2,528 of 2,528 for the second hop alone**, on all 158 items.

| | First hop alone | Second hop alone | Two-hop, no filler | Two-hop, 100 dots |
|---|---|---|---|---|
| All 158 items (first hop x3) | 1.00 | 1.00 | 0.20 | 0.32 |
| Second hop x2 (69 items) | 1.00 | 1.00 | 0.23 | 0.45 |
| Second hop x3 (89 items) | 1.00 | 1.00 | 0.18 | 0.22 |

Pre-registered verdicts:
- **Composition deficit: supported.** Product of the single hops minus the two-hop no-filler pass
  rate is +0.80 [+0.76, +0.84] (bar was 0.25).
- **Single hops set the ceiling under filler: not computable**, because the single-hop product is
  1.00 for every item. Substantively the answer is no: the ceiling the single hops allow is 100%,
  and filler reaches 32%.
- **Stuck items explained by a hard single hop: not supported.** All 97 items that stay wrong with
  filler do both hops perfectly alone. Composition fails for them even with filler.

Reading:
- For these items nothing about either hop is hard. The model can find y's definition, resolve x,
  triple, add, and do the second hop from a given y, every time. It cannot do the two in one pass.
- Filler recovers only a small part of that deficit (0.20 -> 0.32 of a possible 1.00). Writing the
  intermediate down, which is what the second-hop-alone condition amounts to, recovers all of it.
- Which items filler fixes is not explained by single-hop ability, the multipliers aside. The
  item-level difference lies in composition itself.
- The earlier "arithmetic slip" label needs care: these are not failures of arithmetic the model
  cannot do. They are near misses that keep the answer's parity, produced only when the two hops
  share one pass. One interpretation (untested): the second hop runs on a value of y that is not
  yet exact.

---

# Filler placement (2026-10-04)

Human-requested ("do the black-box, partial"). Design and criteria written BEFORE data collection.

## Design
V4 Flash via OpenRouter (Parasail fp8), thinking off, 10-shot, greedy, the frozen 600 items.
100 dots in every filler condition. Every few-shot example shows the same placement as the test item.
- **No filler** and **after the question** (the standard placement): reused from Phase 2.
- **Before everything:** a `Filler: . . .` line above the definitions.
- **Between definitions and question:** a `Filler: . . .` line after the last definition and before
  `Question:`.
In the two new conditions the usual bare `Filler:` line before `Answer:` is kept, so the end of the
prompt is identical to the no-filler baseline. The system prompt is unchanged (it describes filler
as following the question; noted as a small mismatch for the new placements).

## Pre-registered criteria (paired McNemar on 600 items; "standard gain" = after-question gain)
- **Room must come after the question:** both new placements gain less than 3 points over no
  filler with p > 0.05.
- **Room after the definitions is enough:** the between placement gains at least half the standard
  gain with p < 0.01, and the before-everything placement does not.
- **Not about computing on the item at all:** the before-everything placement gains at least half
  the standard gain with p < 0.01. This would count against every computation account.
- Anything else: "no criterion met". Results are also reported by first-hop multiplier.

## Filler placement: results — `python scripts/followup_placement.py`
Cost $0.21 (cumulative $8.57 of $30). 1,200 new calls, parse rate >= 99.8%, no reasoning tokens.
Figure: `scripts/followup_placement.png`. Numbers: `data/results/followup_placement.json`.

| Placement of 100 dots | Accuracy | Gain vs no filler | wrong->right / right->wrong | McNemar p | First hop x2 | First hop x3 |
|---|---|---|---|---|---|---|
| No filler | 53.5% | — | — | — | 83.9% | 26.0% |
| Before everything | 44.3% | −9.2 pp | 29 / 84 | 2e-7 | 72.6% | 18.7% |
| Between definitions and question | 51.0% | −2.5 pp | 53 / 68 | 0.20 | 83.5% | 21.6% |
| After the question (standard) | 63.5% | +10.0 pp | 89 / 29 | 3e-8 | 94.7% | 35.2% |

**Pre-registered verdict: "no criterion met".** The "room must come after the question" criterion
required both new placements to be within 3 points of no filler with p > 0.05. The between
placement satisfies that; the before-everything placement does not, because it is significantly
*worse* than no filler, which the criteria did not anticipate.

Substance:
- **Filler helps only after the question.** Neither other placement gives any gain; both are far
  below the standard placement (p = 1e-12 and 5e-24).
- Filler between the definitions and the question does nothing, in both multiplier classes
  (x2: 83.5% vs 83.9%; x3: 21.6% vs 26.0%, p = 0.12). Positions that can see the definitions but
  not the question do not pre-compute anything the answer can use.
- Filler before everything hurts by 9 points, in both classes. Not investigated; the prompt format
  in that condition differs most from what the system prompt describes.
- The "prompt-format or priming" explanation (gain from the mere presence of dots) is ruled out.

Reading: consistent with the serial-capacity account, in which filler positions work on the
queried variable and therefore need the question. It does not distinguish that account from any
other in which filler positions compute on the question.

---

# Layer of emergence at the answer position: proxy only (2026-10-04)

Human asked for the layer at which y emerges at the answer position on V4 Flash, with and without
filler, for flip vs stay-wrong items. **That measurement was not made: it needs model activations,
and there is no model access.** What follows is the nearest thing available from released data.

`python scripts/followup_brauer_layers.py` ($0). Source: Brauer et al. `release/logit_lens_heatmaps/`
(repo commit a701927). Figure: `scripts/followup_brauer_layers.png`.
Limits: DeepSeek V3, 2-fact addition (intermediates = retrieved addends A1, A2; computed value =
the sum), filler conditions only (**no no-filler heatmap is released**; the paper shows one only as
Figure 7), correct vs incorrect examples instead of flip vs stay-wrong, aggregate shares per
(layer, position) instead of per-example emergence.
"Emergence layer" = first layer at which the share of examples decoding the target reaches 10%.

| Condition | n | Sum at answer position | Sum at best filler position | A1 / A2 at filler positions | A1 / A2 at answer position |
|---|---|---|---|---|---|
| dots 10, correct | 244 | L44 | L53 | L32 / L33 | L42 / L42 |
| dots 25, correct | 245 | L44 | L53 | L32 / L32 | L42 / L42 |
| dots 50, correct | 246 | L44 | L52 | L32 / L32 | L42 / L42 |
| counting 10, correct | 235 | L44 | L54 | L32 / L32 | L42 / L42 |
| counting 25, correct | 214 | L44 | L48 | L32 / L32 | L42 / L43 |
| dots 10, incorrect | 756 | never | never | L40 / L40 | L42 / L42 |
| dots 50, incorrect | 754 | never | never | L39 / L40 | L42 / L42 |
| counting 10, incorrect | 765 | never | never | L40 / L40 | L42 / L42 |

- **The computed value does not emerge earlier at the answer position with more filler.** The sum's
  layer profile at the answer position is the same at 10, 25 and 50 dots (10% at L44, 25% at
  L55–57, 50% at L58, 90–99% only at L60).
- **The computed value is not handed over from the filler.** At filler positions the sum is weak
  (at most 21–43% of examples) and appears later (L48–54) than at the answer position (L44).
- **The intermediates are what sit in the filler, and they are there early.** On correct examples
  A1 and A2 are decodable at filler positions from L32, ten layers before they are at the answer
  position (L42).
- **On incorrect examples the intermediates reach the filler later:** L39–40 instead of L32, in all
  three condition pairs, although they get about as strong eventually. At the answer position they
  appear at L42 either way.

Reading, with heavy caveats:
- The pattern "intermediates early at filler positions on successes, about eight layers later on
  failures" is what an offloading account would expect: the intermediate has to be ready in the
  filler early enough for the answer position to use it.
- It is equally what "easy facts are retrieved early and answered correctly" would give, with
  filler playing no causal role. Without a no-filler comparison these cannot be separated.
- It says nothing about whether y emerges earlier at the answer position with filler than without,
  which was the question.
