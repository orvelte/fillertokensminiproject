# Log

## 2026-10-04 — Phase 0
- Ran: `python scripts/phase0.py` (V1–V4 on Fireworks `deepseek-v4p1-flash`, thinking disabled),
  plus a few manual curl probes (thinking toggle, official DeepSeek API, OpenRouter model list).
- Cost: $0.02. Raw responses: `data/results/phase0.jsonl`.
- Result: V1 PASS, V2 PASS, V3 PASS (n>1 unsupported, prefix caching partial), V4 no route.
- Seeds: none used yet (stand-in items are Brauer's shipped dataset, seed 42).
- Surprises:
  - Served model is V4.1 Flash, not V4 Flash (V4 Flash retired on the official API 2026-09-10).
  - T=0 is not repeatable on hard items (2/5 identical over 4 repeats).
  - `DEEPSEEK_API_KEY` fails authentication.
- Decision: STOP 0, waiting for human (model choice; see `notes/phase0.md`).

## 2026-10-04 — STOP 0 decision
- Human decision: use **V4 Flash via OpenRouter**; accept non-deterministic T=0 in Phase 2.
- Chosen model id: `deepseek/deepseek-v4-flash` (0423 snapshot = HF `deepseek-ai/DeepSeek-V4-Flash`,
  which the J-lens README names as its model). The 0731 snapshot is a different checkpoint.
- Pinned to one upstream host (Parasail: fp8, supports seed + logprobs, $0.14 / $0.28 per 1M) with
  fallbacks off, because OpenRouter hosts mix fp4 / fp8 / unknown quantizations.
- PDFs moved to `paper/` (gitignored). `neelsprompt.md` gitignored.
- Next: re-run `python scripts/phase0.py` on OpenRouter once `OPENROUTER_API_KEY` is in `.env`.

## 2026-10-04 — Phase 0 re-run on V4 Flash (OpenRouter / Parasail)
- Ran: `python scripts/phase0.py` (default provider now `openrouter`).
- Cost: $0.02 (cumulative $0.04). Raw: `data/results/phase0.jsonl`.
- Result: V1 PASS, V2 PASS, V3 PASS (n>1 ignored, caching partial), V4 no route.
- Surprise: T=0 not repeatable (3/5 hard items identical over 4 repeats).
- Decision: STOP 0 (re-report on the new model); waiting for human OK to start Phase 1.

## 2026-10-04 — Phase 1
- Ran: `python scripts/phase1.py` (max_coef 3, const 1–50, 5 terms; seed 20261004).
- Cost: $0.01 (cumulative $0.05). Raw: `data/results/phase1_pilot_coef3_const50_terms5.jsonl`.
- Result: k=0 greedy accuracy 50/100; in target on round 1, generator frozen. N=600 eval + 10 few-shot written.
- Decision: STOP 1, waiting for human.

## 2026-10-04 — Phase 2
- Ran: `python scripts/phase2.py pilot`, then `python scripts/phase2.py full --conds counting_25 dots_100`.
- Cost: $0.26 (cumulative $0.31). Raw: `data/results/phase2_{pilot,full}.jsonl`; figures `scripts/phase2_uplift_{pilot,full}.png`.
- Result: GATE PASS. N=600: k=0 53.5%, dots 100 63.5% (p=3e-8), counting 25 62.2% (p=1e-6).
- Decision: recommend F* = dots k=100. STOP 2, waiting for human.

## 2026-10-04 — Phase 3 pilot (M=60)
- Ran: `python scripts/phase3.py --M 60`; `python scripts/phase3_analysis.py --M 60`. F* = dots k=100 (human "go" at STOP 2).
- Seeds: analysis bootstrap seed 0; no API seed (samples indexed 0–15 in the cache key).
- Cost: $0.37 (cumulative $0.68). Raw: `data/results/phase3_M60.jsonl`; figure `scripts/phase3_delta_vs_p0_M60.png`.
- Result: T1 PASS (scaled), T2 PASS, T3 FAIL at pilot size. Wrong-modal items improve (+19 pp [+6, +33]).
- Surprise: filler effect is concentrated in ~1 in 6 items that flip from near 0 to high pass rate, including items where the correct answer was never sampled without filler.
- Decision: waiting for human OK to run M=300.

## 2026-10-04 — Phase 3 full (M=300)
- Ran: `python scripts/phase3.py --M 300`; `python scripts/phase3_analysis.py --M 300`.
- Cost: $1.48 for the 240 new items (Phase 3 total $1.85; cumulative $2.16).
- Raw: `data/results/phase3_M300.jsonl`; figure `scripts/phase3_delta_vs_p0_M300.png`.
- Result: wrong-modal items improve +22.9 pp [+17.0, +29.2]; voting and temperature predict ~0. T1 marginal FAIL (52 vs 60), T2 PASS, T3 FAIL on the 5 pp guide but predictions separated. Verdict (a) with caveat.
- Surprises: correct answers appear under filler on items with 0/48 no-filler correct (14 of 42); filler accuracy is nearly temperature-insensitive while no-filler accuracy is not.
- Decision: STOP 3, waiting for human.

## 2026-10-04 — Phase 3 T1 fix attempt (harder set, pilot M=60)
- Human decision at STOP 3: raise difficulty once and re-run the pilot.
- Ran: `python scripts/phase1.py --max-coef 4 --suffix _hard`; `python scripts/phase3.py --M 60 --suffix _hard`; `python scripts/phase3_analysis.py --M 60 --suffix _hard`. Seed 20261004 (same as frozen set).
- Cost: $0.36 (cumulative $2.52). Raw: `data/results/phase3_M60_hard.jsonl`.
- Result: greedy baseline 35%. T1 still FAIL (10 / 11 mid-range items vs 12 needed). Wrong-modal gain replicates (+12.7 pp [+5.4, +20.9]).
- Decision: no further difficulty changes (one raise allowed). Waiting for human.

## 2026-10-04 — White-box go-ahead (Phase 5 spike not started)
- Human decision: proceed with white-box.
- Checked: `deepseek-ai/DeepSeek-V4-Flash` is ungated, 160 GB (FP4 experts, FP8 elsewhere), ships its own
  inference code (`inference/model.py`, `kernel.py`), 43 layers, d_model 4096, `hc_mult` 4, per-layer
  `compress_ratios` of 4 / 128, `sliding_window` 128. J-lens and R-lens are 1.4 GB each, target layer 41.
- Blocker: local machine is an M5 Pro with 24 GB RAM; no GPU host or credentials available. Spike needs
  a multi-GPU server. Waiting for human to provide compute and a white-box budget.

## 2026-10-04 — Follow-up on existing data (human-approved options 1 and 2)
- White-box spike judged infeasible by human (time and budget). Approved: free analyses only.
- Ran: `python scripts/followup_existing_data.py`. Cost $0 (cumulative $2.52).
- Result: 85% of errors are a wrong y with a correct final step; filler mainly reduces those. Filler collapses answers to one value per item (right or wrong). Dots-100 and counting-25 fix the same items (61 both vs 26.5 expected).
- Decision: waiting for human on options 3–4 (logprobs, dose-response; about $1).

## 2026-10-04 — Follow-up: per-item dose-response (human-approved)
- Ran: `python scripts/followup_dose.py` (dots k=10/25/50, T=1, K=16, 219 items not at ceiling). No API seed; bootstrap not used.
- Cost: $1.05 (cumulative $3.56). Raw: `data/results/followup_dose.jsonl`; figure `scripts/followup_dose.png`.
- Result: answer diversity falls smoothly with k for all item kinds; flip items mostly jump at item-specific k (10 to 100); 23% non-monotone; stays-wrong items stay wrong at every k but their settled answer varies with k.
- Decision: waiting for human.

## 2026-10-04 — Claim summary figures
- Ran: `python scripts/claims_figures.py` (existing results only, $0). Bootstrap seed 0.
- Wrote: `scripts/claim1_uplift.png`, `claim2_not_vote_not_sharpen.png`, `claim3_shared_across_fillers.png`, `claim4_error_types.png`. Claim 4's dose-response panel is the existing `scripts/followup_dose.png`.

## 2026-10-04 — Follow-up: binding / arithmetic slip / other (human-requested)
- Ran: `python scripts/followup_error_mix.py` ($0). Bootstrap seed 0. Figure `scripts/followup_error_mix.png`.
- Result: arithmetic slips are 79% of errors and 80% of the errors filler removes (−32% [−40, −25]); binding errors are 3% of errors, change not measurable (−16% [−65, +44]). Proportionality test not discriminating on this task.

## 2026-10-04 — Follow-up: binding-heavy variant (human-approved)
- Ran: `python scripts/phase1.py` with `--num-terms 10 --suffix _t10`, `--num-terms 10 --similar-names --suffix _t10sim`, `--num-terms 6 --similar-names --suffix _t6sim`, `--max-coef 2 --const-max 9 --num-terms 15 --similar-names --suffix _easy15sim`, `--max-coef 2 --const-max 9 --num-terms 10 --similar-names --suffix _easy10sim` (seed 20261004; names seed 20261005); then `python scripts/followup_binding.py`.
- Cost: $0.31 (cumulative $3.87). Raw: `data/results/followup_binding.jsonl`; figure `scripts/followup_binding.png`.
- Result: on `_easy15sim`, 23.2% -> 44.8% (p=4e-24). All error types fall by a similar proportion; binding errors are fixed per item at least as often as arithmetic slips.
- Surprises: more distractors/similar names lower accuracy mainly through near-miss answers, not binding errors; accuracy stays ~30% even with easy arithmetic; filler uplift is about twice as large on the cluttered variant.
- Decision: waiting for human.

## 2026-10-04 — Follow-up: filler disruption by location (human-approved)
- Pre-registered design and criteria in `notes/followup.md` before data collection.
- Ran: `python scripts/followup_disrupt.py` (48 flip items, 6 conditions, T=1, K=16, sample indices 100–115; letter seeds 1000+shot and item idx; bootstrap seed 0).
- Cost: $0.68 (cumulative $4.55). Raw: `data/results/followup_disrupt.jsonl`; figure `scripts/followup_disrupt.png`.
- Result: intact 87%, early 82%, middle 92%, late 90%, all letters 51%, no filler 22%. Verdicts: graceful; location criterion not met (early worse by 8–10 pp, under the 10 pp bar).
- Surprise: middle block slightly improves on intact; random letters alone give +29 pp.
- Decision: waiting for human.

## 2026-10-04 — Follow-up: disruption in a 25-unit filler (human-approved)
- Pre-registered design, primary item set and criteria in `notes/followup.md` before data collection.
- Ran: `python scripts/followup_disrupt25.py` (48 flip items, 7 conditions, T=1, K=16, sample indices 100–115; no-filler samples reused from the 100-unit run).
- Cost: $0.47 (cumulative $5.02). Raw: `data/results/followup_disrupt25.jsonl`; figure `scripts/followup_disrupt25.png`.
- Result (primary, 33 items): intact 25 dots 85%, 17 dots 78%, early 72%, middle 73%, late 79%, all letters 63%, no filler 26%. Verdict: redundant positions.
- Decision: waiting for human.

## 2026-10-04 — Follow-ups: settled-answer centrality, cross-model overlap (human-approved)
- Ran: `python scripts/followup_center.py` ($0); `python scripts/followup_crossmodel.py` ($0.09; Fireworks `deepseek-v4p1-flash`, thinking disabled, greedy).
- Results: no clear evidence the settled answer is numerically central. V4.1 Flash shows uplift (43.5% -> 57.2%, p=6e-11); failures overlap across models (OR 3.5) but filler-fixed items overlap only weakly (OR 1.9, p=0.058).
- Phase 4 pilot (`python scripts/phase4.py pilot`, 100 items): 0 format violations, all fillers 100 tokens; full run launched.

## 2026-10-04 — Phase 4 (human-approved)
- Ran: `python scripts/phase4.py pilot`; `python scripts/phase4.py full`. Seed 20261004 (shuffle, random vocabulary). Tokenizer: `cache/tokenizer/tokenizer.json` from HF `deepseek-ai/DeepSeek-V4-Flash`.
- Cost: $0.88 (cumulative $5.99). Raw: `data/results/phase4_full.jsonl`; figure `scripts/phase4_ladder_full.png`.
- Result: seven fillers match dots (62–65%); counting in words 57.5%, short cycle 57.2%, random vocabulary 55.8%. Prose vs shuffled p=0.13; P+Q vs P+P p=0.64; digits vs words p=5e-7. 1 format violation in 6,600.
- Surprise: highly predictable fillers (word counting, 3-word cycle) are among the worst.
- Decision: STOP 4, waiting for human.

## 2026-10-04 — White-box route: small-model uplift screen (human-approved)
- Ran: `python scripts/whitebox_screen.py`; `python scripts/phase1.py --max-coef 2 --const-max 30 --suffix _easy`; `python scripts/phase1.py --max-coef 2 --const-max 9 --suffix _veasy`; `python scripts/whitebox_screen.py --suffix _veasy`; `--suffix _easy --models <27B trio>`; then `--models qwen/qwen3.5-122b-a10b` on main and `_easy`.
- Cost: $1.61 (cumulative $7.60). Raw: `data/results/whitebox_screen*.jsonl`.
- Result: FAIL. Six lens-equipped models at floor on main/easy tasks; no uplift where there is room (Qwen 27B models at ~20% on `_veasy`).
- Surprise: V4 Flash is worse with constants 1–9 (35%) than 1–30 (78%).
- Decision: cheap white-box route closed; waiting for human.

## 2026-10-04 — Follow-up: Brauer artifact reanalysis (human-requested)
- Ran: `python scripts/followup_brauer.py` ($0). Source: `vendor/filler-token-reasoning` commit a701927, `release/top_tokens/*_2fact_*.json`. Bootstrap seed 0.
- Result: near-miss sums are co-present with the correct sum well above the other-item control (±2: 32% vs 6%; ±10: 17% vs 6%), structured by parity and tens digit; retrieved addends show a much weaker halo.
- Limits: 2-fact task (no system-of-equations decodes released), DeepSeek V3 / Kimi K2, correct examples only, decodes pooled over layers and positions.

## 2026-10-04 — Follow-up: Brauer decodes, offset spectrum (human-requested)
- Ran: `python scripts/followup_brauer_spectrum.py` ($0). Bootstrap seed 0, 1,000 resamples over examples.
- Result: sum halo = parity (+7.0 pp DeepSeek, +5.0 Kimi) plus closeness; no mod 5, no mod 10 beyond parity. Addend halo: ±1/±2 and a mod 10 term, no parity. No no-filler decodes in the release.
- Correction: the earlier "±10 stands out" claim was a parity artefact (neighbour set included odd offsets).

## 2026-10-04 — Follow-up: slips parity / size / spectrum (human-requested)
- Ran: `python scripts/followup_slips.py` ($0). Bootstrap over items, seed 0.
- Result: answer parity kept in ~96% of wrong answers (98–100% with filler), also where not forced (c2=3). Intermediate parity separable only for c2=2: 64% and 44%. Slips get rarer, not smaller (median |dy| 10 vs 10; within-item p=0.95). Spectrum shares parity with the Brauer halo but is much broader and has tens structure.
- Correction: earlier "correct final step on a wrong y" inference from divisibility by c2 is not safe (residue mod c2 is fixed by k2).

## 2026-10-04 — Follow-up: independence test (human-requested)
- Ran: `python scripts/followup_independence.py` ($0). Seed 0; grid NPMLE of per-item error rates; bootstrap over items.
- Result: mean-relation slope 1.18 [0.83, 1.89] (ambiguous). No single N fits (1.2 to 5.2 by bin). Latent-model slope of 6.2 is an artefact of all-or-nothing item outcomes. Predictive check: 23% of items that are >= 75% wrong without filler become <= 25% wrong with it; attempts model predicts 4%, slope-1 model 0%.
- Not previously covered by maj@k (different model).

## 2026-10-04 — Follow-up: tipping point vs difficulty; repeated wrong answers (human-requested)
- Ran: `python scripts/followup_tipping.py` ($0).
- Result: first-hop multiplier dominates (c1=2: 72% right without filler, 25% flip, 3% never; c1=3: 9% / 19% / 72%). c1=3 with c2=3 gets almost no filler benefit (0.18 -> 0.22). Carries, digits and product sizes add nothing within multiplier classes; tipping length among flippers is unrelated to features. Stay-wrong items repeating one wrong answer in >= 75% of samples: 8% without filler, 35% at k=100.
- Surprise: the main task is effectively two sub-populations split by the first-hop multiplier; noted for every earlier aggregate.

## 2026-10-04 — Follow-up: checks within (c1, c2) cells; aggregates re-split by c1 (human-requested)
- Ran: `python scripts/followup_by_cell.py` ($0).
- Result: (3,2) cell is heterogeneous (23% of items rise >= 50 pts, 51% within ±25; SD 0.36 vs 0.13 under a uniform shift). c1=2: all error types fall ~80% incl. binding; c1=3: only slips fall (−19%). Independence slope below 1 in c1=2 cells. Slips shrink only for c1=2.
- Corrections to earlier aggregates noted in `notes/followup.md` (proportional error reduction was a blend).

## 2026-10-04 — Follow-up: hop-cost experiment (human-requested)
- Pre-registered design and criteria in `notes/followup.md`. Ran: `python scripts/followup_hopcost.py` (seed 20261004; greedy; new generator in the script).
- Cost: $0.42 (cumulative $8.02). Raw: `data/results/followup_hopcost.jsonl`; figure `scripts/followup_hopcost.png`.
- Result: one-hop accuracy 100% for all multipliers, so the cost-correlation criteria are not computable. Two-hop no filler -> filler by first multiplier: x2 84 -> 97, x3 37 -> 42, x4 18 -> 24, x5 28 -> 28, x10 30 -> 46, x11 14 -> 18. Wording gap persists in symbolic form (59 pts); forced parity (x4, x10) does not make items easy.
- Surprise: every single hop is at ceiling; difficulty is entirely compositional.
- Launched `python scripts/followup_singlehop.py` (single hops for the c1=3 Phase 3 items) after this run finished.

## 2026-10-04 — Follow-up: single hops in isolation (human-requested add-on)
- Pre-registered in `notes/followup.md`. Ran: `python scripts/followup_singlehop.py` after the hop-cost run finished (T=1, K=16, no filler, sample indices 0–15).
- Cost: $0.34 (cumulative $8.36). Raw: `data/results/followup_singlehop.jsonl`.
- Result: both single hops correct in 2,528/2,528 samples each on all 158 c1=3 items. Composition deficit +0.80. Single-hop ability explains none of the stuck items.

## 2026-10-04 — Follow-up: filler placement (human-requested)
- Pre-registered in `notes/followup.md`. Ran: `python scripts/followup_placement.py` (greedy, 600 items, 100 dots).
- Cost: $0.21 (cumulative $8.57). Raw: `data/results/followup_placement.jsonl`; figure `scripts/followup_placement.png`.
- Result: before everything 44.3% (−9.2, p=2e-7), between definitions and question 51.0% (−2.5, p=0.2), after the question 63.5% (+10.0). Pre-registered verdict: no criterion met (before-everything placement hurts, which the criteria did not anticipate).
- White-box cost estimate given to human: roughly $65–130 best case, $120–500 realistic, for the decode-y measurement on a 4x H100 node.

## 2026-10-04 — Follow-up: layer of emergence (proxy from Brauer's released heatmaps)
- Requested measurement (V4 Flash, y at the answer position by layer, with vs without filler) NOT made: no model access.
- Ran: `python scripts/followup_brauer_layers.py` ($0) on DeepSeek V3 2-fact heatmaps (filler conditions only).
- Result: sum emerges at the answer position at the same layers for 10/25/50 dots (10% at L44); weaker and later at filler positions. Addends at filler positions from L32 on correct examples vs L39–40 on incorrect.

## 2026-10-05 — Archived follow-ups that do not bear on current conclusions (human decision)
- Moved 51 files to `archive/` (gitignored), original relative paths kept; list and reasons in `archive/README.md`.
- Includes, at the human's request, the binding-heavy variant (`followup_binding`, `_easy15sim` task set) and the 100-dot disruption run (`followup_disrupt`).
- Notes above still refer to the original paths under `scripts/` and `data/`.

## 2026-10-05 — Summary figures (human-requested)
- Ran: `python scripts/summary_figures.py` ($0; existing results; seed 0).
- Wrote: `scripts/fig_core_delta_by_group.png`, `scripts/fig_change_histogram.png`, `scripts/fig_disruption.png`.

## 2026-10-05 — Presentation figures (human-requested)
- Ran: `python scripts/talk_figures.py` ($0; existing results; seed 0).
- Wrote: `scripts/fig_f12_groups.png`, `fig_f3_never_sampled.png`, `fig_f5_positions.png`, `fig_hypotheses_schematic.png` (the last is a conceptual sketch, not data).
- Harder-task pilot mean in `fig_f3_never_sampled.png` is read from `archive/data/results/phase3_M60_hard_summary.json`, with the values recorded in `notes/phase3.md` as fallback.
