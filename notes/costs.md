# Costs

<!-- AUTO:BEGIN (rewritten by scripts/api.py; do not edit) -->
## Running total: $8.5736 of $30.00 budget

| tag | paid calls | prompt tok | cached tok | completion tok | reasoning tok | USD |
|---|---|---|---|---|---|---|
| phase0 | 353 | 393140 | 230826 | 717 | 0 | 0.0379 |
| phase1 | 900 | 1098192 | 778496 | 1812 | 0 | 0.0998 |
| phase2_pilot | 800 | 1126558 | 829696 | 1599 | 0 | 0.1001 |
| phase2_full | 1350 | 1990659 | 1721856 | 2708 | 0 | 0.1589 |
| phase3_M60 | 3840 | 4596480 | 3928832 | 7681 | 0 | 0.3706 |
| phase3_M300 | 15360 | 18372032 | 15723264 | 30721 | 0 | 1.4801 |
| phase3_M60_hard | 3840 | 4464256 | 3910656 | 7932 | 0 | 0.3535 |
| followup_dose | 10512 | 12970176 | 11086336 | 21051 | 0 | 1.0457 |
| followup_binding | 1100 | 2835935 | 2298624 | 2200 | 0 | 0.2367 |
| followup_disrupt | 4608 | 8495808 | 7315200 | 9214 | 0 | 0.6799 |
| followup_disrupt25 | 4608 | 5471424 | 4275712 | 9195 | 0 | 0.4693 |
| followup_crossmodel | 1200 | 1766432 | 1420568 | 2462 | 0 | 0.0877 |
| phase4_pilot | 900 | 1828980 | 1476608 | 1800 | 0 | 0.1532 |
| phase4_full | 4500 | 9143964 | 7888896 | 9018 | 0 | 0.7305 |
| whitebox_screen | 2706 | 4133805 | 570688 | 9698 | 0 | 0.7169 |
| whitebox_screen_veasy | 2255 | 3252127 | 1311536 | 8569 | 0 | 0.4009 |
| whitebox_screen_easy | 1804 | 2651066 | 548032 | 6373 | 0 | 0.4780 |
| followup_hopcost | 4000 | 5240820 | 4467712 | 8864 | 0 | 0.4235 |
| followup_singlehop | 5056 | 4327824 | 3844608 | 10112 | 0 | 0.3396 |
| followup_placement | 1200 | 2464832 | 1924864 | 2412 | 0 | 0.2110 |
<!-- AUTO:END -->

## Pricing used (OpenRouter -> Parasail fp8, DeepSeek V4 Flash 0423)
$0.14 / 1M input, $0.07 / 1M cached input, $0.28 / 1M output (OpenRouter endpoint listing,
2026-10-04). Spend is recorded from the billed `usage.cost` field, not from this table.

## Estimates at planned sizes (2026-10-04, V4 Flash)
Upper bound assumes no cache hits ($0.14 / 1M). Expected uses the measured ~$0.10 / 1M effective
rate from Phase 0. Output tokens are negligible (2 per call).

| Phase | Calls | Prompt tokens | Upper bound | Expected |
|---|---|---|---|---|
| 1: calibration, 2 rounds x 100, k=0 | 200 | 0.2M | $0.03 | $0.02 |
| 2: pilot 150 x 6 conditions | 900 | 1.2M | $0.17 | $0.12 |
| 2: full 600 x 3 conditions (worst case dots 50 + 100) | 1,800 | 2.7M | $0.38 | $0.27 |
| 2: retry pilot + 5-shot diagnostic (only if FAIL) | ~1,500 | 1.7M | $0.24 | $0.17 |
| 3: 300 items x 64 samples (F* = dots 100 worst case) | 19,200 | 23.0M | $3.22 | $2.30 |
| 4 (optional): ~12 conditions x 600, ~1,500 tok each | 7,200 | 10.8M | $1.51 | $1.08 |
| **Total** | | | **~$5.55** | **~$3.96** |

Budget cap is $30 (OpenRouter key limit is $50). Phase 0 itself cost $0.04 across both providers.
