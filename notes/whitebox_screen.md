# Uplift screen on smaller lens-equipped models (2026-10-04)

Human-approved ("go"). Purpose: find a standard-attention model with a published J-lens that shows
filler uplift, so the attention-masking experiment could run cheaply.

**Result: FAIL. None of the six models shows filler uplift. The cheap white-box route is closed.**

Command: `python scripts/whitebox_screen.py [--suffix _veasy | _easy] [--models ...]`.
150 items, 10-shot, greedy, thinking off, k=0 vs dots k=25 vs dots k=100; each model pinned to the
host that answered a probe call. All routes: 0 calls with reasoning tokens, 100% parse rate.
Cost $1.61 (cumulative $7.60 of $30). Raw: `data/results/whitebox_screen*.jsonl`.

Accuracy without filler -> with 100 dots (150 items):

| Model | Main task | Easier (coef 2, const 1–30) | Easiest tried (coef 2, const 1–9) |
|---|---|---|---|
| Qwen3.5-9B | 0% -> 0% | not run | 3% -> 4% |
| Qwen3.5-27B | 4% -> 5% | 7% -> 7% | 21% -> 15% (p = 0.05, worse) |
| Qwen3.6-27B | 5% -> 7% | 5% -> 7% | 19% -> 17% |
| Gemma-3-27B | 1% -> 1% | 1% -> 2% | 5% -> 7% |
| Qwen3.6-35B-A3B | 2% -> 1% | not run | 7% -> 8% |
| Qwen3.5-122B-A10B | 3% -> 3% | 5% -> 4% | not run |
| V4 Flash, for reference (100-item pilots) | 50% | 78% | 35% |

- On the main and easier tasks every model is at the floor, so there is no room to see uplift.
- Where two models have room (about 20% on the easiest set), filler does not help; no condition
  shows a positive effect anywhere near significance (dots k=25 results are equally flat).
- The low scores are real: responses are integers close to the truth (median error 12–20), not a
  format problem.
- One retry at lower difficulty was allowed and used. Qwen3.5-4B is not hosted and was not tested.

Surprise (not investigated): V4 Flash scores *lower* with constants 1–9 (35%) than with constants
1–30 (78%), although the arithmetic is nominally easier.

## What remains for white-box
Only the original route: V4 Flash itself (160 GB, multi-GPU, custom compressed attention), i.e.
the Phase 5 spike. Human decides.
