# Phase 0 — setup and API verification (2026-10-04)

## FINAL ROUTE: V4 Flash via OpenRouter (re-run after STOP 0 decision)
`deepseek/deepseek-v4-flash` (0423 = HF `deepseek-ai/DeepSeek-V4-Flash`, the J-lens checkpoint),
pinned to Parasail (fp8), fallbacks off, `reasoning: {"enabled": false}`. 10-shot, max_tokens=10.

| Check | Result |
|---|---|
| **V1 no hidden reasoning** | **PASS.** 0/20 calls with reasoning tokens or reasoning text (also 0 in the 120 V2 calls). |
| **V2 answer format** | **PASS.** 120/120 greedy responses parse as one signed integer. (At T=1, 1 of 40 samples failed to parse.) |
| **V3 temperature** | **PASS.** 5/5 hard items give varied answers over 8 samples at T=1. |
| V3 `n > 1` | **Not supported**: `n=8` is silently ignored and returns 1 choice. One call per sample. |
| V3 prefix caching | Partial: 39/40 calls report cached tokens; roughly a third to a half of the prompt is billed as cached. |
| **V4 prompt-token logprobs** | **No route found.** `echo` is ignored (only output-token logprobs come back). Fireworks refuses it for V4.1; official API no longer serves V4 Flash. Surprisal deferred to white-box. |

- T=0 is not deterministic here either: 3/5 hard items identical over 4 repeats.
- Output-token `top_logprobs` work.
- Stand-in accuracy, n=40, Brauer default-difficulty items (not a test): k=0 52%, dots k=25 65%,
  counting k=25 65%.
- Tokens per call: k=0 923; dots 25 1198; counting 25 1484 (same tokenizer counts as V4.1).
- Measured cost per call: k=0 $0.00010, dots 25 $0.00011, counting 25 $0.00015.
- Raw responses: `data/results/phase0.jsonl` (V4.1/Fireworks run kept as
  `phase0_fireworks_v41flash.jsonl`).

---
Everything below is the earlier run on **V4.1 Flash via Fireworks**, kept for the record.

**Route: Fireworks serverless, `accounts/fireworks/models/deepseek-v4p1-flash`, `thinking: {"type": "disabled"}`.**
Human asked for Fireworks instead of the official API. Total Phase 0 spend: $0.02.

## FLAG: this is V4.1 Flash, not V4 Flash
- Fireworks' only DeepSeek serverless model is **V4.1 Flash**. The official API retired V4 Flash on
  2026-09-10; `deepseek-flash` / `deepseek-v4-flash` now route to V4.1 Flash.
- Per secondary sources (not verified against the tech report), V4.1 Flash is a different
  architecture and size from V4 Flash (552B vs 284B), not a re-tune.
- Neel's prompt and Phase 5 assume **V4 Flash** (uplift reported, J-lens `deepseek-v4-flash`).
  Nothing is known about filler uplift on V4.1, and the V4 Flash J-lens would not apply to it.
- OpenRouter's public model list still shows `deepseek/deepseek-v4-flash` (0423, HF
  `deepseek-ai/DeepSeek-V4-Flash`) and `deepseek/deepseek-v4-flash-0731`. Untested (no key).
- **All results below are for V4.1 Flash.** Human decision needed (see STOP 0).

## What is reusable from Brauer's repo (`vendor/filler-token-reasoning`)
1. `scripts/data/generate_varbind_dataset.py` is the system-of-equations generator: CVC nonsense
   terms filtered against a word list, 5 terms, chain_len 1, literals 10–99, constants 1–50,
   coefficients {2,3} ("twice"/"three times"), topological shuffle, non-negative values.
2. Its knobs map onto our allowed calibration knobs: `--max-coef`, `--const-max`, `--num-terms`.
3. The "easy" variant is `--max-coef 2 --const-max 30` (our fallback knob).
4. It stores definitions, queried term, y (`queried_value`), question coef/op/const, answer. It does
   **not** store x, c1·x, c2·y, distractor values or rival answers; we must add those.
5. Shipped datasets (500 items + 8 few-shot, seed 42) exist for default and easy difficulty; used
   here as stand-in items only.
6. Prompt scaffold `defs / Question: / (blank) / Filler: ... / (blank) / Answer:` as chat turns
   with bare-number assistant replies; reused in `scripts/prompts.py`.
7. Differences we impose: 10 few-shot (theirs 5), one system prompt for all conditions (theirs
   names the filler type), and k=0 keeps the `Filler:` line (theirs drops it).
8. Their filler: dots = `. . .` (1 token/unit), counting = `1 2 3` (2 tokens/unit).
9. Their answer parser and eval scripts target vLLM/local models; not reused.
10. Needs a word list (`/usr/share/dict/words` or words_alpha.txt) or it refuses to generate.

## Verification checks (V4.1 Flash on Fireworks, 10-shot prompt, max_tokens=10)
| Check | Result |
|---|---|
| **V1 no hidden reasoning** | **PASS.** 0/20 calls with reasoning tokens or `reasoning_content` (also 0 in the 120 V2 calls). Default mode does think, so the flag is required. |
| **V2 answer format** | **PASS.** 120/120 responses parse as a single signed integer (k=0, dots 25, counting 25). 2 completion tokens per call. |
| **V3 temperature** | **PASS.** 5/5 hard items give varied answers over 8 samples at T=1. |
| V3 `n > 1` | **Not supported** (HTTP 400). One call per sample. |
| V3 prefix caching | Works partially: all calls report cached tokens, on average ~55% of the prompt. |
| **V4 prompt-token logprobs** | **No route found.** Fireworks: "temporarily unsupported for this model". Official API untested (key invalid). ~15 min spent. Phase 4 surprisal would be deferred to white-box. |

Other observations:
- **T=0 is not deterministic.** Only 2/5 hard items gave identical answers over 4 repeats at T=0.
  Phase 2's "greedy, paired" comparison will carry sampling noise on borderline items.
- Output-token logprobs with `top_logprobs` do work on chat completions.
- The `DEEPSEEK_API_KEY` in `.env` fails authentication on the official API.
- Stand-in accuracy, n=40, Brauer's default-difficulty items (not a test): k=0 40%, dots k=25 60%,
  counting k=25 48%.

## Tokens per call (measured)
k=0: 924 prompt tokens. Dots add 11 per unit (1 token × 11 turns), counting about 22 per unit.
dots 25 = 1199, counting 25 = 1485; extrapolated dots 50 = 1474, dots 100 = 2024.

## STOP 0 — decisions needed
1. Model: continue on V4.1 Flash (Fireworks), or switch to V4 Flash via OpenRouter (needs a key)?
2. Accept non-deterministic T=0 for Phase 2, or handle it differently?
