# Why do filler tokens help? A black-box study on DeepSeek V4 Flash

Putting meaningless text (for example 100 dots) between a question and the answer makes some
language models more accurate, even though they write no reasoning. This project asks what the
model is doing with those extra positions.

The starting hypothesis: model uses the filler to run
**several attempts in parallel and combine them**. 

## Short answer

- The effect is real: 100 dots raise accuracy from 53.5% to 63.5% on 600 items.
- It is **not** a vote among answers the model already produces, and it is **not** the same as
  lowering temperature.
- Independent parallel attempts, as originally proposed, have little support left.
- The best current explanation is about **composition**. The task needs two steps. The model can do
  each step perfectly on its own but often cannot do both in one pass. Filler seems to give the
  first step room to finish before the second step uses it.
- That explanation is an interpretation of behaviour. Confirming it needs the model's activations.

Everything below is for one task family, one model (with one replication), and mostly one filler.

## The task

Each item defines a few nonsense variables and asks a two-step question:

```
zif = 48
yaj = three times the number for zif minus 30      <- first step gives y = 114
...three distractor variables...
Question: What is twice the number for yaj plus 36?  <- second step gives 264

Filler: . . . . . . . .
Answer:
```

The model must answer with just the number. Prompts have 10 worked examples, and reasoning tokens
are switched off and checked to be zero. "Pass rate" means the share of repeated samples of one
item that are correct.

## What we found, and where to look

Each row gives the finding in plain words, the script that produced it, and the figure that shows
it. All scripts and figures are in `scripts/`.

| # | Finding | Script | Figure |
|---|---|---|---|
| 1 | **Filler helps.** 53.5% -> 63.5% with 100 dots (600 items). The gain appears by about 25 dots. | `phase2.py` | `phase2_uplift_pilot.png`, `phase2_uplift_full.png`, `claim1_uplift.png` |
| 2 | **It is not voting and not temperature.** Items whose most common no-filler answer is wrong gain 23 points. A vote predicts no gain there, and temperature 0.4 gives about 2. | `phase3.py`, `phase3_analysis.py` | `fig_f12_groups.png`, `fig_core_delta_by_group.png` |
| 3 | **Filler produces answers the model otherwise never gives.** Of 42 items with no correct answer in 48 no-filler samples, 14 get at least one with filler and 4 reach a majority. | `talk_figures.py` | `fig_f3_never_sampled.png` |
| 4 | **Items switch; the gain is not spread evenly.** 48 of 300 items jump by 50 points or more, and about half barely move. | `summary_figures.py`, `followup_by_cell.py` | `fig_change_histogram.png`, `followup_by_cell.png` |
| 5 | **Items tip at their own filler length**, anywhere from 10 to 100 dots, mostly in one jump. | `followup_dose.py` | `followup_dose.png` |
| 6 | **Filler positions are interchangeable.** Replacing a third of a 25-dot filler with random letters costs about what removing those dots costs, wherever the damage is. | `followup_disrupt25.py` | `fig_f5_positions.png`, `fig_disruption.png` |
| 7 | **Most filler content works.** Prose, shuffled prose, a repeated word and digits match dots. Counting in words, a three-word cycle and random vocabulary do not. | `phase4.py` | `phase4_ladder_full.png` |
| 8 | **The "independent attempts" arithmetic does not fit.** If filler gave N independent tries, easy items should improve most in relative terms. Where filler works, the hardest items improve most. | `followup_independence.py`, `followup_by_cell.py` | `followup_independence.png`, `followup_by_cell.png` |
| 9 | **Wrong answers are structured near misses.** About 96% keep the correct answer's parity (odd or even). With filler they get rarer, not smaller. | `followup_slips.py` | `followup_slips.png` |
| 10 | **The first step decides whether filler can help.** If the first step is a doubling, filler removes about four-fifths of errors. If it is a tripling, filler helps only when the second step is a doubling. | `followup_tipping.py`, `followup_by_cell.py` | `followup_tipping.png`, `followup_by_cell.png` |
| 11 | **Each step alone is perfect.** On the hard items both single steps are right in 2,528 of 2,528 samples, yet the two-step pass rate is 20%. Giving the model the intermediate value fixes everything; 100 dots fix a little. | `followup_singlehop.py`, `followup_hopcost.py` | `followup_singlehop.png`, `followup_hopcost.png` |
| 12 | **Filler only helps after the question.** Dots before the question do nothing; dots at the very start hurt. | `followup_placement.py` | `followup_placement.png` |
| 13 | **The effect replicates on V4.1 Flash, but mostly on different items.** Six smaller open models show no uplift. | `followup_crossmodel.py`, `whitebox_screen.py`, `fig_open_models.py` | `followup_crossmodel.png`, `fig_open_model_uplift.png` |

Two supporting analyses use data released with Brauer et al. (DeepSeek V3, a different task):
`followup_brauer_spectrum.py` / `followup_brauer_spectrum.png` shows the same parity-keeping near
misses in that paper's decoded hidden states.

## Where each hypothesis stands

| Hypothesis | In plain words | Status | Decided by finding |
|---|---|---|---|
| Majority vote | Filler picks the answer the model gives most often | Ruled out | 2 |
| Sharpening | Filler acts like turning the temperature down | Ruled out as the main effect | 2 |
| Small uniform nudge | Every item gets a little better | Ruled out | 4 |
| Prompt format | The mere presence of dots helps | Ruled out | 12 |
| One hard arithmetic step | Filler relieves a step the model finds difficult | Ruled out | 11 |
| Independent parallel attempts | Several tries, keep a good one | Little support | 4, 8, 11 |
| Predictable filler frees capacity | Easy-to-predict filler leaves more room to think | Not supported in simple form | 7 |
| **Room to compose two steps** | Filler lets the first step finish before the second uses it | **Leading, unconfirmed** | 10, 11, 12 |

`fig_hypotheses_schematic.png` is a conceptual sketch (not data) of the two pictures still worth
separating: several complete rival answers held in the filler, versus parts of one computation.

## What is still unexplained

- Why one item switches and a similar one does not, and why at that filler length.
- Why 100 dots recover so little of what writing the intermediate value recovers.
- Why counting in words and a short repeating cycle fail as filler.
- Why different models are helped on different items.

**What would settle it:** reading the intermediate value out of the model's hidden states at the
filler positions and at the answer position. The leading explanation predicts an exact value in the
filler on items that switch, and an approximate or missing one on items that stay wrong. This was
not done here for cost reasons; see `notes/followup.md` for an estimate.

## Corrections made along the way

Three early claims were later withdrawn or narrowed. The notes keep the original text and the
correction.

- "A bump at ±10 in Brauer's decodes" was a parity effect.
- "Errors are a correct final step applied to a wrong intermediate" could not be inferred the way
  it first was.
- "All error types fall by the same proportion" was a blend of two very different groups of items
  (finding 10).

## Repository layout

```
README.md          this file
scripts/           one script per experiment, each figure saved next to its script
  api.py           cached, budget-capped API client
  prompts.py       the prompt format shared by every experiment
  phase0..4.py     the planned phases (setup, task, uplift gate, main test, filler types)
  followup_*.py    experiments added after the main test
  *_figures.py     figure-only scripts that read existing results
data/tasks/        the generated items (eval.jsonl is the frozen 600-item set)
data/results/      raw responses (jsonl) and summary tables
notes/             results notes written at each checkpoint
  phase0..4.md     one note per planned phase
  followup.md      every follow-up, in the order it was run, including design and
                   pass/fail criteria written before the data where applicable
  log.md           dated log: what ran, the exact command, cost, result
  costs.md         running spend
vendor/            clone of Brauer et al.'s public repository (task generator, released decodes)
```

## Reproducing

- **Setup:** Python 3.9+ with `requests`, `pandas`, `numpy`, `scipy`, `matplotlib`, `tokenizers`.
  Put `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY` and `BUDGET_USD` in `.env`.
- **Model route:** DeepSeek V4 Flash through OpenRouter, pinned to one host so the weights'
  precision never changes between calls. V4.1 Flash through Fireworks for the replication.
- **Analysis-only scripts cost nothing** and run from the committed results:
  `phase3_analysis.py`, `followup_existing_data.py`, `followup_slips.py`,
  `followup_independence.py`, `followup_tipping.py`, `followup_by_cell.py`,
  `followup_brauer_spectrum.py`, `summary_figures.py`, `talk_figures.py`, `claims_figures.py`,
  `fig_open_models.py`.
- **Collection scripts call the API.** The response cache is not committed, so rerunning them
  spends money again; `api.py` refuses to exceed `BUDGET_USD`. Total spend for everything here was
  about $8.60.
- Exact commands for every run are in `notes/log.md`.
- Greedy decoding on these API routes is not fully deterministic, so reruns will differ slightly.
