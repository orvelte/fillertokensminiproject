# Filler-token project: instructions for Claude Code

## 0. Purpose, scope, and working rules

**Main question (Neel's hypothesis).** When filler tokens (e.g. `. . . .`) sit between a question
and the answer, the model uses the extra positions to run *multiple attempts in parallel* and
aggregates them at the answer position: an internal best-of-N.

**What this file covers (in order):**
1. Phase 0: setup and API verification
2. Phase 1: generate the system-of-equations task
3. Phase 2: uplift screen (GATE: no uplift, no project)
4. Phase 3: E3 black-box, the primary test of Neel's hypothesis
5. Phase 4: E2 behavioral (OPTIONAL; tests the partner's "freed capacity" hypothesis, not Neel's)
6. Phase 5: go/no-go checklist for white-box work (do NOT execute from this file)

**Working rules. Follow strictly; they exist to limit cost and rabbit-holing.**
- **STOP checkpoints are hard stops.** At each `STOP`, write a short results note to `notes/`
  (what ran, cost, result, recommendation) and wait for a human decision. Do not continue to
  the next phase on your own.
- **Pilot before scaling.** Every phase runs a small pilot first. Never launch a full run until
  its pilot passes.
- **Cache every API call** (see Phase 0). Re-running analysis must never re-spend money.
- **Respect the budget cap.** Read `BUDGET_USD` from `.env`; abort a run if projected or actual
  spend would exceed it.
- **No scope creep.** Do not add models, filler types, analyses, or abstractions not listed here.
  Write any new idea as one line in `notes/ideas.md` and move on.
- **Record surprises, don't chase them.** If something unexpected happens, log it in
  `notes/log.md` and flag it at the next STOP.
- **Keep it simple.** One script per phase in `scripts/`; plain Python + pandas + matplotlib.
  No frameworks.
- **Respect the timeboxes.** If a phase exceeds its timebox, stop and report.

---

## 1. Repo layout

```
paper/
  brauer_2607.03502.pdf          # "Reading Between the Dots" (provided by human)
  baherwani_2607.22925.pdf       # "Not All LLM Reasoning is Visible in the CoT" (provided by human)
  greenblatt_filler_post.md      # Greenblatt blog post, saved as markdown (provided by human)
  project_brief.md               # Neel's project description + our hypotheses (provided by human)
vendor/
  filler-token-reasoning/        # git clone https://github.com/kaleybrauer/filler-token-reasoning
data/
  tasks/                         # generated items (jsonl)
  results/                       # raw responses (jsonl) + tables
cache/                           # API response cache (sqlite or jsonl keyed by request hash)
scripts/
notes/
  log.md  costs.md  ideas.md  phase0.md ... phase4.md
.env                             # DEEPSEEK_API_KEY, BUDGET_USD (never commit)
```

---

## 2. Phase 0: setup and API verification (timebox: half a day)

1. Clone `vendor/filler-token-reasoning`. Read its task-generation code, prompt templates, and
   README. Read Appendix A of `paper/brauer_2607.03502.pdf` (task spec + example prompts).
   Write a 10-line summary of what is reusable to `notes/phase0.md`.
2. Write `scripts/api.py`:
   - Uses the official DeepSeek API. Look up the current model id for **DeepSeek V4 Flash** and
     how to select **non-thinking mode** in the current API docs. Do not guess.
   - Cache key = sha256 of the full request (model, messages, temperature, max_tokens, seed, n).
     Cache hits cost nothing.
   - Retries with backoff, a concurrency limit, and per-call cost tracking from the `usage`
     fields. Running total in `notes/costs.md`; abort at `BUDGET_USD`.
3. Verification checks. Each must PASS; record results in `notes/phase0.md`.
   - **V1 No hidden reasoning (critical).** On 20 calls, confirm zero reasoning tokens in
     `usage` and no `reasoning_content`. **If thinking cannot be fully disabled: STOP.** The
     no-CoT premise of the whole project fails on this route.
   - **V2 Answer format.** With `max_tokens` around 10, at least 95% of pilot responses must parse
     as a single signed integer.
   - **V3 Sampling.** Confirm temperature is honored (T=1 gives varied answers on hard items).
     Check whether `n > 1` is supported (saves prompt cost) and whether prefix/context caching
     reports cache hits on the shared few-shot prefix.
   - **V4 Prompt-token logprobs.** Check whether the official API, or any provider serving the
     *same* V4 Flash checkpoint, returns logprobs for **prompt** tokens (echo / prompt_logprobs).
     Only Phase 4 needs this. **Spend at most 1 hour.** Record yes/no and the route.
4. **Cost model.** Measure tokens per call for the final prompt format. Using the current pricing
   page, estimate the cost of Phases 1–4 at the planned sizes. Write the estimates to
   `notes/costs.md`.

**STOP 0.** Report V1–V4 and the cost estimates.

---

## 3. Phase 1: generate the system-of-equations task (timebox: half a day)

Use Brauer's generator if the repo has one; otherwise implement from the Appendix A spec:
- Nonsense 3-letter variable names. Some are assigned literal two-digit values; others are defined
  as a linear function of one earlier variable (multiply by 2 or 3, then add or subtract a
  constant from 0–50).
- **Distractor variables** that do not feed the queried variable.
- The question asks for a linear function of one variable. Chain:
  `x -> c1*x -> y -> c2*y -> answer`.
- The "Easier variation" in Appendix A (coefficients fixed to 2, constants 1–30) is the
  fallback difficulty knob.

For every item, store: prompt text pieces, queried variable, the full chain
`{x, c1x, y, c2y, answer}`, every distractor variable with its value, and **binding-error rival
answers**: the answer obtained by applying the question's operation to each *other* variable.
These rivals are not used until white-box work, but they are free to record now.

**Prompt format (identical across conditions except the filler tokens):**
- System prompt adapted from Appendix A: answer immediately with just the number, no reasoning.
  Mention neutrally that filler text may follow the question. Use **one system prompt for all
  conditions**; do not name the filler type.
- **10 fixed few-shot examples** (excluded from eval). Each shows the **same filler condition**
  as the test item. Filler effects interact strongly with few-shot context, so never mix.
- Test item: `Question: ...` then `Filler: <tokens>` then `Answer:`.
- **The baseline (k=0) keeps the `Filler:` label line with zero filler tokens.** This is
  deliberate: the label is itself a position that can carry computation, so it must be present in
  both conditions.

**Calibrating difficulty (at most 2 rounds):**
- Pilot 100 items, greedy (T=0), k=0. Target baseline accuracy **25–60%** (items need room to
  move, and Phase 3 needs many items the model sometimes gets right).
- Adjust only: coefficient set, constant range, number of distractors. Then **freeze** the
  generator (fixed seed) and generate the eval set: **N = 600 items**, plus the fixed few-shot set.

**STOP 1.** Report baseline accuracy, the final generator settings, and 3 example prompts.

---

## 4. Phase 2: uplift screen = GATE (timebox: 1 day)

**Pilot (150 items, greedy, paired):**
- Conditions: dots with k ∈ {0, 10, 25, 50, 100}, plus counting (`1 2 3 ...`) k = 25.
- k counts filler units, not tokens. Record the actual token counts from the tokenizer.
- Per condition: accuracy, McNemar test vs k=0, and flip counts in both directions
  (wrong→right and right→wrong).

**Scaling rule:** if the best pilot condition shows a gain of at least 5 pp with McNemar
p < 0.05, run k=0 plus the best 2 conditions on all N=600.
**PASS** requires at least +5 pp with p < 0.01 at N=600.

**If FAIL:**
1. One retry only: shift difficulty so the k=0 baseline is around 40%, using the fallback knob,
   and re-run the pilot.
2. Diagnostic (150 items): run the same comparison with a 5-shot prompt. If uplift appears only
   at 5-shot, record "uplift is prompt-substitutable." Neel's guidance is that strong prompting
   should absorb spurious uplift, so this still counts as **FAIL**.
3. If still failing: **STOP with a PIVOT recommendation** (see Section 8).

**STOP 2.** Uplift curve figure + table, and the chosen filler condition **F\*** for Phase 3.

---

## 5. Phase 3: E3 black-box, testing Neel's hypothesis (timebox: 1.5 days)

### Hypotheses and their per-question predictions
Let `p0` = an item's no-filler pass rate and `pf` = its pass rate with filler F\*.
Call an item **correct-modal** if the correct answer is the most common no-filler answer.

| Hypothesis | Prediction |
|---|---|
| **H_vote**: parallel attempts + majority vote (Neel's hypothesis, simplest form) | Correct-modal items improve; **wrong-modal items stay the same or get worse**; `pf` tracks maj@k for some small k |
| **H_verify**: attempts + smarter selection | Items where the correct answer is present but **not modal** also improve |
| **H_nudge**: no attempts; uniform small boost | Improvement similar for correct-modal and wrong-modal items |
| **H_sharpen** (confound): filler just lowers output entropy, like lowering temperature | **Same per-item pattern as H_vote.** Must be checked with the temperature control below |

**Black-box limit, stated up front:** a voting-like pattern cannot by itself tell parallel voting
apart from simple confidence sharpening. Both boost the modal answer. Black-box *can*:
- **rule out H_nudge;**
- **positively support "more than sharpening"** if wrong-modal items improve (H_verify), since
  sharpening and pure voting can only suppress non-modal answers.

If the result is voting-like and matches the temperature control, the conclusion is "consistent
with Neel's hypothesis but not discriminating"; resolving it needs white-box work (Phase 5).

### Design
- Items: M = 300 from the eval set. **Pilot M = 60 first.**
- Per item:
  - k=0: K = 16 samples at T=1.0
  - F\*: K = 16 samples at T=1.0
  - **Temperature control:** k=0, K = 16 samples at T=0.7 and K = 16 at T=0.4
- `max_tokens` small. Use `n > 1` if supported; otherwise rely on prefix caching.
- **Compute the cost from Phase 0's cost model before running.** This is the most expensive phase.

### Analysis (`scripts/phase3_analysis.py`)
1. **Avoid regression to the mean.** Split each item's k=0 samples into two halves of 8. Use
   half A to classify the item (correct-modal or not; correct answer present or not). Use half B
   to estimate `p0`. Never use the same samples for both.
2. **Main plot:** Δ = `pf − p0` against `p0`, colored by group (correct-modal / correct present
   but non-modal / correct absent). Bootstrap 95% CIs over items (10k resamples).
3. **maj@k fit:** for each item, compute maj@k accuracy from the k=0 samples (bootstrap within the
   item) for k ∈ {1, 3, 5, 9, 15}. Report which k best matches `pf` on average and per item.
4. **Temperature control:** do the same comparison for T=0.7 and T=0.4. If some temperature
   reproduces filler's per-item Δ pattern about as well as maj@k does, record that black-box
   cannot separate H_vote from H_sharpen here.
5. Headline numbers: the fraction of wrong-modal items that get worse with filler, and the
   fraction of correct-present-but-non-modal items that improve.

### Testability checks (report explicitly, PASS or FAIL)
- **T1 Power:** at least about 60 items in each of correct-modal and wrong-modal with `p0` in
  [0.1, 0.9]. If FAIL: the black-box test is underpowered on this task. Report what would fix it
  (e.g. harder items to create more wrong-modal cases).
- **T2 Uplift survives sampling:** mean `pf > p0` at T=1. If greedy uplift existed but vanishes
  at T=1, the analysis is invalid. Allowed fix: re-run once at T=0.7 for both conditions.
- **T3 Discriminability:** CIs on the group means of Δ are narrow enough to separate the
  predictions in the table (rough guide: CI half-width under 5 pp). If not, report "not
  discriminable at this M" plus the M that would be needed. Do not increase M yourself.

**STOP 3.** Report which hypothesis pattern holds, the T1–T3 results, and an explicit verdict:
- (a) evidence for Neel's hypothesis beyond sharpening;
- (b) consistent but not discriminating, so white-box is needed;
- (c) evidence against;
- (d) not testable with this setup, with the reason.

---

## 6. Phase 4 (OPTIONAL): E2 behavioral, freed-capacity hypothesis (timebox: 1 day)

Run only with explicit human OK after STOP 3. **This tests the partner's hypothesis, not Neel's.**

- **Filler ladder** (all matched on token count using the V4 Flash tokenizer, which can be
  downloaded from the HF repo without weights): dots, repeated word (`the the ...`), counting in
  words, counting in digits, short cycle (`red blue green ...`), memorized public-domain text
  (Gettysburg Address), a bland self-written paragraph with no numbers, the same paragraph
  shuffled, random vocabulary tokens.
- **Matched pairs** (the core comparisons): prose vs shuffled prose; paragraph P repeated twice
  (P+P) vs two different paragraphs (P+Q); counting in words vs counting in digits.
- Greedy, N=600, paired, with filler in the few-shot examples as always.
- **Track format violations separately.** Report accuracy both counting violations as wrong and
  among format-following responses only.
- **Surprisal:** compute the mean per-token NLL of each filler *inside the real prompt*, only if
  Phase 0 V4 found a prompt-logprob route. Otherwise report the behavioral ladder alone and mark
  surprisal as deferred to white-box.

**STOP 4.**

---

## 7. Phase 5: white-box go/no-go checklist (planning only; do not execute)

White-box work starts only if STOP 3 returns verdict (a) or (b). Before committing, a **one-day
infrastructure spike** must hit all of:
1. Load V4 Flash weights and reproduce API accuracy on 50 items within ±5 pp.
2. Extract hidden states at every filler position and layer.
3. Load the J-lens (`camilablank/workspace-lenses`, `deepseek-v4-flash`). Confirm its layer
   convention (block input vs output) and **which mHC residual stream** it was fit on. Confirm
   the lens checkpoint matches the weights.
4. Run one KV-cache transplant end to end, handling the compressed attention (CSA/HCA). Confirm
   how filler positions map onto compressed blocks.

If any item fails within one day, pivot white-box work to a standard-attention model that has a
published J-lens **and** shows uplift in a Phase 2-style screen.

---

## 8. Pivot table

| Condition | Action |
|---|---|
| V1 fails (thinking can't be disabled) | Try another provider serving the same checkpoint; else pivot to Gemini 3 Flash (black-box, Neel's recommendation), verifying zero thinking tokens |
| Phase 2 fails after one retry | Pivot E3 black-box to Gemini 3 Flash on the same task; keep the generator and pipeline |
| Uplift only at 5-shot | Same as Phase 2 fail |
| T1 fails (too few wrong-modal items) | Raise difficulty once, re-run the Phase 3 pilot only |
| T3 fails | Report the required M; human decides whether to pay for it |
| Phase 3 verdict (b) | Plan white-box (Phase 5) |
| Phase 3 verdict (c) or (d) | STOP; human decides the pivot |

---

## 9. Conventions
- Fixed seeds everywhere; record them in `notes/log.md`.
- All raw responses saved as jsonl in `data/results/` (request, response, parsed answer, usage).
- Paired tests: McNemar for single-sample paired accuracy; bootstrap over items for per-item
  rates.
- Every figure saved next to the script that produced it, with the exact command in
  `notes/log.md`.
- `notes/log.md` entries are dated: what ran, cost, result, decision.