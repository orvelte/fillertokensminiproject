"""One summary figure per supported black-box claim (reads existing results only; no API calls).

Run: python scripts/claims_figures.py
Writes scripts/claim1_uplift.png, claim2_not_vote_not_sharpen.png,
claim3_shared_across_fillers.png, claim4_error_types.png.
(Claim 4's dose-response panel already exists as scripts/followup_dose.png.)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, HERE = ROOT / "data" / "results", Path(__file__).parent
rng = np.random.default_rng(0)

# ---------------- claim 1: uplift is real ----------------
pilot, full = pd.read_csv(RES / "phase2_pilot_table.csv"), pd.read_csv(RES / "phase2_full_table.csv")
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
dots = pilot[pilot.cond.str.startswith("dots")]
a.errorbar(range(len(dots)), 100 * dots.acc, yerr=100 * dots.se, marker="o", capsize=3, label="dots")
cnt = pilot[pilot.cond == "counting_25"].iloc[0]
a.errorbar([2.1], [100 * cnt.acc], yerr=[100 * cnt.se], marker="s", capsize=3, color="C1", label="counting (k=25)")
a.set_xticks(range(len(dots)))
a.set_xticklabels(dots.cond.str.split("_").str[1])
a.set_xlabel("filler units k")
a.set_ylabel("accuracy (%)")
a.set_title("Pilot, 150 items (±1 SE)", fontsize=10)
a.legend(fontsize=8)
names = {"dots_0": "no filler", "counting_25": "counting, k=25", "dots_100": "dots, k=100"}
bars = b.bar([names[c] for c in full.cond], 100 * full.acc, yerr=100 * full.se, capsize=4, color=["C7", "C1", "C0"])
for rect, (_, r) in zip(bars, full.iterrows()):
    label = f"{100 * r.acc:.1f}%" if r.cond == "dots_0" else f"{100 * r.acc:.1f}%\n+{r.gain_pp:.1f} pp, p={r.mcnemar_p:.0e}"
    b.text(rect.get_x() + rect.get_width() / 2, 100 * r.acc + 3, label, ha="center", fontsize=9)
b.set_ylim(0, 80)
b.set_ylabel("accuracy (%)")
b.set_title("Full run, 600 items, paired (McNemar p vs no filler)", fontsize=10)
fig.suptitle("Claim 1: filler raises V4 Flash accuracy (10-shot, greedy, no reasoning tokens)", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "claim1_uplift.png", dpi=150)

# ---------------- claim 2: not majority voting, not sharpening ----------------
it = pd.read_csv(RES / "phase3_M300_items.csv")
groups = [("correct answer is modal", it.group == "correct-modal"),
          ("correct present,\nnot modal", it.group == "present, non-modal"),
          ("correct never sampled", it.group == "absent")]
series = [("with filler (dots, k=100)", "pf", "C0"), ("predicted by 15-way majority vote", "maj@15", "C2"),
          ("temperature 0.7, no filler", "pT0.7", "C4"), ("temperature 0.4, no filler", "pT0.4", "C6")]
fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.2
for j, (label, col, color) in enumerate(series):
    means, los, his = [], [], []
    for _, m in groups:
        d = (it[m][col] - it[m].p0).values * 100
        bs = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(1)
        means.append(d.mean()); los.append(d.mean() - np.percentile(bs, 2.5)); his.append(np.percentile(bs, 97.5) - d.mean())
    ax.bar(np.arange(len(groups)) + (j - 1.5) * w, means, w, yerr=[los, his], capsize=3, color=color, label=label)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g}\n(n={int(m.sum())})" for g, m in groups])
ax.set_ylabel("change in pass rate vs no filler at T=1 (pp)")
ax.set_title("Claim 2: filler lifts items that voting and lower temperature leave unchanged\n"
             "(300 items, 16 samples each; groups from a held-out half of the no-filler samples; 95% CI)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "claim2_not_vote_not_sharpen.png", dpi=150)

# ---------------- claim 3: item-specific, shared across two fillers ----------------
fu = json.load(open(RES / "followup_existing_data.json"))
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
keys = [("cross_filler_wrong at k=0", "fixed by both fillers\n(of 279 items wrong\nwithout filler)"),
        ("cross_filler_right at k=0", "broken by both fillers\n(of 321 items right\nwithout filler)")]
x = np.arange(2)
obs, exp = [fu[k]["both"] for k, _ in keys], [fu[k]["expected_independent"] for k, _ in keys]
a.bar(x - 0.2, obs, 0.4, color="C0", label="observed")
a.bar(x + 0.2, exp, 0.4, color="C7", label="expected if the two fillers\nacted independently")
for xi, o, e, (k, _) in zip(x, obs, exp, keys):
    a.text(xi - 0.2, o + 1, str(o), ha="center", fontsize=9)
    a.text(xi + 0.2, e + 1, f"{e:.1f}", ha="center", fontsize=9)
    a.text(xi, max(o, e) + 7, f"Fisher p={fu[k]['fisher_p']:.0e}", ha="center", fontsize=8)
a.set_xticks(x)
a.set_xticklabels([lab for _, lab in keys], fontsize=8)
a.set_ylabel("number of items")
a.set_ylim(0, 80)
a.set_title("Dots k=100 and counting k=25 change the same items\n(greedy, 600 items)", fontsize=10)
a.legend(fontsize=8)
lp = fu["low_p0_pf_by_counting_greedy"]
vals = [lp["False"], lp["True"]]
b.bar(["counting k=25 wrong", "counting k=25 right"], [100 * v["mean_pf"] for v in vals], color=["C3", "C2"])
for i, v in enumerate(vals):
    b.text(i, 100 * v["mean_pf"] + 2, f"{100 * v['mean_pf']:.0f}%  (n={v['n']})", ha="center", fontsize=9)
b.set_ylim(0, 100)
b.set_ylabel("mean pass rate with dots k=100 (%)")
b.set_title("Items rarely solved without filler (pass rate ≤ 25%):\none filler's outcome predicts the other's", fontsize=10)
fig.suptitle("Claim 3: the effect is item-specific and shared across two filler types", fontsize=11)
fig.tight_layout()
fig.savefig(HERE / "claim3_shared_across_fillers.png", dpi=150)

# ---------------- claim 4 (part a): errors are in the intermediate value, and filler removes them ----------------
et = fu["error_types_pct_of_samples"]
short = {"other, consistent with wrong y + correct final step": "wrong intermediate value y,\nfinal step done correctly",
         "final-step structural (sign / coefficient / dropped step)": "final step: wrong sign /\ncoefficient / dropped step",
         "other, final step itself wrong": "final step arithmetic wrong",
         "hop-1 structural (sign / coefficient / dropped constant / wrong source)": "first hop: wrong sign /\ncoefficient / source",
         "binding: question applied to another variable": "question applied to\nthe wrong variable",
         "unparsed": "unparsed response"}
order = sorted(short, key=lambda k: -et["k0_T1.0"][k])
fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(order))[::-1]
for off, col, color, label in [(0.2, "k0_T1.0", "C7", "no filler"), (-0.2, "F_T1.0", "C0", "dots, k=100")]:
    v = [et[col][k] for k in order]
    ax.barh(y + off, v, 0.4, color=color, label=label)
    for yi, vi in zip(y + off, v):
        ax.text(vi + 0.5, yi, f"{vi:.1f}%", va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels([short[k] for k in order], fontsize=8)
ax.set_xlabel("% of all sampled answers (300 items × 16 samples, T=1)")
ax.set_xlim(0, 55)
ax.set_title("Claim 4a: most errors are a wrong intermediate value, and that is what filler reduces\n"
             f"(all wrong answers: {et['k0_T1.0']['TOTAL wrong']:.1f}% without filler, {et['F_T1.0']['TOTAL wrong']:.1f}% with)", fontsize=10)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
fig.savefig(HERE / "claim4_error_types.png", dpi=150)
print("wrote 4 figures")
