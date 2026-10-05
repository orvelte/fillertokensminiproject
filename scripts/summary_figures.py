"""Three summary figures from existing results (no API calls).

Run: python scripts/summary_figures.py
1. fig_core_delta_by_group.png  - per-item change vs held-out no-filler pass rate, by group, for
                                  filler (left) and for temperature 0.4 (right); shaded where a
                                  majority vote over no-filler samples predicts no gain.
2. fig_change_histogram.png     - histogram of per-item change with filler, overlaid with what a
                                  uniform boost of the same average size would produce.
3. fig_disruption.png           - 25-dot filler: intact, letters early / middle / late, and the
                                  matched 17-dot control, with 95% intervals.
Data: Phase 3 M=300 (300 items, 16 samples per condition, T=1) and followup_disrupt25.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
RES, HERE = ROOT / "data" / "results", Path(__file__).parent
rng = np.random.default_rng(0)

# ---------------- 1. core figure ----------------
it = pd.read_csv(RES / "phase3_M300_items.csv")
GROUPS = [("correct-modal", "correct answer most common", "C0"), ("present, non-modal", "correct present, not most common", "C1"),
          ("absent", "correct never sampled", "C3"), ("tie", "tie", "C7")]
# where does a 15-way vote over the no-filler samples predict no gain? (mean prediction at each pass-rate level)
vote = it.assign(v=it["maj@15"] - it.p0).groupby("p0").v.mean()
flat_to = max([p for p, v in vote.items() if v <= 0.005 and p < 1], default=0.0)  # shade up to here


def boot(x):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), size=(10_000, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, col, title in [(axes[0], "pf", "With filler (100 dots)"), (axes[1], "pT0.4", "Temperature 0.4, no filler")]:
    ax.axvspan(-0.05, flat_to + 0.0625, color="0.9", zorder=0, label="majority vote predicts no gain")
    ax.plot(vote.index, vote.values, color="0.4", lw=1.5, ls="--", label="majority-vote prediction (15 votes)")
    for g, label, color in GROUPS:
        s = it[it.group == g]
        d = s[col] - s.p0
        ax.scatter(s.p0 + rng.uniform(-0.022, 0.022, len(s)), d + rng.uniform(-0.012, 0.012, len(s)), s=14, alpha=0.45, color=color,
                   label=f"{label} (n={len(s)})")
        if g != "tie":
            m, lo, hi = boot(d)
            ax.errorbar([s.p0.mean()], [m], yerr=[[m - lo], [hi - m]], marker="D", ms=10, color=color, mec="k", capsize=5, zorder=5)
            ax.annotate(f"{100 * m:+.0f} pp", (s.p0.mean(), m), textcoords="offset points", xytext=(10, 8), fontsize=9, fontweight="bold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("no-filler pass rate (held-out half of the samples, T=1)")
    ax.set_title(title, fontsize=11)
axes[0].set_ylabel("change in pass rate vs no filler at T=1")
axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="upper right")
fig.suptitle("Per-item change by group (300 items; groups from the other half of the no-filler samples; diamonds: group mean, 95% CI)", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_core_delta_by_group.png", dpi=150)

# ---------------- 2. histogram of per-item change vs a uniform boost ----------------
p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
e0 = (16 - p3[p3.cond == "k0_T1.0"].groupby("idx").correct.sum()).values  # errors out of 16, no filler
ef = (16 - p3[p3.cond == "F_T1.0"].groupby("idx").correct.sum()).values
delta = (e0 - ef) / 16
GRID = np.linspace(0, 1, 201)
L = binom.pmf(e0[:, None], 16, GRID[None, :])
pi = np.full(len(GRID), 1 / len(GRID))
for _ in range(300):  # distribution of true per-item no-filler error rates
    post = L * pi
    post /= post.sum(1, keepdims=True)
    pi = post.mean(0)
post = L * pi
post /= post.sum(1, keepdims=True)
cum = post.cumsum(1)


def simulate(shift, n=400):
    """Per-item change if every item's pass rate rose by the same amount (capped at 100%)."""
    u = rng.random((n, len(e0)))
    q = GRID[(u[:, :, None] > cum[None, :, :]).sum(2).clip(max=len(GRID) - 1)]  # true error rate per item
    return (rng.binomial(16, q) - rng.binomial(16, np.clip(q - shift, 0, 1))) / 16


target = delta.mean()
lo_s, hi_s = 0.0, 1.0
for _ in range(18):  # choose the boost whose average realised gain equals the observed one
    mid = (lo_s + hi_s) / 2
    lo_s, hi_s = (mid, hi_s) if simulate(mid, 60).mean() < target else (lo_s, mid)
shift = (lo_s + hi_s) / 2
sims = simulate(shift, 1000)
bins = np.linspace(-1.03125, 1.03125, 34)  # one bin per possible value (steps of 1/16)
centers = (bins[:-1] + bins[1:]) / 2
obs = np.histogram(delta, bins)[0]
sim_counts = np.array([np.histogram(s, bins)[0] for s in sims])
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(centers, obs, width=0.055, color="C0", label="observed (300 items)")
ax.plot(centers, sim_counts.mean(0), color="k", lw=1.8, label=f"uniform boost of the same average size (+{100 * target:.0f} pp)")
ax.fill_between(centers, np.percentile(sim_counts, 2.5, 0), np.percentile(sim_counts, 97.5, 0), color="k", alpha=0.15, label="95% range under a uniform boost")
ax.set_xlabel("change in pass rate with 100 dots (16 samples per item per condition)")
ax.set_ylabel("number of items")
big, none = int((delta >= 0.5).sum()), int((np.abs(delta) <= 0.0625).sum())
exp_big, exp_none = (sims >= 0.5).sum(1).mean(), (np.abs(sims) <= 0.0625).sum(1).mean()
ax.set_title(f"Per-item change with filler vs a uniform boost of the same average size\n"
             f"{big} items rise by ≥ 50 pp (uniform boost: {exp_big:.0f}); {none} barely move (uniform boost: {exp_none:.0f})", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_change_histogram.png", dpi=150)
print(f"histogram: observed SD {delta.std():.3f} vs uniform boost {sims.std(1).mean():.3f}; rise >= 50 pp: {big} vs {exp_big:.1f}; "
      f"within ±1/16: {none} vs {exp_none:.1f}; fall >= 25 pp: {(delta <= -0.25).sum()} vs {(sims <= -0.25).sum(1).mean():.1f}")

# ---------------- 3. disruption ----------------
dz = json.load(open(RES / "followup_disrupt25.json"))["primary"]
order = [("intact 25", "25 dots\nintact", "C0"), ("early", "letters at\nunits 1–8", "C1"), ("middle", "letters at\nunits 10–17", "C1"),
         ("late", "letters at\nunits 18–25", "C1"), ("17 dots", "17 dots intact\n(matched control)", "C2")]
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for i, (key, label, color) in enumerate(order):
    c = dz["conditions"][key]
    m, lo, hi = 100 * c["pass_rate"], 100 * c["ci"][0], 100 * c["ci"][1]
    ax.bar(i, m, color=color)
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color="k", capsize=6, lw=1.5)
    ax.text(i, hi + 2, f"{m:.0f}%\n[{lo:.0f}, {hi:.0f}]", ha="center", fontsize=9)
nf = 100 * dz["conditions"]["no filler"]["pass_rate"]
ax.axhline(nf, color="0.4", ls="--", lw=1)
ax.text(len(order) - 0.45, nf + 1.5, f"no filler: {nf:.0f}%", ha="right", fontsize=9, color="0.2", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_xticks(range(len(order)))
ax.set_xticklabels([o[1] for o in order], fontsize=9)
ax.set_ylim(0, 110)
ax.set_ylabel("pass rate (%), T=1, 16 samples per item")
ax.set_title(f"Damaging a third of a 25-dot filler, by location ({dz['n_items']} items; bars: mean, whiskers: 95% CI over items)\n"
             "No location differs detectably from simply having 17 dots; intervals are about ±10 pp", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_disruption.png", dpi=150)
print("wrote 3 figures; vote-flat region shaded up to no-filler pass rate", flat_to)
