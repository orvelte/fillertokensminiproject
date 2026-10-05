"""One figure per finding, for presentation (existing results only; no API calls).

Run: python scripts/talk_figures.py
Writes scripts/fig_f12_groups.png, fig_f3_never_sampled.png, fig_f5_positions.png and
fig_hypotheses_schematic.png. (Finding 4's histogram is fig_change_histogram.png and the
appendix scatter is fig_core_delta_by_group.png, both from summary_figures.py; the context
curve is phase2_uplift_pilot.png.)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RES, HERE = ROOT / "data" / "results", Path(__file__).parent
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), size=(10_000, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


# ---------------- findings 1 + 2: grouped bars ----------------
it = pd.read_csv(RES / "phase3_M300_items.csv")
groups = [("correct-modal", "correct answer\nmost common"), ("present, non-modal", "correct present,\nnot most common"), ("absent", "correct\nnever sampled")]
series = [("pf", "filler (100 dots)", "C0"), ("pT0.7", "temperature 0.7, no filler", "C4"), ("pT0.4", "temperature 0.4, no filler", "C6")]
fig, ax = plt.subplots(figsize=(9.5, 5))
w = 0.25
for j, (col, label, color) in enumerate(series):
    for i, (g, _) in enumerate(groups):
        s = it[it.group == g]
        m, lo, hi = boot(s[col] - s.p0)
        ax.bar(i + (j - 1) * w, m, w, color=color, label=label if i == 0 else None)
        ax.errorbar(i + (j - 1) * w, m, yerr=[[m - lo], [hi - m]], color="k", capsize=4, lw=1.2)
        ax.text(i + (j - 1) * w, (hi if m >= 0 else lo) + (1.2 if m >= 0 else -3.5), f"{m:+.0f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=1)
for i in (1, 2):  # the groups where the most common no-filler answer is wrong
    ax.plot([i - 0.42, i + 0.42], [0, 0], color="C3", lw=4, solid_capstyle="butt", zorder=4)
    ax.annotate("voting and sharpening\npredict ≤ 0 here", (i + 0.25, 0), xytext=(i + 0.05, -9.5), fontsize=8.5, color="C3", ha="center",
                arrowprops=dict(arrowstyle="-", color="C3", lw=0.8))
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{lab}\n(n={int((it.group == g).sum())})" for g, lab in groups])
ax.set_ylim(-14, 50)
ax.set_ylabel("mean change in pass rate vs no filler at T=1 (pp)")
ax.set_title("Filler lifts items that voting and lower temperature cannot\n"
             "(300 items, 16 samples per condition; groups from a held-out half of the no-filler samples; 95% CI over items)", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(HERE / "fig_f12_groups.png", dpi=150)

# ---------------- finding 3: never-sampled items ----------------
p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
nofill = p3[p3.cond.str.startswith("k0")].groupby("idx").correct.agg(["sum", "size"])
fill = p3[p3.cond == "F_T1.0"].groupby("idx").correct.sum()
never = nofill[nofill["sum"] == 0].index
assert (nofill.loc[never, "size"] == 48).all()
nf = fill.loc[never]
cats = [("0 of 16 correct with filler", nf == 0, "0.6", 1.0), ("at least one correct with filler", (nf >= 1) & (nf < 8), "C1", 1.6),
        ("majority correct with filler", nf >= 8, "C3", 2.4)]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]})
for label, mask, color, lw in cats:
    ys = (nf[mask] / 16).values
    for y in ys:
        a.plot([0, 1], [0, y + rng.uniform(-0.006, 0.006)], color=color, lw=lw, alpha=0.8, marker="o", ms=4)
    a.plot([], [], color=color, lw=lw, marker="o", ms=4, label=f"{label}: {int(mask.sum())} items")
a.set_xticks([0, 1])
a.set_xticklabels(["no filler\n(0 correct in 48 samples\nacross T = 1, 0.7, 0.4)", "100 dots\n(16 samples, T=1)"])
a.set_xlim(-0.25, 1.25)
a.set_ylim(-0.05, 1.05)
a.set_ylabel("pass rate")
n_any, n_maj = int((nf >= 1).sum()), int((nf >= 8).sum())
a.set_title(f"{len(never)} items never answered correctly without filler:\n{n_any} get at least one correct answer with filler, {n_maj} reach a majority", fontsize=10)
a.legend(fontsize=9, loc="upper left")
main = json.load(open(RES / "phase3_M300_summary.json"))["groups"]["absent"]
hard_path = ROOT / "archive/data/results/phase3_M60_hard_summary.json"
hard = json.load(open(hard_path))["groups"]["absent"] if hard_path.exists() else {"n": 27, "delta": 0.150, "ci": [0.067, 0.248]}  # values recorded in notes/phase3.md
for i, (label, g, color) in enumerate([("main run\n(300 items)", main, "C0"), ("harder-task pilot\n(60 items)", hard, "C2")]):
    m, lo, hi = 100 * g["delta"], 100 * g["ci"][0], 100 * g["ci"][1]
    b.errorbar(i, m, yerr=[[m - lo], [hi - m]], marker="o", ms=9, color=color, capsize=6, lw=1.5)
    b.text(i + 0.12, m, f"+{m:.0f} pp\n[{lo:+.0f}, {hi:+.0f}]\nn = {g['n']}", va="center", fontsize=9)
    b.text(i, -3.2, label, ha="center", va="top", fontsize=9)
b.axhline(0, color="k", lw=1)
b.set_xlim(-0.5, 1.9)
b.set_ylim(-9, 32)
b.set_xticks([])
b.set_ylabel("mean change in pass rate with filler (pp)")
b.set_title("Group mean for 'correct never sampled'\n(group defined on 8 held-out no-filler samples)", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "fig_f3_never_sampled.png", dpi=150)

# ---------------- finding 5: positions interchangeable ----------------
dz = json.load(open(RES / "followup_disrupt25.json"))["primary"]
order = [("no filler", "no filler\n(reference)", "0.4"), ("intact 25", "25 dots\nintact", "C0"), ("17 dots", "17 clean dots\n(matched control)", "C2"),
         ("early", "letters at\nunits 1–8", "C1"), ("middle", "letters at\nunits 10–17", "C1"), ("late", "letters at\nunits 18–25", "C1")]
fig, ax = plt.subplots(figsize=(9.5, 5))
ctrl = 100 * dz["conditions"]["17 dots"]["pass_rate"]
ax.axhline(ctrl, color="C2", ls="--", lw=1.2, label=f"matched 17-dot control ({ctrl:.0f}%)")
for i, (key, label, color) in enumerate(order):
    c = dz["conditions"][key]
    m, lo, hi = 100 * c["pass_rate"], 100 * c["ci"][0], 100 * c["ci"][1]
    ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], marker="o", ms=10, color=color, capsize=7, lw=1.8)
    ax.text(i + 0.13, m, f"{m:.0f}%\n[{lo:.0f}, {hi:.0f}]", va="center", fontsize=9)
ax.axvspan(2.6, 5.6, color="C1", alpha=0.07)
ax.text(4.1, 100, "8 of the 25 dots replaced by random letters", ha="center", fontsize=9, color="C1")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([o[1] for o in order], fontsize=9)
ax.set_xlim(-0.5, 5.75)
ax.set_ylim(0, 105)
ax.set_ylabel("pass rate (%), T=1, 16 samples per item")
ax.set_title(f"Damaging a third of the filler behaves like having a third fewer dots, wherever the damage is\n"
             f"({dz['n_items']} items; points: mean, bars: 95% CI over items)", fontsize=10)
ax.legend(fontsize=9, loc="lower right")
fig.text(0.5, 0.005, "Intervals are about ±10 pp: an early-location effect of up to about 18 points cannot be excluded.", ha="center", fontsize=9, style="italic")
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "fig_f5_positions.png", dpi=150)

# ---------------- hypotheses schematic (conceptual, not data) ----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))


def box(ax, x, y, w, h, text, fc, fs=10, ec="k"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, p, q, color="0.3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, color=color, lw=1.3))


xs = [0.6, 2.4, 4.2, 6.0]
for ax, title in zip(axes, ["H1: parallel attempts", "H2: one computation, spread out"]):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 8.2)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    box(ax, 0.1, 6.7, 8.0, 0.9, "question:  y = 3·x − 30,   answer = 2·y + 36      (x = 48, so y = 114)", "#eeeeee", 9.5)
    for x in xs:
        box(ax, x, 5.35, 1.6, 0.55, ".", "white", 13)
        arrow(ax, (x + 0.8, 6.68), (x + 0.8, 5.95))
    ax.text(0.3, 5.62, "filler\npositions", ha="right", va="center", fontsize=9)
a1, a2 = axes
for x, v, fc in zip(xs, ["y = 114", "y = 104", "y = 114", "y = 124"], ["#cfe8cf", "#f6d2d2", "#cfe8cf", "#f6d2d2"]):
    box(a1, x, 4.2, 1.6, 0.8, v, fc, 11)
    arrow(a1, (x + 0.8, 5.33), (x + 0.8, 5.03))
    arrow(a1, (x + 0.8, 4.18), (4.1, 3.25))
box(a1, 2.6, 2.45, 3.0, 0.8, "selector at the answer\npicks one candidate", "#fff2cc", 10)
arrow(a1, (4.1, 2.43), (4.1, 1.85))
box(a1, 3.0, 1.05, 2.2, 0.8, "answer = 264", "#cfe8cf", 11)
a1.text(8.2, 4.6, "each position holds\na complete, rival\nvalue of y", fontsize=9.5, va="center")
for x, v in zip(xs, ["product\n3·48", "subtract\n− 30", "size\n≈ 110", "parity\neven"]):
    box(a2, x, 4.2, 1.6, 0.8, v, "#dbe9f6", 10)
    arrow(a2, (x + 0.8, 5.33), (x + 0.8, 5.03))
    arrow(a2, (x + 0.8, 4.18), (4.1, 3.25))
box(a2, 2.6, 2.45, 3.0, 0.8, "components combine\ninto one value: y = 114", "#fff2cc", 10)
arrow(a2, (4.1, 2.43), (4.1, 1.85))
box(a2, 3.0, 1.05, 2.2, 0.8, "answer = 264", "#cfe8cf", 11)
a2.text(8.2, 4.6, "positions hold\ndifferent parts of\na single value of y", fontsize=9.5, va="center")
a1.text(4.1, -0.05, "White-box prediction: decoded candidates at different positions\nshift independently of one another.", ha="center", fontsize=10,
        bbox=dict(fc="white", ec="0.5", pad=5))
a2.text(4.1, -0.05, "White-box prediction: no rival candidates; damaging positions\nonly makes the one value less precise.", ha="center", fontsize=10,
        bbox=dict(fc="white", ec="0.5", pad=5))
fig.suptitle("CONCEPTUAL SKETCH, not data: two ways filler positions could be used", fontsize=11, color="C3")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(HERE / "fig_hypotheses_schematic.png", dpi=150)
print(f"wrote 4 figures | never-sampled items: {len(never)}, with >= 1 correct under filler: {n_any}, majority: {n_maj}")
