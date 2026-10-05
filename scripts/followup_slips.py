"""Follow-up: parity, size and offset spectrum of V4 Flash's slips (existing data, no API calls).

Run: python scripts/followup_slips.py
Task: y = c1*x +/- k1, answer z = c2*y +/- k2, with c1, c2 in {2, 3}.
Implied error in y for a wrong answer: dy = (z_wrong -/+ k2) / c2 - y, defined only when the
division is exact ("clean inverse"); other wrong answers are final-step errors and are excluded.
Parity caveats handled by splitting on the coefficients:
  - z's parity is fixed by k2 when c2 = 2, so parity is tested on dy, never on z.
  - y's parity is fixed by k1 when c1 = 2 (2x is always even), so a model that merely knows
    "twice anything is even" keeps y's parity for free. The clean test is c1 = 3.
  - with c2 = 2, a tens slip in the FINAL multiplication (z off by 10) shows up as dy = +/-5;
    with c2 = 3 it is not a clean inverse and is excluded. So c2 = 3 gives the cleanest dy.
Data: Phase 3 M=300 samples (T=1, 16 per item, no filler vs dots k=100) and the dose-response run.
Writes data/results/followup_slips.json and scripts/followup_slips.png.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
COEF = {"twice": 2, "three times": 3}
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
D = 30
ABS = np.arange(1, D + 1)
rng = np.random.default_rng(0)
out = {}

items = {it["idx"]: it for it in map(json.loads, open(ROOT / "data/tasks/eval.jsonl"))}
for it in items.values():
    w = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).group(1)
    it["c1"] = COEF[w]


def implied_dy(z_wrong, it):
    rem = z_wrong - it["constant"] if it["operation"] == "plus" else z_wrong + it["constant"]
    return rem // it["coefficient"] - it["chain"]["y"] if rem % it["coefficient"] == 0 else None


def prep(df):
    df = df[df.parsed.notna() & ~df.correct].copy()
    df["z_err"] = [int(a) - items[i]["answer"] for a, i in zip(df.parsed, df.idx)]
    df["dy"] = [implied_dy(int(a), items[i]) for a, i in zip(df.parsed, df.idx)]
    df["c1"] = [items[i]["c1"] for i in df.idx]
    df["c2"] = [items[i]["coefficient"] for i in df.idx]
    # drop binding errors (question applied to another variable): they are not slips
    df = df[[int(a) not in items[i]["rivals"].values() for a, i in zip(df.parsed, df.idx)]]
    return df


p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
n_samples = p3.groupby("cond").size()
w = prep(p3[p3.cond.isin(["k0_T1.0", "F_T1.0"])])
w["cond"] = w.cond.map({"k0_T1.0": "no filler", "F_T1.0": "dots k=100"})
slips = w[w.dy.notna() & (w.dy != 0)].copy()
slips["dy"] = slips.dy.astype(int)
CONDS = ["no filler", "dots k=100"]
print(f"wrong, non-binding answers: {w.groupby('cond').size().to_dict()} | with a clean inverse: {slips.groupby('cond').size().to_dict()}")


def cluster_ci(d, fn, n=2000):
    """Bootstrap over items of a statistic fn(sub-frame)."""
    groups = [g for _, g in d.groupby("idx")]
    if len(groups) < 5:
        return float("nan"), float("nan")
    vals = [fn(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(n)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


# ---------------- 1. do slips keep the parity of y? ----------------
print("\n=== 1. share of slips whose implied y error is EVEN (chance 50%) ===")
out["parity"] = {}
even = lambda d: float((d.dy % 2 == 0).mean())
for c1 in [3, 2]:
    for c2 in [3, 2, "all"]:
        for cond in CONDS:
            d = slips[(slips.c1 == c1) & (slips.cond == cond) & ((slips.c2 == c2) if c2 != "all" else True)]
            lo, hi = cluster_ci(d, even, 1000)
            note = "  <- cleanest test" if (c1 == 3 and c2 == 3) else "  (y parity forced by k1)" if c1 == 2 else ""
            out["parity"][f"c1={c1}|c2={c2}|{cond}"] = {"share_even": even(d), "ci": [lo, hi], "n": int(len(d)), "items": int(d.idx.nunique())}
            print(f"c1={c1} c2={c2!s:3s} {cond:11s}: {even(d):.1%} [{lo:.1%}, {hi:.1%}]  (n={len(d)}, {d.idx.nunique()} items){note}")
# parity among even non-multiples of 10 vs odd, to separate "parity" from "tens slip"
d = slips[(slips.c1 == 3) & (slips.dy.abs() <= 9)]
print("c1=3, |dy| <= 9 only (no tens slips possible): share even:",
      {c: f"{even(d[d.cond == c]):.1%} (n={len(d[d.cond == c])})" for c in CONDS})
out["parity"]["c1=3_absdy_le9"] = {c: {"share_even": even(d[d.cond == c]), "n": int(len(d[d.cond == c]))} for c in CONDS}

# ---------------- 2. do slips shrink, or just become rarer? ----------------
print("\n=== 2. size of slips: |implied y error| ===")
out["size"] = {}
for label, sub in [("all slips", slips), ("c1=3 only", slips[slips.c1 == 3])]:
    for cond in CONDS:
        a = sub[sub.cond == cond].dy.abs()
        rate = len(a) / n_samples[{"no filler": "k0_T1.0", "dots k=100": "F_T1.0"}[cond]]
        out["size"][f"{label}|{cond}"] = {"n": int(len(a)), "rate_of_samples": float(rate), "median": float(a.median()), "mean": float(a.mean()),
                                         "q25": float(a.quantile(.25)), "q75": float(a.quantile(.75)), "share_le2": float((a <= 2).mean()),
                                         "share_le10": float((a <= 10).mean()), "share_gt30": float((a > 30).mean())}
        print(f"{label:10s} {cond:11s}: {rate:.1%} of samples | median {a.median():.0f} | quartiles {a.quantile(.25):.0f}-{a.quantile(.75):.0f} | "
              f"mean {a.mean():.1f} | <=2: {(a <= 2).mean():.0%} | <=10: {(a <= 10).mean():.0%} | >30: {(a > 30).mean():.0%}")
med = lambda d: d[d.cond == "dots k=100"].dy.abs().median() - d[d.cond == "no filler"].dy.abs().median()
lo, hi = cluster_ci(slips, med, 1000)
print(f"difference in median |dy| (filler - no filler): {med(slips):+.1f} [{lo:+.1f}, {hi:+.1f}]")
out["size"]["median_diff"] = {"diff": float(med(slips)), "ci": [lo, hi]}
# within-item: items with at least 4 slips in BOTH conditions
per = slips.groupby(["idx", "cond"]).dy.agg(lambda s: s.abs().median()).unstack()
cnt = slips.groupby(["idx", "cond"]).size().unstack(fill_value=0)
both = per[(cnt["no filler"] >= 4) & (cnt["dots k=100"] >= 4)].dropna()
wt = wilcoxon(both["dots k=100"], both["no filler"])
print(f"within-item (>= 4 slips in both conditions, {len(both)} items): median |dy| no filler {both['no filler'].median():.1f} "
      f"vs filler {both['dots k=100'].median():.1f}; smaller with filler in {(both['dots k=100'] < both['no filler']).mean():.0%}, "
      f"larger in {(both['dots k=100'] > both['no filler']).mean():.0%} (Wilcoxon p={wt.pvalue:.2g})")
out["size"]["within_item"] = {"n_items": int(len(both)), "median_no_filler": float(both["no filler"].median()), "median_filler": float(both["dots k=100"].median()),
                              "share_smaller": float((both["dots k=100"] < both["no filler"]).mean()), "share_larger": float((both["dots k=100"] > both["no filler"]).mean()),
                              "wilcoxon_p": float(wt.pvalue)}
# by filler length (dose-response items: 219 not at ceiling)
dose = pd.DataFrame(json.loads(l) for l in open(RES / "followup_dose.jsonl"))
run = set(dose.idx)
ends = p3[p3.cond.isin(["k0_T1.0", "F_T1.0"]) & p3.idx.isin(run)].assign(k=lambda d: np.where(d.cond == "k0_T1.0", 0, 100))
allk = pd.concat([ends, dose])
n_by_k = allk.groupby("k").size()
wk = prep(allk)
sk = wk[wk.dy.notna() & (wk.dy != 0)]
print("by filler length (219 items not at ceiling):")
out["size"]["by_k"] = {}
for k, g in sk.groupby("k"):
    a = g.dy.abs()
    out["size"]["by_k"][int(k)] = {"rate": float(len(a) / n_by_k[k]), "median": float(a.median()), "share_le10": float((a <= 10).mean()), "mean": float(a.mean())}
    print(f"  k={int(k):3d}: slips {len(a) / n_by_k[k]:.1%} of samples | median |dy| {a.median():.0f} | mean {a.mean():.1f} | <=10: {(a <= 10).mean():.0%}")

# ---------------- 3. offset spectrum of implied y errors ----------------
print("\n=== 3. offset spectrum of implied y errors (share of slips at each |dy|, +/- pooled) ===")
X = np.column_stack([np.ones(D), ABS % 2 == 0, ABS % 5 == 0, ABS % 10 == 0, 1.0 / ABS]).astype(float)
NAMES = ["baseline", "same mod 2", "same mod 5", "same mod 10", "1/|d|"]
out["spectrum"] = {}
specs = {}


def spectrum(d):
    a = d.dy.abs()
    return np.array([(a == k).mean() for k in ABS]) * 100  # % of slips


for label, sel in [("c1=3, c2=3 (cleanest)", (slips.c1 == 3) & (slips.c2 == 3)), ("c1=3, all c2", slips.c1 == 3),
                   ("c1=2 (y parity forced)", slips.c1 == 2), ("all", slips.c1 > 0)]:
    for cond in CONDS:
        d = slips[sel & (slips.cond == cond)]
        s = spectrum(d)
        specs[(label, cond)] = s
        coef = np.linalg.lstsq(X, s, rcond=None)[0]
        groups = [g for _, g in d.groupby("idx")]
        boot = np.array([np.linalg.lstsq(X, spectrum(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])), rcond=None)[0]
                         for _ in range(500)])
        res = {n: {"coef": float(c), "ci": [float(np.percentile(boot[:, j], 2.5)), float(np.percentile(boot[:, j], 97.5))]} for j, (n, c) in enumerate(zip(NAMES, coef))}
        res["share_within_30"] = float((d.dy.abs() <= D).mean())
        res["n"] = int(len(d))
        res["spectrum_pct_of_slips"] = {int(k): float(v) for k, v in zip(ABS, s)}
        out["spectrum"][f"{label}|{cond}"] = res
        print(f"\n{label} | {cond} (n={len(d)}, {(d.dy.abs() <= D).mean():.0%} within ±{D})")
        print("  % of slips at |dy| = 1..20: " + " ".join(f"{k}:{v:.1f}" for k, v in zip(ABS[:20], s[:20])))
        print("  regression (pp): " + " | ".join(f"{n} {res[n]['coef']:+.2f} [{res[n]['ci'][0]:+.2f}, {res[n]['ci'][1]:+.2f}]" for n in NAMES[1:]))
        print(f"  contrasts (pp): ±5 vs ±3,±7: {s[4] - (s[2] + s[6]) / 2:+.2f} | ±10 vs ±8,±12: {s[9] - (s[7] + s[11]) / 2:+.2f} | "
              f"even vs odd (|d|<=9, non-multiples of 5): {s[[1, 3, 5, 7]].mean() - s[[0, 2, 6, 8]].mean():+.2f} | ±1 vs ±3: {s[0] - s[2]:+.2f}")

# compare with the Brauer decode halo around the computed sum
bh = json.load(open(RES / "followup_brauer_spectrum.json"))
for model in ["deepseek_v3", "kimi_k2"]:
    halo = np.array([bh[f"{model}|sum"]["excess_by_abs_offset"][str(k)] for k in ABS])
    for label in ["c1=3, c2=3 (cleanest)", "c1=3, all c2", "all"]:
        r = float(np.corrcoef(halo, specs[(label, "no filler")])[0, 1])
        out["spectrum"][f"corr_{model}_halo_vs_{label}"] = r
        print(f"correlation over |d| = 1..30, {model} decode halo vs V4 Flash slips ({label}, no filler): r = {r:.2f}")
json.dump(out, open(RES / "followup_slips.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
colors = ["C3" if a % 10 == 0 else "C1" if a % 5 == 0 else "C0" if a % 2 == 0 else "C7" for a in ABS]
for ax, label in zip(axes[:2], ["c1=3, all c2", "c1=2 (y parity forced)"]):
    ax.bar(ABS, specs[(label, "no filler")], color=colors)
    ax.plot(ABS, specs[(label, "dots k=100")], "k.-", lw=0.8, ms=4, label="dots k=100")
    ax.set_title(f"V4 Flash slips, {label}: no filler (bars)", fontsize=9)
    ax.set_xlabel("|implied error in y|  (red: ×10, orange: other ×5, blue: other even, grey: odd)", fontsize=8)
    ax.set_ylabel("% of slips")
    ax.legend(fontsize=8)
bins = [0, 2, 5, 10, 20, 30, 60, 1000]
labels = ["1–2", "3–5", "6–10", "11–20", "21–30", "31–60", ">60"]
for off, cond, color in [(-0.2, "no filler", "C7"), (0.2, "dots k=100", "C0")]:
    a = slips[slips.cond == cond].dy.abs()
    h = np.histogram(a, bins=bins)[0] / len(a) * 100
    axes[2].bar(np.arange(len(labels)) + off, h, 0.4, color=color, label=cond)
axes[2].set_xticks(range(len(labels)))
axes[2].set_xticklabels(labels)
axes[2].set_xlabel("|implied error in y|")
axes[2].set_ylabel("% of slips")
axes[2].set_title("Size of slips, with and without filler", fontsize=9)
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_slips.png", dpi=150)

# ---------------- 1b. answer parity vs intermediate parity (added after the first pass) ----------------
# For c2 = 3, dy's parity equals the parity of the answer error, so "dy even" cannot separate
# "y's parity is kept" from "the answer's parity is kept". Only c2 = 2 separates them.
# Structural errors (sign flips etc.) are dropped here because a sign flip is always an even error.
ap_ = lambda c, op, k, v: c * v + k if op == "plus" else c * v - k
fl_ = lambda op: "minus" if op == "plus" else "plus"


def structural(a, it):
    w_, ref, op1, k1 = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).groups()
    c1, k1, x, y = COEF[w_], int(k1), it["chain"]["x"], it["chain"]["y"]
    c2, op2, k2 = it["coefficient"], it["operation"], it["constant"]
    others = [v for n, v in it["values"].items() if n != it["queried_term"]]
    ys = {ap_(c1, fl_(op1), k1, x), c1 * x} | {ap_(c, op1, k1, x) for c in (2, 3) if c != c1} | {ap_(c1, op1, k1, v) for v in others if v != x}
    return a in {ap_(c2, op2, k2, v) for v in ys} | {ap_(c2, fl_(op2), k2, y), c2 * y, y, ap_(1, op2, k2, y)} | {ap_(c, op2, k2, y) for c in (2, 3) if c != c2}


w["struct"] = [structural(int(a), items[i]) for a, i in zip(w.parsed, w.idx)]
ww = w[~w.struct]
out["answer_parity"], out["dy_parity_nonstructural"] = {}, {}
print("\n=== 1b. parity of the FINAL ANSWER kept (answer error even), non-structural wrong answers ===")
for c2 in (2, 3):
    for cond in CONDS:
        d = ww[(ww.c2 == c2) & (ww.cond == cond)]
        r = {"n": int(len(d)), "answer_error_even": float((d.z_err % 2 == 0).mean()), "divisible_by_c2": float((d.z_err % c2 == 0).mean()),
             "divisible_by_2c2": float((d.z_err % (2 * c2) == 0).mean())}
        out["answer_parity"][f"c2={c2}|{cond}"] = r
        print(f"c2={c2} {cond:11s}: answer error even {r['answer_error_even']:.1%} | divisible by c2 {r['divisible_by_c2']:.1%} | "
              f"divisible by 2*c2 {r['divisible_by_2c2']:.1%} (n={len(d)})")
s2 = ww[ww.dy.notna() & (ww.dy != 0)].copy()
s2["dy"] = s2.dy.astype(int)
print("parity of the implied y error, structural errors dropped (c2 = 2 is the only cell not tied to answer parity):")
for c1 in (3, 2):
    for c2 in (3, 2):
        for cond in CONDS:
            d = s2[(s2.c1 == c1) & (s2.c2 == c2) & (s2.cond == cond)]
            if len(d) < 20:
                continue
            lo, hi = cluster_ci(d, even, 600)
            out["dy_parity_nonstructural"][f"c1={c1}|c2={c2}|{cond}"] = {"share_even": even(d), "ci": [lo, hi], "n": int(len(d)),
                                                                        "share_even_excl_multiples_of_5": even(d[d.dy % 5 != 0])}
            print(f"  c1={c1} c2={c2} {cond:11s}: even {even(d):.1%} [{lo:.1%}, {hi:.1%}] (n={len(d)}) | excluding multiples of 5: {even(d[d.dy % 5 != 0]):.1%}")
out["size"]["nonstructural"] = {c: {"median": float(s2[s2.cond == c].dy.abs().median()), "mean": float(s2[s2.cond == c].dy.abs().mean()),
                                    "share_undershoot": float((s2[s2.cond == c].dy < 0).mean())} for c in CONDS}
print("size with structural errors dropped:", out["size"]["nonstructural"])
json.dump(out, open(RES / "followup_slips.json", "w"), indent=1)
