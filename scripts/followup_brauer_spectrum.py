"""Follow-up: offset spectrum of the near-miss halo in Brauer et al.'s released decodes.

Run: python scripts/followup_brauer_spectrum.py   (no API calls, no model runs)
1. Excess (own decode minus other items' decodes) at each |offset| 1..30 from the 2-fact SUM,
   regressed on: same mod 2, same mod 5, same mod 10.
2. The same spectrum around the retrieved addends: same shape at lower amplitude, or flat?
3. The release has no no-filler decodes; the nearest available comparison is filler length.
(item, offset) pairs within 2 of one of the item's other true values are dropped, so the halo of
one quantity cannot leak into another's spectrum.
Writes data/results/followup_brauer_spectrum.json and scripts/followup_brauer_spectrum.png.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "vendor/filler-token-reasoning/release/top_tokens"
D = 30
OFFS = [d for d in range(-D, D + 1) if d != 0]
ABS = np.arange(1, D + 1)
X = np.column_stack([np.ones(D), ABS % 2 == 0, ABS % 5 == 0, ABS % 10 == 0]).astype(float)
XD = np.column_stack([X, 1.0 / ABS])  # same, plus a smooth closeness term
NAMES = ["baseline", "same mod 2", "same mod 5", "same mod 10"]
rng = np.random.default_rng(0)
out = {}


def load(model, conds):
    ex = []
    for c in conds:
        for e in json.load(open(REL / f"{model}_2fact_{c}.json")):
            nums = {int(s) for s in (t["str"].strip() for t in e["top_tokens"]) if s.isascii() and s.isdecimal() and len(s) <= 4}
            ex.append({"file": c, "a1": e["fact_value_1"], "a2": e["fact_value_2"], "sum": e["answer"], "nums": nums})
    return ex


def matrices(ex, target):
    """own[i, j], other[i, j] for offsets OFFS; NaN where the offset is near another true value."""
    own = np.full((len(ex), len(OFFS)), np.nan)
    oth = np.full((len(ex), len(OFFS)), np.nan)
    counts = {f: Counter(n for e in ex if e["file"] == f for n in e["nums"]) for f in {e["file"] for e in ex}}
    size = Counter(e["file"] for e in ex)
    for i, e in enumerate(ex):
        rest = [e[k] for k in ("a1", "a2", "sum") if k != target]
        for j, d in enumerate(OFFS):
            n = e[target] + d
            if n < 0 or any(abs(n - r) <= 2 for r in rest):
                continue
            o = n in e["nums"]
            own[i, j] = o
            oth[i, j] = (counts[e["file"]][n] - o) / (size[e["file"]] - 1)
    return own, oth


def folded(own, oth, rows=None):
    """Excess by |d| = 1..30, averaging the + and - offsets."""
    if rows is not None:
        own, oth = own[rows], oth[rows]
    ex = np.nanmean(own, 0) - np.nanmean(oth, 0)
    return np.array([(ex[OFFS.index(a)] + ex[OFFS.index(-a)]) / 2 for a in ABS])


def analyse(ex, target):
    own, oth = matrices(ex, target)
    spec = folded(own, oth)
    boots = np.array([folded(own, oth, rng.integers(0, len(ex), len(ex))) for _ in range(1000)])
    fit = lambda M, y: np.linalg.lstsq(M, y, rcond=None)[0]
    coef, cb = fit(X, spec), np.array([fit(X, b) for b in boots])
    coefd, cbd = fit(XD, spec), np.array([fit(XD, b) for b in boots])
    res = {"excess_by_abs_offset": {int(a): float(v) for a, v in zip(ABS, spec)},
           "regression": {n: {"coef": float(c), "ci": [float(np.percentile(cb[:, k], 2.5)), float(np.percentile(cb[:, k], 97.5))]}
                          for k, (n, c) in enumerate(zip(NAMES, coef))},
           "regression_with_closeness": {n: {"coef": float(c), "ci": [float(np.percentile(cbd[:, k], 2.5)), float(np.percentile(cbd[:, k], 97.5))]}
                                         for k, (n, c) in enumerate(zip(NAMES + ["1/|d|"], coefd))}}

    def contrast(a, b):
        pick = lambda s, ds: s[..., [d - 1 for d in ds]].mean(-1)
        v, bs = pick(spec, a) - pick(spec, b), pick(boots, a) - pick(boots, b)
        return {"diff": float(v), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    res["contrasts"] = {"±5 vs ±3, ±7": contrast([5], [3, 7]), "±15 vs ±13, ±17": contrast([15], [13, 17]),
                        "±1 vs ±3": contrast([1], [3]), "±10 vs ±8, ±12": contrast([10], [8, 12]),
                        "±20 vs ±18, ±22": contrast([20], [18, 22]), "±30 vs ±28": contrast([30], [28]),
                        "even vs odd (non-multiples of 5, |d| <= 9)": contrast([2, 4, 6, 8], [1, 3, 7, 9])}
    return res, spec, boots


pc = lambda v: f"{100 * v:+.1f}"
fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
for row, (model, name) in enumerate([("deepseek_v3", "DeepSeek V3"), ("kimi_k2", "Kimi K2")]):
    ex = load(model, ["dots_10", "dots_25", "dots_50"])
    print(f"\n================ {name}, dots filler, {len(ex)} correct examples ================")
    specs = {}
    for col, (target, label) in enumerate([("sum", "SUM (computed)"), ("addend", "ADDENDS (retrieved, A1 and A2 pooled)")]):
        if target == "sum":
            res, spec, _ = analyse(ex, "sum")
        else:  # pool A1 and A2 by stacking examples with the roles swapped
            ra, sa, _ = analyse(ex, "a1")
            rb, sb, _ = analyse(ex, "a2")
            spec = (sa + sb) / 2
            res = {"A1": ra, "A2": rb, "excess_by_abs_offset": {int(a): float(v) for a, v in zip(ABS, spec)}}
        specs[target] = spec
        out[f"{model}|{target}"] = res
        print(f"\n--- {label} ---")
        print("excess (pp) at |d| = 1..30: " + " ".join(f"{a}:{pc(v)}" for a, v in zip(ABS, spec)))
        for r, tag in ([(res, "")] if target == "sum" else [(res["A1"], "A1 "), (res["A2"], "A2 ")]):
            print(f"{tag}regression (pp): " + " | ".join(f"{n} {pc(v['coef'])} [{pc(v['ci'][0])}, {pc(v['ci'][1])}]" for n, v in r["regression"].items()))
            print(f"{tag}with closeness term: " + " | ".join(f"{n} {pc(v['coef'])} [{pc(v['ci'][0])}, {pc(v['ci'][1])}]" for n, v in r["regression_with_closeness"].items()))
            print(f"{tag}contrasts (pp): " + " | ".join(f"{k}: {pc(v['diff'])} [{pc(v['ci'][0])}, {pc(v['ci'][1])}]" for k, v in r["contrasts"].items()))
        ax = axes[row, col]
        colors = ["C3" if a % 10 == 0 else "C1" if a % 5 == 0 else "C0" if a % 2 == 0 else "C7" for a in ABS]
        ax.bar(ABS, 100 * spec, color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name}: excess around the {label.lower()}", fontsize=9)
        ax.set_ylabel("own − other items (pp)")
    r = float(np.corrcoef(specs["sum"], specs["addend"])[0, 1])
    amp = float(np.abs(specs["addend"]).mean() / np.abs(specs["sum"]).mean())
    out[f"{model}|shape"] = {"corr_sum_vs_addend_spectrum": r, "mean_abs_excess_ratio_addend_over_sum": amp}
    print(f"\nshape comparison: correlation between the sum and addend spectra over |d| = 1..30: r = {r:.2f}; "
          f"mean |excess| around addends is {amp:.0%} of that around the sum")

    print("\nby filler length (no no-filler decodes exist in the release):")
    for c in ["dots_10", "dots_25", "dots_50"]:
        e = load(model, [c])
        own, oth = matrices(e, "sum")
        s = folded(own, oth)
        present = np.mean([x["sum"] in x["nums"] for x in e])
        out[f"{model}|{c}"] = {"n": len(e), "sum_present": float(present), "mean_excess_1_10": float(s[:10].mean()),
                               "even": float(s[[1, 3, 5, 7]].mean()), "odd": float(s[[0, 2, 6, 8]].mean()), "pm10": float(s[9]), "pm5": float(s[4])}
        print(f"  {c}: n={len(e)} | sum itself present {present:.0%} | mean excess |d|<=10: {pc(s[:10].mean())} pp | "
              f"even {pc(s[[1, 3, 5, 7]].mean())} | odd {pc(s[[0, 2, 6, 8]].mean())} | ±5 {pc(s[4])} | ±10 {pc(s[9])}")
for ax in axes[1]:
    ax.set_xlabel("|offset| from the true value   (red: multiple of 10, orange: other multiple of 5, blue: other even, grey: odd)")
fig.suptitle("Offset spectrum of near-miss values in Brauer et al.'s filler decodes (2-fact addition, dots)", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_brauer_spectrum.png", dpi=150)
json.dump(out, open(ROOT / "data/results/followup_brauer_spectrum.json", "w"), indent=1)
