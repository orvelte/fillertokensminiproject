"""Follow-up: per-item dose-response of dot filler length (T=1, K=16 samples).

Run: python scripts/followup_dose.py
Adds k = 10, 25, 50 to the existing Phase 3 M=300 data (k = 0 and k = 100). Items already at
ceiling in both existing conditions (>= 15/16 correct) are skipped as uninformative.
Writes data/results/followup_dose.jsonl, data/results/followup_dose_items.csv,
data/results/followup_dose_summary.json and scripts/followup_dose.png.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import build_messages, parse_answer

NEW_KS, ALL_KS, K = [10, 25, 50], [0, 10, 25, 50, 100], 16
USD_PER_PROMPT_TOKEN = 0.14e-6

load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load("fewshot.jsonl"), {it["idx"]: it for it in load("eval.jsonl")}
p3 = pd.DataFrame(json.loads(l) for l in open(ROOT / "data/results/phase3_M300.jsonl"))
p3 = p3[p3.cond.isin(["k0_T1.0", "F_T1.0"])].assign(k=lambda d: np.where(d.cond == "k0_T1.0", 0, 100))
ends = p3.pivot_table(index="idx", columns="k", values="correct", aggfunc="mean")
run_idx = [i for i in ends.index if not (ends.loc[i, 0] >= 15 / 16 and ends.loc[i, 100] >= 15 / 16)]

reqs, meta = [], []
for k in NEW_KS:
    for i in run_idx:
        msgs = build_messages(few_shot, items[i], "dots", k)
        for s in range(K):
            reqs.append({"messages": msgs, "temperature": 1.0, "sample": s})
            meta.append((k, s, i))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(run_idx)} items, {len(reqs)} calls, projected <= ${projected:.2f} if nothing is cached", flush=True)
resps = api.chat_many(reqs, max_tokens=10, tag="followup_dose", projected_usd=projected)

rows = []
with open(ROOT / "data/results/followup_dose.jsonl", "w") as f:
    for (k, s, i), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        assert not api.has_reasoning(r)
        row = {"idx": i, "k": k, "sample": s, "response": text, "parsed": ans, "answer": items[i]["answer"],
               "correct": ans == items[i]["answer"]}
        rows.append(row)
        f.write(json.dumps({**row, "temperature": 1.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
df = pd.concat([pd.DataFrame(rows), p3[p3.idx.isin(run_idx)][["idx", "k", "sample", "parsed", "answer", "correct"]]])
df["ans"] = df.parsed.where(df.parsed.notna(), -10**9)

rate = df.pivot_table(index="idx", columns="k", values="correct", aggfunc="mean")[ALL_KS]
ndist = df.groupby(["idx", "k"]).ans.nunique().unstack()[ALL_KS]
mode = df.groupby(["idx", "k"]).ans.agg(lambda a: Counter(a).most_common(1)[0][0]).unstack()[ALL_KS]
d = rate[100] - rate[0]
kind = pd.Series(np.where(d >= 0.5, "flips up", np.where((rate[0] <= 0.25) & (rate[100] <= 0.25), "stays wrong", "other")), index=rate.index)
rate.assign(kind=kind).to_csv(ROOT / "data/results/followup_dose_items.csv")
out = {"n_items_run": len(run_idx), "kinds": kind.value_counts().to_dict()}

print("\nmean pass rate by k:")
m = rate.groupby(kind).mean().round(3)
print(m.assign(n=kind.value_counts()).to_string())
out["mean_rate_by_kind"] = m.to_dict("index")

print("\nmean number of distinct answers in 16 samples, by k:")
nd = ndist.groupby(kind).mean().round(2)
print(nd.to_string())
out["mean_distinct_by_kind"] = nd.to_dict("index")

# ---- per-item shape for the flip items (endpoints k=0 and k=100 were used to select them) ----
fl = rate[kind == "flips up"]
mid = fl[NEW_KS]
cells = {"low (<=25%)": float((mid <= 0.25).values.mean()), "middle": float(((mid > 0.25) & (mid < 0.75)).values.mean()),
         "high (>=75%)": float((mid >= 0.75).values.mean())}
first_high = fl.apply(lambda r: next((k for k in [10, 25, 50, 100] if r[k] >= 0.75), None), axis=1)
gain_frac = fl[NEW_KS].sub(fl[0], axis=0).div(fl[100] - fl[0], axis=0)  # share of the k=100 gain reached
mono = (fl[[10, 25, 50, 100]].diff(axis=1).iloc[:, 1:] >= -0.125).all(axis=1)
print(f"\nflip items (n={len(fl)}): share of (item, k) cells at k=10/25/50 that are low / middle / high: "
      + ", ".join(f"{k} {v:.0%}" for k, v in cells.items()))
print("  by k:", {k: {"low": round(float((fl[k] <= .25).mean()), 2), "mid": round(float(((fl[k] > .25) & (fl[k] < .75)).mean()), 2),
                      "high": round(float((fl[k] >= .75).mean()), 2)} for k in NEW_KS})
print("  smallest k with pass rate >= 75%:", first_high.value_counts(dropna=False).sort_index().to_dict())
print("  median share of the k=100 gain reached at k=10/25/50:", gain_frac.median().round(2).to_dict())
print(f"  roughly monotone from k=10 up (no drop > 2/16): {mono.mean():.0%}")
jump = fl[ALL_KS].diff(axis=1).iloc[:, 1:].max(axis=1)
print(f"  largest single-step rise between adjacent k values: median {jump.median():.2f}; "
      f">= 0.5 in {np.mean(jump >= 0.5):.0%} of flip items")
out["flip_items"] = {"n": int(len(fl)), "cells_mid_k": cells, "first_k_high": {str(k): int(v) for k, v in first_high.value_counts(dropna=False).items()},
                     "median_gain_share": gain_frac.median().to_dict(), "monotone_share": float(mono.mean()),
                     "median_largest_step": float(jump.median()), "share_step_ge_0.5": float(np.mean(jump >= 0.5))}

# ---- items that stay wrong: does the settled wrong answer depend on k? ----
st = mode[kind == "stays wrong"]
same = {f"{a} vs {b}": float((st[a] == st[b]).mean()) for a, b in [(0, 100), (10, 100), (25, 100), (50, 100), (25, 50)]}
print(f"\nstays-wrong items (n={len(st)}): share whose most common answer is identical at two filler lengths:",
      {k: round(v, 2) for k, v in same.items()})
out["stays_wrong_same_mode"] = same

# ---- all run items: how many move at intermediate k but not at k=100, i.e. non-monotone ----
peak_mid = rate[NEW_KS].max(axis=1)
print(f"\nitems whose best intermediate k beats k=100 by >= 0.5: {(peak_mid - rate[100] >= 0.5).sum()} of {len(rate)}")
out["mid_beats_100_by_half"] = int((peak_mid - rate[100] >= 0.5).sum())
json.dump(out, open(ROOT / "data/results/followup_dose_summary.json", "w"), indent=1, default=str)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
x = range(len(ALL_KS))
for _, r in fl.iterrows():
    axes[0].plot(x, r[ALL_KS], color="C0", alpha=0.25, lw=1)
axes[0].plot(x, fl[ALL_KS].mean(), color="k", lw=2.5, marker="o", label="mean")
axes[0].set_title(f"Items that flip up by k=100 (n={len(fl)}): one line per item", fontsize=10)
for kd, c in [("flips up", "C0"), ("other", "C7"), ("stays wrong", "C3")]:
    axes[1].plot(x, rate[kind == kd][ALL_KS].mean(), marker="o", color=c, label=f"{kd} (n={(kind == kd).sum()})")
    axes[2].plot(x, ndist[kind == kd][ALL_KS].mean(), marker="o", color=c, label=kd)
axes[1].set_title("Mean pass rate by item kind", fontsize=10)
axes[2].set_title("Mean distinct answers in 16 samples", fontsize=10)
for ax in axes:
    ax.set_xticks(list(x))
    ax.set_xticklabels(ALL_KS)
    ax.set_xlabel("dots k")
    ax.legend(fontsize=8)
axes[0].set_ylabel("pass rate (T=1, 16 samples)")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_dose.png", dpi=150)
print(f"\ntotal spend ${api.total_spend():.4f}")
