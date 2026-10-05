"""Figure: filler uplift on every open-weight model tested (existing results; no API calls).

Run: python scripts/fig_open_models.py
Same 10-shot prompt, greedy, thinking off, no filler vs 100 dots.
Main task: V4 Flash and V4.1 Flash on 600 items; the six lens-equipped models on the first 150.
Easier task sets were run only for the models listed in each panel.
Writes scripts/fig_open_model_uplift.png and data/results/open_model_uplift.csv.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
NAMES = {"deepseek/deepseek-v4-flash": "DeepSeek\nV4 Flash", "deepseek-v4p1-flash": "DeepSeek\nV4.1 Flash", "qwen/qwen3.5-122b-a10b": "Qwen3.5\n122B-A10B",
         "qwen/qwen3.6-35b-a3b": "Qwen3.6\n35B-A3B", "qwen/qwen3.6-27b": "Qwen3.6\n27B", "qwen/qwen3.5-27b": "Qwen3.5\n27B",
         "google/gemma-3-27b-it": "Gemma 3\n27B", "qwen/qwen3.5-9b": "Qwen3.5\n9B"}


def load(path, model=None):
    d = pd.DataFrame(json.loads(l) for l in open(path))
    if model:
        d["model"] = model
    return d[d.cond.isin(["dots_0", "dots_100"])][["model", "idx", "cond", "correct"]].drop_duplicates(["model", "idx", "cond"], keep="last")


sets = {"Main task": pd.concat([load(RES / "phase2_full.jsonl", "deepseek/deepseek-v4-flash"), load(RES / "followup_crossmodel.jsonl", "deepseek-v4p1-flash"),
                                load(RES / "whitebox_screen.jsonl")]),
        "Easier task (multiplier 2, constants 1–30)": load(RES / "whitebox_screen_easy.jsonl"),
        "Easiest task tried (multiplier 2, constants 1–9)": load(RES / "whitebox_screen_veasy.jsonl")}
rows = []
for task, d in sets.items():
    for model in [m for m in NAMES if m in set(d.model)]:
        c = d[d.model == model].pivot(index="idx", columns="cond", values="correct").dropna()
        up, down = int((~c.dots_0 & c.dots_100).sum()), int((c.dots_0 & ~c.dots_100).sum())
        rows.append({"task": task, "model": model, "n": len(c), "no_filler": c.dots_0.mean(), "dots_100": c.dots_100.mean(),
                     "gain_pp": 100 * (c.dots_100.mean() - c.dots_0.mean()), "wrong_to_right": up, "right_to_wrong": down,
                     "mcnemar_p": binomtest(up, up + down, 0.5).pvalue if up + down else float("nan")})
t = pd.DataFrame(rows)
t.to_csv(RES / "open_model_uplift.csv", index=False)
print(t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

widths = [len(t[t.task == k]) for k in sets]
fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": widths}, sharey=True)
se = lambda p, n: 100 * np.sqrt(p * (1 - p) / n)
for ax, task in zip(axes, sets):
    s = t[t.task == task].reset_index(drop=True)
    x = np.arange(len(s))
    ax.bar(x - 0.2, 100 * s.no_filler, 0.4, yerr=se(s.no_filler, s.n), capsize=3, color="C7", label="no filler")
    ax.bar(x + 0.2, 100 * s.dots_100, 0.4, yerr=se(s.dots_100, s.n), capsize=3, color="C0", label="100 dots")
    for i, r in s.iterrows():
        top = 100 * max(r.no_filler, r.dots_100) + max(se(r.no_filler, r.n), se(r.dots_100, r.n)) + 2
        p = "" if np.isnan(r.mcnemar_p) else f"\np = {r.mcnemar_p:.0e}" if r.mcnemar_p < 0.001 else f"\np = {r.mcnemar_p:.2f}"
        ax.text(i, top, f"{r.gain_pp:+.0f} pp{p}", ha="center", fontsize=8.5, fontweight="bold" if r.mcnemar_p < 0.01 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{NAMES[m]}\nn = {n}" for m, n in zip(s.model, s.n)], fontsize=8.5)
    ax.set_title(task, fontsize=10)
    ax.set_ylim(0, 85)
axes[0].set_ylabel("accuracy (%), greedy (±1 SE)")
axes[0].legend(fontsize=9, loc="upper right")
fig.suptitle("Filler uplift on the open-weight models tested (same 10-shot prompt, no reasoning tokens; label: change with filler, McNemar p)", fontsize=10.5)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "fig_open_model_uplift.png", dpi=150)
