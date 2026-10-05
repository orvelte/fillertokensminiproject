"""Follow-up: does filler flip the same items on a second model?

Run: python scripts/followup_crossmodel.py
Second model: DeepSeek V4.1 Flash via Fireworks (thinking disabled), same 600 items, same
10-shot prompt, greedy, k=0 vs dots k=100. Compared with the Phase 2 V4 Flash results.
Writes data/results/followup_crossmodel.jsonl / .json and scripts/followup_crossmodel.png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import build_messages, parse_answer

CONDS = ["dots_0", "dots_100"]
USD_PER_PROMPT_TOKEN = 0.22e-6
load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load("fewshot.jsonl"), load("eval.jsonl")

reqs, meta = [], []
for c in CONDS:
    kind, k = c.split("_")
    for it in items:
        reqs.append({"messages": build_messages(few_shot, it, kind, int(k))})
        meta.append((c, it))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
resps = api.chat_many(reqs, temperature=0.0, max_tokens=10, provider="fireworks", tag="followup_crossmodel", projected_usd=projected)

rows = []
with open(ROOT / "data/results/followup_crossmodel.jsonl", "w") as f:
    for (c, it), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        row = {"idx": it["idx"], "cond": c, "response": text, "parsed": ans, "answer": it["answer"],
               "correct": ans == it["answer"], "reasoning": api.has_reasoning(r)}
        rows.append(row)
        f.write(json.dumps({**row, "model": "fireworks/deepseek-v4p1-flash", "temperature": 0.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
new = pd.DataFrame(rows)
print(f"calls with reasoning: {int(new.reasoning.sum())} | parse rate: {new.parsed.notna().mean():.3f}")
b = new.pivot(index="idx", columns="cond", values="correct")            # V4.1 Flash
a = pd.DataFrame(json.loads(l) for l in open(ROOT / "data/results/phase2_full.jsonl")).pivot(index="idx", columns="cond", values="correct")  # V4 Flash
out = {"reasoning_calls": int(new.reasoning.sum()), "parse_rate": float(new.parsed.notna().mean())}


def uplift(c, name):
    w2r, r2w = int((~c["dots_0"] & c["dots_100"]).sum()), int((c["dots_0"] & ~c["dots_100"]).sum())
    p = binomtest(w2r, w2r + r2w, 0.5).pvalue
    print(f"{name}: no filler {c['dots_0'].mean():.1%} -> dots k=100 {c['dots_100'].mean():.1%} "
          f"({100 * (c['dots_100'].mean() - c['dots_0'].mean()):+.1f} pp) | wrong->right {w2r}, right->wrong {r2w}, McNemar p={p:.2g}")
    return {"k0": float(c["dots_0"].mean()), "dots_100": float(c["dots_100"].mean()), "wrong_to_right": w2r, "right_to_wrong": r2w, "mcnemar_p": float(p)}


print()
out["v4_flash"] = uplift(a, "V4 Flash   (OpenRouter)")
out["v41_flash"] = uplift(b, "V4.1 Flash (Fireworks) ")


def assoc(x, y, label):
    t = pd.crosstab(x, y).reindex(index=[False, True], columns=[False, True], fill_value=0)
    odds, p = fisher_exact(t.values)
    exp = x.sum() * y.sum() / len(x)
    print(f"{label}: n={len(x)} | V4 {int(x.sum())}, V4.1 {int(y.sum())}, both {int(t.loc[True, True])} "
          f"(expected if independent {exp:.1f}) | odds ratio {odds:.1f}, Fisher p={p:.2g}")
    return {"n": int(len(x)), "v4": int(x.sum()), "v41": int(y.sum()), "both": int(t.loc[True, True]),
            "expected_independent": float(exp), "odds_ratio": float(odds), "fisher_p": float(p)}


print("\nDo the two models fail the same items without filler?")
out["same_failures_k0"] = assoc(~a["dots_0"], ~b["dots_0"], "wrong at k=0")
print("\nDo the same items flip? (items both models get wrong without filler)")
both_wrong = ~a["dots_0"] & ~b["dots_0"]
out["same_fixes"] = assoc(a["dots_100"][both_wrong], b["dots_100"][both_wrong], "fixed by filler")
print("\nDo the same items break? (items both models get right without filler)")
both_right = a["dots_0"] & b["dots_0"]
out["same_breaks"] = assoc(~a["dots_100"][both_right], ~b["dots_100"][both_right], "broken by filler")

# finer check with V4 Flash's sampled pass rates on the first 300 items
per = pd.read_csv(ROOT / "data/results/phase3_M300_items.csv").set_index("idx").join(b)
low = per[(per.p0 <= 0.25) & ~per["dots_0"]]  # V4 rarely solves it without filler, and V4.1 got it wrong at k=0
g = low.groupby("dots_100").pf.agg(["mean", "size"])
print("\nItems V4 Flash rarely solves without filler and V4.1 gets wrong at k=0:")
print("mean V4 Flash pass rate WITH filler, split by whether filler fixes the item on V4.1:")
print(g.round(3).to_string())
out["v4_pf_by_v41_fixed"] = {str(k): {"mean_v4_pf": float(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()}
json.dump(out, open(ROOT / "data/results/followup_crossmodel.json", "w"), indent=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(2)
for off, key, color, lab in [(-0.2, "k0", "C7", "no filler"), (0.2, "dots_100", "C0", "dots k=100")]:
    v = [100 * out["v4_flash"][key], 100 * out["v41_flash"][key]]
    ax1.bar(x + off, v, 0.4, color=color, label=lab)
    for xi, vi in zip(x + off, v):
        ax1.text(xi, vi + 1, f"{vi:.1f}%", ha="center", fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(["V4 Flash", "V4.1 Flash"])
ax1.set_ylabel("accuracy (%), greedy, 600 items")
ax1.set_ylim(0, 100)
ax1.set_title("Uplift on each model", fontsize=10)
ax1.legend(fontsize=9)
keys = [("same_failures_k0", "wrong without\nfiller on both"), ("same_fixes", "fixed by filler\non both"), ("same_breaks", "broken by filler\non both")]
obs, exp = [out[k]["both"] for k, _ in keys], [out[k]["expected_independent"] for k, _ in keys]
xx = np.arange(3)
ax2.bar(xx - 0.2, obs, 0.4, color="C0", label="observed")
ax2.bar(xx + 0.2, exp, 0.4, color="C7", label="expected if independent")
for xi, o, e in zip(xx, obs, exp):
    ax2.text(xi - 0.2, o + 2, str(o), ha="center", fontsize=9)
    ax2.text(xi + 0.2, e + 2, f"{e:.0f}", ha="center", fontsize=9)
ax2.set_xticks(xx)
ax2.set_xticklabels([l for _, l in keys], fontsize=9)
ax2.set_ylabel("number of items")
ax2.set_title("Item overlap between the two models", fontsize=10)
ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_crossmodel.png", dpi=150)
print(f"\ntotal spend ${api.total_spend():.4f}")
