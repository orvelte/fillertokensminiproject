"""Follow-up: disrupt an 8-unit block of a 25-dot filler (no slack), early / middle / late.

Run: python scripts/followup_disrupt25.py
Design and pre-registered criteria: notes/followup.md ("Disruption with no slack").
Items: the 48 flip items; primary set = those with >= 50% pass rate at 25 dots in the earlier
dose-response run. T=1, 16 fresh samples per item per condition (sample indices 100-115).
Control: 17 intact dots (same number of clean positions as the disrupted conditions).
Writes data/results/followup_disrupt25.jsonl / .json and scripts/followup_disrupt25.png.
"""
import json
import random
import string
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import N_FEW_SHOT, SYSTEM, parse_answer, user_turn

K_SAMPLES, SAMPLE_OFFSET = 16, 100
# condition -> (total units, start, end of the replaced units), or None for no filler
CONDS = {"no filler": None, "intact 25": (25, 0, 0), "early": (25, 0, 8), "middle": (25, 9, 17), "late": (25, 17, 25),
         "17 dots": (17, 0, 0), "all replaced": (25, 0, 25)}
LOCS = ["early", "middle", "late"]
USD_PER_PROMPT_TOKEN = 0.14e-6
rng = np.random.default_rng(0)


def filler(span, seed):
    """Dots with units [start, end) replaced by random lowercase letters (seeded)."""
    if span is None:
        return ""
    n, start, end = span
    r = random.Random(seed)
    letters = [r.choice(string.ascii_lowercase) for _ in range(25)]  # same letters at every location
    return " ".join(letters[i] if start <= i < end else "." for i in range(n))


def messages(few_shot, item, span):
    msgs = [{"role": "system", "content": SYSTEM}]
    for j, fs in enumerate(few_shot[:N_FEW_SHOT]):
        msgs.append({"role": "user", "content": user_turn(fs, filler(span, 1000 + j))})
        msgs.append({"role": "assistant", "content": str(fs["answer"])})
    msgs.append({"role": "user", "content": user_turn(item, filler(span, item["idx"]))})
    return msgs


load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load("fewshot.jsonl"), {it["idx"]: it for it in load("eval.jsonl")}
kinds = pd.read_csv(ROOT / "data/results/followup_dose_items.csv")
flip_idx = sorted(kinds[kinds.kind == "flips up"].idx)
primary = sorted(kinds[(kinds.kind == "flips up") & (kinds["25"] >= 0.5)].idx)  # fixed before data collection

reqs, meta = [], []
for cond, span in CONDS.items():
    for i in flip_idx:
        msgs = messages(few_shot, items[i], span)
        for s in range(K_SAMPLES):
            reqs.append({"messages": msgs, "temperature": 1.0, "sample": SAMPLE_OFFSET + s})
            meta.append((cond, s, i))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(flip_idx)} items, {len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
resps = api.chat_many(reqs, max_tokens=10, tag="followup_disrupt25", projected_usd=projected)

rows = []
with open(ROOT / "data/results/followup_disrupt25.jsonl", "w") as f:
    for (cond, s, i), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        assert not api.has_reasoning(r)
        row = {"idx": i, "cond": cond, "sample": SAMPLE_OFFSET + s, "response": text, "parsed": ans,
               "answer": items[i]["answer"], "correct": ans == items[i]["answer"]}
        rows.append(row)
        f.write(json.dumps({**row, "temperature": 1.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
df = pd.DataFrame(rows)
df["ans"] = df.parsed.where(df.parsed.notna(), -10**9)
rate_all = df.pivot_table(index="idx", columns="cond", values="correct", aggfunc="mean")[list(CONDS)]
ndist_all = df.groupby(["idx", "cond"]).ans.nunique().unstack()[list(CONDS)]
rate_all.to_csv(ROOT / "data/results/followup_disrupt25_items.csv")
out = {"parse_rate": df.groupby("cond").parsed.apply(lambda s: float(s.notna().mean())).to_dict()}


def analyse(idx, label):
    rate, ndist = rate_all.loc[idx], ndist_all.loc[idx]
    draws = rng.integers(0, len(rate), size=(10_000, len(rate)))

    def ci(x):
        x = np.asarray(x, float)
        b = x[draws].mean(1)
        return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    res = {"n_items": len(rate), "conditions": {}, "gap_vs_17": {}, "pairwise": {}}
    print(f"\n===== {label}: {len(rate)} items =====")
    print(f"{'condition':14s} {'pass rate [95% CI]':26s} {'distinct answers':>17s}")
    for c in CONDS:
        m, lo, hi = ci(rate[c])
        res["conditions"][c] = {"pass_rate": m, "ci": [lo, hi], "distinct_answers": float(ndist[c].mean())}
        print(f"{c:14s} {m:.3f} [{lo:.3f}, {hi:.3f}]       {ndist[c].mean():17.2f}")
    b = ci(rate["intact 25"] - rate["no filler"])
    res["benefit"] = {"mean": b[0], "ci": list(b[1:]), "informative": bool(b[0] >= 0.25)}
    print(f"benefit (intact 25 - no filler): {b[0]:+.3f} [{b[1]:+.3f}, {b[2]:+.3f}] -> informative: {b[0] >= 0.25}")
    print("gap = 17 intact dots - disrupted (positive = the block hurts beyond losing 8 positions):")
    for c in LOCS + ["all replaced", "intact 25"]:
        m, lo, hi = ci(rate["17 dots"] - rate[c])
        res["gap_vs_17"][c] = {"gap": m, "ci": [lo, hi]}
        print(f"  {c:13s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("pairwise differences between locations (positive = first location hurts more):")
    for a, b_ in combinations(LOCS, 2):
        m, lo, hi = ci(rate[b_] - rate[a])
        res["pairwise"][f"{a} vs {b_}"] = {"diff": m, "ci": [lo, hi]}
        print(f"  {a} vs {b_}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    big = lambda p, key: abs(p[key]) >= 0.10 and (p["ci"][0] > 0 or p["ci"][1] < 0)
    small = lambda p, key: abs(p[key]) < 0.10 and p["ci"][0] <= 0 <= p["ci"][1]
    gaps, pw = [res["gap_vs_17"][c] for c in LOCS], list(res["pairwise"].values())
    hurts = lambda g: g["gap"] >= 0.10 and g["ci"][0] > 0
    if all(small(g, "gap") for g in gaps) and all(small(p, "diff") for p in pw):
        v = "redundant positions"
    elif any(hurts(g) for g in gaps) and any(big(p, "diff") for p in pw):
        v = "location-specific"
    elif all(hurts(g) for g in gaps) and not any(big(p, "diff") for p in pw):
        v = "general disruption"
    else:
        v = "no criterion met"
    res["preregistered_verdict"] = v if res["benefit"]["informative"] else "uninformative"
    res["items_losing_half_vs_17"] = {c: int(((rate["17 dots"] - rate[c]) >= 0.5).sum()) for c in LOCS + ["all replaced"]}
    print("pre-registered verdict:", res["preregistered_verdict"], "| items losing >= 50 points vs 17 dots:", res["items_losing_half_vs_17"])
    return res


out["primary"] = analyse(primary, "PRIMARY set (25 dots was enough in the earlier run)")
out["all_flip_items"] = analyse(flip_idx, "SECONDARY: all flip items")
json.dump(out, open(ROOT / "data/results/followup_disrupt25.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(9, 4.2))
names = list(CONDS)
cc = out["primary"]["conditions"]
m = [cc[c]["pass_rate"] * 100 for c in names]
err = [[m[i] - cc[c]["ci"][0] * 100 for i, c in enumerate(names)], [cc[c]["ci"][1] * 100 - m[i] for i, c in enumerate(names)]]
ax.bar(range(len(names)), m, yerr=err, capsize=4, color=["C7", "C0", "C1", "C1", "C1", "C2", "C3"])
for i, v in enumerate(m):
    ax.text(i, v + err[1][i] + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(["no filler", "25 dots\n(intact)", "letters at\nunits 1–8", "letters at\nunits 10–17", "letters at\nunits 18–25",
                    "17 dots\n(control)", "all 25\nletters"], fontsize=9)
ax.set_ylabel("pass rate (%), T=1, 16 samples per item")
ax.set_ylim(0, 110)
ax.set_title(f"Replacing an 8-unit block of a 25-dot filler with random letters "
             f"({out['primary']['n_items']} primary items; 95% CI)", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_disrupt25.png", dpi=150)
print(f"\ntotal spend ${api.total_spend():.4f}")
