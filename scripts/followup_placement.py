"""Follow-up: does it matter where the filler sits?

Run: python scripts/followup_placement.py
Design and pre-registered criteria: notes/followup.md ("Filler placement").
Greedy, frozen 600 items, 100 dots. New placements: before everything; between the definitions
and the question. No filler and the standard after-question placement come from Phase 2.
Writes data/results/followup_placement.jsonl / .json and scripts/followup_placement.png.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import N_FEW_SHOT, SYSTEM, make_filler, parse_answer

RES = ROOT / "data" / "results"
DOTS = make_filler("dots", 100)
USD_PER_PROMPT_TOKEN = 0.14e-6
load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load("fewshot.jsonl"), load("eval.jsonl")
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
c1_of = {it["idx"]: 3 if EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).group(1) == "three times" else 2 for it in items}


def turn(it, place):
    defs = "\n".join(f"{n} = {v}" for n, v in it["definitions"])
    tail = f"Question: {it['question']}\n\nFiller:\n\nAnswer:"  # same ending as the no-filler baseline
    if place == "before everything":
        return f"Filler: {DOTS}\n\n{defs}\n{tail}"
    if place == "between definitions and question":
        return f"{defs}\n\nFiller: {DOTS}\n\n{tail}"
    raise ValueError(place)


NEW = ["before everything", "between definitions and question"]
reqs, meta = [], []
for place in NEW:
    for it in items:
        msgs = [{"role": "system", "content": SYSTEM}]
        for fs in few_shot[:N_FEW_SHOT]:
            msgs += [{"role": "user", "content": turn(fs, place)}, {"role": "assistant", "content": str(fs["answer"])}]
        msgs.append({"role": "user", "content": turn(it, place)})
        reqs.append({"messages": msgs})
        meta.append((place, it))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
resps = api.chat_many(reqs, temperature=0.0, max_tokens=10, tag="followup_placement", projected_usd=projected)

rows = []
with open(RES / "followup_placement.jsonl", "w") as f:
    for (place, it), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        assert not api.has_reasoning(r)
        row = {"idx": it["idx"], "cond": place, "response": text, "parsed": ans, "answer": it["answer"], "correct": ans == it["answer"]}
        rows.append(row)
        f.write(json.dumps({**row, "temperature": 0.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
p2 = pd.DataFrame(json.loads(l) for l in open(RES / "phase2_full.jsonl"))
p2 = p2[p2.cond.isin(["dots_0", "dots_100"])].assign(cond=lambda d: d.cond.map({"dots_0": "no filler", "dots_100": "after the question"}))
df = pd.concat([pd.DataFrame(rows), p2[["idx", "cond", "response", "parsed", "answer", "correct"]]])
c = df.pivot(index="idx", columns="cond", values="correct")
CONDS = ["no filler", "before everything", "between definitions and question", "after the question"]
out = {"parse_rate": df.groupby("cond").parsed.apply(lambda s: float(s.notna().mean())).to_dict()}


def compare(a, b, sub=None):
    cc = c if sub is None else c.loc[sub]
    up, down = int((~cc[a] & cc[b]).sum()), int((cc[a] & ~cc[b]).sum())
    return {"gain_pp": float(100 * (cc[b].mean() - cc[a].mean())), "up": up, "down": down,
            "p": float(binomtest(up, up + down, 0.5).pvalue) if up + down else float("nan")}


print(f"\n{'placement':34s} {'accuracy':>9s} {'gain vs no filler':>18s} {'wrong->right / right->wrong':>28s} {'p':>9s} {'p vs after-question':>20s}")
out["all"] = {}
for cond in CONDS:
    r = compare("no filler", cond) if cond != "no filler" else {"gain_pp": 0.0, "up": 0, "down": 0, "p": float("nan")}
    r["acc"] = float(c[cond].mean())
    r["p_vs_standard"] = compare(cond, "after the question")["p"] if cond not in ("no filler", "after the question") else float("nan")
    out["all"][cond] = r
    print(f"{cond:34s} {r['acc']:9.1%} {r['gain_pp']:+17.1f} {r['up']:>14d} / {r['down']:<11d} {r['p']:9.2g} {r['p_vs_standard']:20.2g}")

print("\nby first-hop multiplier:")
out["by_c1"] = {}
for c1 in (2, 3):
    sub = [i for i in c.index if c1_of[i] == c1]
    out["by_c1"][c1] = {cond: {"acc": float(c.loc[sub, cond].mean()), **(compare("no filler", cond, sub) if cond != "no filler" else {})} for cond in CONDS}
    print(f"  first hop x{c1} ({len(sub)} items): " + " | ".join(f"{cond}: {c.loc[sub, cond].mean():.1%}" for cond in CONDS))
    print("     p vs no filler: " + " | ".join(f"{cond}: {out['by_c1'][c1][cond]['p']:.2g}" for cond in CONDS[1:]))

std = out["all"]["after the question"]["gain_pp"]
bef, bet = out["all"]["before everything"], out["all"]["between definitions and question"]
big = lambda r: r["gain_pp"] >= std / 2 and r["p"] < 0.01
small = lambda r: r["gain_pp"] < 3 and not r["p"] < 0.05
verdict = ("not about computing on the item" if big(bef) else "room after the definitions is enough" if big(bet)
           else "room must come after the question" if small(bef) and small(bet) else "no criterion met")
out["preregistered_verdict"] = verdict
print(f"\nstandard gain {std:+.1f} pts | pre-registered verdict: {verdict}")
json.dump(out, open(RES / "followup_placement.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(8, 4.2))
acc = [100 * c[cond].mean() for cond in CONDS]
se = [100 * (a / 100 * (1 - a / 100) / len(c)) ** 0.5 for a in acc]
ax.bar(range(4), acc, yerr=se, capsize=4, color=["C7", "C1", "C1", "C0"])
for i, a in enumerate(acc):
    ax.text(i, a + se[i] + 1, f"{a:.1f}%", ha="center", fontsize=9)
ax.set_xticks(range(4))
ax.set_xticklabels(["no filler", "100 dots before\neverything", "100 dots between\ndefinitions and question", "100 dots after\nthe question"], fontsize=9)
ax.set_ylabel("accuracy (%), greedy, 600 items (±1 SE)")
ax.set_ylim(0, 80)
ax.set_title("Where the filler sits", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_placement.png", dpi=150)
print(f"total spend ${api.total_spend():.4f}")
