"""Follow-up: each hop on its own, for the items whose first hop is a tripling.

Run: python scripts/followup_singlehop.py
Design and pre-registered criteria: notes/followup.md ("Single hops in isolation").
158 items with c1 = 3 among the Phase 3 items; first hop alone ("What is the number for <y>?")
and second hop alone (y given as a literal); no filler, T=1, 16 samples per item per hop.
Writes data/results/followup_singlehop.jsonl / .json / _items.csv and scripts/followup_singlehop.png.
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import N_FEW_SHOT, SYSTEM, parse_answer, user_turn

RES = ROOT / "data" / "results"
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
K = 16
USD_PER_PROMPT_TOKEN = 0.14e-6
rng = np.random.default_rng(0)
load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load("fewshot.jsonl"), load("eval.jsonl")[:300]
c1_of = lambda it: 3 if EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).group(1) == "three times" else 2
sel = [it for it in items if c1_of(it) == 3]


def hop1(it):
    """First hop alone: same definitions, ask for y."""
    return {"definitions": it["definitions"], "question": f"What is the number for {it['queried_term']}?", "answer": it["chain"]["y"]}


def hop2(it):
    """Second hop alone: y given as a literal, original question."""
    defs = [[n, it["chain"]["y"] if n == it["queried_term"] else d] for n, d in it["definitions"]]
    return {"definitions": defs, "question": it["question"], "answer": it["answer"]}


HOPS = {"first hop alone": hop1, "second hop alone": hop2}


def messages(make, it):
    msgs = [{"role": "system", "content": SYSTEM}]
    for fs in few_shot[:N_FEW_SHOT]:
        f = make(fs)
        msgs.append({"role": "user", "content": user_turn(f, "")})
        msgs.append({"role": "assistant", "content": str(f["answer"])})
    msgs.append({"role": "user", "content": user_turn(make(it), "")})
    return msgs


reqs, meta = [], []
for hop, make in HOPS.items():
    for it in sel:
        msgs = messages(make, it)
        for s in range(K):
            reqs.append({"messages": msgs, "temperature": 1.0, "sample": s})
            meta.append((hop, it, s))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(sel)} items, {len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
resps = api.chat_many(reqs, max_tokens=10, tag="followup_singlehop", projected_usd=projected)

rows = []
with open(RES / "followup_singlehop.jsonl", "w") as f:
    for (hop, it, s), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        assert not api.has_reasoning(r)
        truth = HOPS[hop](it)["answer"]
        row = {"idx": it["idx"], "hop": hop, "sample": s, "response": text, "parsed": ans, "answer": truth, "correct": ans == truth}
        rows.append(row)
        f.write(json.dumps({**row, "temperature": 1.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
df = pd.DataFrame(rows)
print("parse rate:", df.groupby("hop").parsed.apply(lambda s: round(float(s.notna().mean()), 3)).to_dict())

p = df.pivot_table(index="idx", columns="hop", values="correct", aggfunc="mean").rename(columns={"first hop alone": "p1", "second hop alone": "p2"})
p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
two = p3[p3.cond.isin(["k0_T1.0", "F_T1.0"])].pivot_table(index="idx", columns="cond", values="correct", aggfunc="mean").rename(columns={"k0_T1.0": "p0", "F_T1.0": "pf"})
d = p.join(two).assign(c2=[next(it["coefficient"] for it in sel if it["idx"] == i) for i in p.index])
d["prod"] = d.p1 * d.p2
d.to_csv(RES / "followup_singlehop_items.csv")
out = {"n_items": int(len(d))}


def ci(x, n=4000):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


print("\n=== mean pass rates ===")
out["means"] = {}
for name, sub in [("all c1 = 3 items", d), ("c2 = 2", d[d.c2 == 2]), ("c2 = 3", d[d.c2 == 3])]:
    m = {k: ci(sub[k]) for k in ["p1", "p2", "prod", "p0", "pf"]}
    out["means"][name] = {k: {"mean": v[0], "ci": [v[1], v[2]]} for k, v in m.items()}
    print(f"{name:17s} (n={len(sub):3d}): first hop alone {m['p1'][0]:.2f} | second hop alone {m['p2'][0]:.2f} | product {m['prod'][0]:.2f} "
          f"| two-hop no filler {m['p0'][0]:.2f} | two-hop with filler {m['pf'][0]:.2f}")
print("share of items with single-hop pass rate >= 75% / <= 25%: "
      f"first hop {(d.p1 >= .75).mean():.0%} / {(d.p1 <= .25).mean():.0%} | second hop {(d.p2 >= .75).mean():.0%} / {(d.p2 <= .25).mean():.0%}")

print("\n=== pre-registered criteria ===")
gap = ci(d["prod"] - d.p0)
out["composition_deficit"] = {"mean_prod_minus_p0": gap[0], "ci": list(gap[1:]), "supported": bool(gap[0] >= 0.25)}
print(f"composition deficit: mean(p1·p2) − mean(p0) = {gap[0]:+.2f} [{gap[1]:+.2f}, {gap[2]:+.2f}] -> {'supported' if gap[0] >= 0.25 else 'not supported'}")
rho = spearmanr(d.pf, d["prod"])
bs = [spearmanr(*d.iloc[rng.integers(0, len(d), len(d))][["pf", "prod"]].values.T)[0] for _ in range(2000)]
hi, lo = d[d["prod"] >= 0.75], d[d["prod"] <= 0.25]
ceil_ok = rho[0] >= 0.5 and hi.pf.mean() >= 0.60 and lo.pf.mean() <= 0.15
out["ceiling"] = {"spearman": float(rho[0]), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "n_high": int(len(hi)), "pf_when_prod_high": float(hi.pf.mean()) if len(hi) else None,
                  "n_low": int(len(lo)), "pf_when_prod_low": float(lo.pf.mean()) if len(lo) else None,
                  "verdict": "supported" if ceil_ok else "not supported" if rho[0] < 0.25 else "in between"}
print(f"single hops set the ceiling: Spearman(pf, p1·p2) = {rho[0]:+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] | "
      f"mean pf when p1·p2 >= 0.75: {hi.pf.mean():.2f} (n={len(hi)}) | when <= 0.25: {lo.pf.mean() if len(lo) else float('nan'):.2f} (n={len(lo)}) -> {out['ceiling']['verdict']}")
for other, lab in [("p1", "first hop alone"), ("p2", "second hop alone"), ("p0", "two-hop without filler")]:
    r = spearmanr(d.pf, d[other])[0]
    out["ceiling"][f"spearman_pf_vs_{other}"] = float(r)
    print(f"   for reference, Spearman(pf, {lab}) = {r:+.2f}")
stuck, flips = d[d.pf <= 0.25], d[(d.pf - d.p0) >= 0.5]
share_low = float((stuck["prod"] < 0.5).mean())
out["stuck"] = {"n": int(len(stuck)), "share_prod_below_half": share_low, "share_prod_high": float((stuck["prod"] >= 0.75).mean()),
                "mean_p1": float(stuck.p1.mean()), "mean_p2": float(stuck.p2.mean()),
                "weaker_hop_first": float((stuck.p1 < stuck.p2).mean()), "weaker_hop_second": float((stuck.p2 < stuck.p1).mean()),
                "verdict": "supported" if share_low >= 0.7 else "single hops do not explain stuck items" if (stuck["prod"] >= 0.75).mean() > 0.5 else "in between"}
print(f"stuck items (pf <= 0.25, n={len(stuck)}): p1·p2 < 0.5 in {share_low:.0%}, >= 0.75 in {(stuck['prod'] >= 0.75).mean():.0%} "
      f"| mean first hop {stuck.p1.mean():.2f}, second hop {stuck.p2.mean():.2f} | first hop weaker in {(stuck.p1 < stuck.p2).mean():.0%}, "
      f"second weaker in {(stuck.p2 < stuck.p1).mean():.0%} -> {out['stuck']['verdict']}")
out["flips"] = {"n": int(len(flips)), "mean_p1": float(flips.p1.mean()), "mean_p2": float(flips.p2.mean()), "mean_prod": float(flips["prod"].mean())}
print(f"flip items (rise >= 50 pts, n={len(flips)}): mean first hop {flips.p1.mean():.2f}, second hop {flips.p2.mean():.2f}, product {flips['prod'].mean():.2f}")

print("\n=== two-hop pass rates by single-hop product ===")
d["bin"] = pd.cut(d["prod"], [-0.01, 0.25, 0.5, 0.75, 1.0], labels=["<= 0.25", "0.25–0.5", "0.5–0.75", "> 0.75"])
g = d.groupby("bin", observed=True).agg(n=("p0", "size"), p1=("p1", "mean"), p2=("p2", "mean"), p0=("p0", "mean"), pf=("pf", "mean"),
                                        mostly_right_with_filler=("pf", lambda s: (s >= .75).mean()))
print(g.round(2).to_string())
out["by_product_bin"] = g.round(4).reset_index().astype({"bin": str}).to_dict("records")
json.dump(out, open(RES / "followup_singlehop.json", "w"), indent=1)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
jit = lambda v: v + rng.uniform(-0.015, 0.015, len(v))
for ax, col, lab in [(axes[0], "p0", "two-hop pass rate, no filler"), (axes[1], "pf", "two-hop pass rate, 100 dots")]:
    for c2, color in [(2, "C0"), (3, "C3")]:
        s = d[d.c2 == c2]
        ax.scatter(jit(s["prod"]), jit(s[col]), s=14, alpha=0.6, color=color, label=f"second hop ×{c2}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("first hop alone × second hop alone (pass rates)")
    ax.set_ylabel(lab)
    ax.legend(fontsize=8)
fig.suptitle("Items whose first hop is a tripling: two-hop pass rate vs what the single hops allow (dashed: equal)", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_singlehop.png", dpi=150)
print(f"\ntotal spend ${api.total_spend():.4f}")
