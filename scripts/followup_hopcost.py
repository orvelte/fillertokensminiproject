"""Follow-up: does the cost of the first hop, measured on its own, predict whether filler helps?

Run: python scripts/followup_hopcost.py
Design and pre-registered criteria: notes/followup.md ("Hop-cost experiment").
1. one-hop items (c*x +/- k of a literal), c in {2,3,4,5,10,11}, no filler  -> cost per multiplier
2. two-hop items, c1 in {2,3,4,5,10,11} x c2 in {2,3}, no filler vs 100 dots
3. symbolic wording ("3·zab + 14"), c1 in {2,3,4,10} x c2 in {2,3}, plus one-hop symbolic
Greedy, 10-shot, V4 Flash. Writes data/tasks/hopcost_items.jsonl, data/results/followup_hopcost.jsonl /
.json and scripts/followup_hopcost.png.
"""
import json
import random
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
sys.path.insert(0, str(ROOT / "vendor/filler-token-reasoning/scripts"))
import api
from data.generate_varbind_dataset import load_real_words, make_fake_term
from prompts import SYSTEM, make_filler, parse_answer, user_turn

SEED = 20261004
WORD = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 10: "ten times", 11: "eleven times"}
MULTS, MULTS_SYM, C2S = [2, 3, 4, 5, 10, 11], [2, 3, 4, 10], [2, 3]
N_FEW = 10
SYSTEM_SYM = SYSTEM.replace("(for example 'twice the number for X plus 3')", "(for example '2·X + 3')")
USD_PER_PROMPT_TOKEN = 0.14e-6
rng = random.Random(SEED)
words = load_real_words()
assert words


def expr(c, ref, op, k, wording):
    return f"{WORD[c]} the number for {ref} {op} {k}" if wording == "words" else f"{c}·{ref} {'+' if op == 'plus' else '-'} {k}"


def derive(c, v):
    """Pick op and constant so that c*v +/- k is non-negative."""
    while True:
        op, k = rng.choice(["plus", "minus"]), rng.randint(1, 50)
        val = c * v + k if op == "plus" else c * v - k
        if val >= 0:
            return op, k, val


def make_item(hops, c1, c2, mults, wording):
    """5 variables: x literal, (two-hop: y derived from x), distractors; random topological order."""
    names, terms = set(), []
    new = lambda: (lambda t: (names.add(t), t)[1])(make_fake_term(words, rng, names))
    x = {"name": new(), "value": rng.randint(10, 99), "def": None, "dep": None}
    x["def"] = x["value"]
    terms.append(x)
    if hops == 2:
        op1, k1, yv = derive(c1, x["value"])
        y = {"name": new(), "value": yv, "def": expr(c1, x["name"], op1, k1, wording), "dep": x["name"]}
        terms.append(y)
    while len(terms) < 5:
        if rng.random() < 0.4:
            v = rng.randint(10, 99)
            terms.append({"name": new(), "value": v, "def": v, "dep": None})
        else:
            ref, c = rng.choice(terms), rng.choice(mults)
            op, k, v = derive(c, ref["value"])
            terms.append({"name": new(), "value": v, "def": expr(c, ref["name"], op, k, wording), "dep": ref["name"]})
    target = terms[1] if hops == 2 else x
    cq = c2 if hops == 2 else c1
    opq, kq, ans = derive(cq, target["value"])
    q = f"What is {expr(cq, target['name'], opq, kq, wording)}?"
    order, done = [], set()
    while len(order) < len(terms):  # random order with every reference defined first
        t = rng.choice([t for t in terms if t["name"] not in done and (t["dep"] is None or t["dep"] in done)])
        order.append(t)
        done.add(t["name"])
    return {"definitions": [[t["name"], t["def"]] for t in order], "question": q, "answer": ans, "hops": hops,
            "c1": c1, "c2": c2 if hops == 2 else None, "wording": wording, "x": x["value"], "y": terms[1]["value"] if hops == 2 else None}


# variant -> (hops, multipliers, wording, items per cell)
VARIANTS = {"one-hop, words": (1, MULTS, "words", 100), "two-hop, words": (2, MULTS, "words", 100),
            "one-hop, symbolic": (1, MULTS_SYM, "symbolic", 50), "two-hop, symbolic": (2, MULTS_SYM, "symbolic", 50)}
items, few = [], {}
for vname, (hops, mults, wording, n) in VARIANTS.items():
    few[vname] = [make_item(hops, rng.choice(mults), rng.choice(C2S), mults, wording) for _ in range(N_FEW)]
    for c1 in mults:
        for c2 in (C2S if hops == 2 else [None]):
            for _ in range(n):
                items.append({"id": len(items), "variant": vname, **make_item(hops, c1, c2, mults, wording)})
with open(ROOT / "data/tasks/hopcost_items.jsonl", "w") as f:
    f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
json.dump(few, open(ROOT / "data/tasks/hopcost_fewshot.json", "w"), ensure_ascii=False, indent=1)


def messages(it, k):
    filler = make_filler("dots", k)
    msgs = [{"role": "system", "content": SYSTEM if it["wording"] == "words" else SYSTEM_SYM}]
    for fs in few[it["variant"]]:
        msgs.append({"role": "user", "content": user_turn(fs, filler)})
        msgs.append({"role": "assistant", "content": str(fs["answer"])})
    msgs.append({"role": "user", "content": user_turn(it, filler)})
    return msgs


reqs, meta = [], []
for it in items:
    for k in ([0, 100] if it["hops"] == 2 else [0]):
        reqs.append({"messages": messages(it, k)})
        meta.append((it, k))
projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
print(f"{len(items)} items, {len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
resps = api.chat_many(reqs, temperature=0.0, max_tokens=10, tag="followup_hopcost", projected_usd=projected)

rows = []
with open(ROOT / "data/results/followup_hopcost.jsonl", "w") as f:
    for (it, k), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        assert not api.has_reasoning(r)
        row = {"id": it["id"], "variant": it["variant"], "c1": it["c1"], "c2": it["c2"], "k": k, "response": text, "parsed": ans,
               "answer": it["answer"], "correct": ans == it["answer"]}
        rows.append(row)
        f.write(json.dumps({**row, "temperature": 0.0, "max_tokens": 10, "usage": r["usage"]}, ensure_ascii=False) + "\n")
df = pd.DataFrame(rows)
out = {"parse_rate": df.groupby("variant").parsed.apply(lambda s: float(s.notna().mean())).to_dict()}
print("parse rate by variant:", {k: round(v, 3) for k, v in out["parse_rate"].items()})
se = lambda p, n: (p * (1 - p) / n) ** 0.5

# ---- 1. one-hop cost ----
print("\n=== 1. one-hop accuracy without filler (the cost measure) ===")
one = {}
for wording in ["words", "symbolic"]:
    d = df[df.variant == f"one-hop, {wording}"]
    one[wording] = d.groupby("c1").correct.mean()
    print(f"{wording}: " + " | ".join(f"x{c}: {a:.0%}" for c, a in one[wording].items()) + f"   (n = {int(d.groupby('c1').size().iloc[0])} each)")
out["one_hop_accuracy"] = {w: {int(c): float(a) for c, a in s.items()} for w, s in one.items()}

# ---- 2. two-hop by cell ----
out["two_hop"], out["criteria"] = {}, {}
fixes = {}
for wording in ["words", "symbolic"]:
    d = df[df.variant == f"two-hop, {wording}"]
    acc = d.pivot_table(index=["c1", "c2"], columns="k", values="correct", aggfunc="mean")
    n = int(d[d.k == 0].groupby(["c1", "c2"]).size().iloc[0])
    acc["fixability"] = 1 - (1 - acc[100]) / (1 - acc[0]).clip(lower=1e-9)
    acc["one-hop acc"] = [one[wording][c1] for c1, _ in acc.index]
    fixes[wording] = acc
    print(f"\n=== 2. two-hop, {wording} wording ({n} items per cell): accuracy without filler -> with 100 dots ===")
    for (c1, c2), r in acc.iterrows():
        print(f"  c1 = {c1:2d}, c2 = {c2}: {r[0]:.0%} -> {r[100]:.0%} ({100 * (r[100] - r[0]):+.0f} pts) | share of errors removed {r['fixability']:+.0%} | one-hop x{c1}: {r['one-hop acc']:.0%}")
    out["two_hop"][wording] = {f"c1={c1},c2={c2}": {"k0": float(r[0]), "k100": float(r[100]), "fixability": float(r["fixability"]), "n": n} for (c1, c2), r in acc.iterrows()}
    by_c1 = d.pivot_table(index="c1", columns="k", values="correct", aggfunc="mean")
    print("  averaged over c2: " + " | ".join(f"x{c}: {r[0]:.0%} -> {r[100]:.0%}" for c, r in by_c1.iterrows()))
    out["two_hop"][wording]["by_c1"] = {int(c): {"k0": float(r[0]), "k100": float(r[100])} for c, r in by_c1.iterrows()}

# ---- criteria ----
acc = fixes["words"]
print("\n=== pre-registered criteria ===")
for c2 in C2S:
    s = acc.xs(c2, level="c2")
    r_fix = spearmanr(1 - s["one-hop acc"], s["fixability"])[0]
    r_base = spearmanr(1 - s["one-hop acc"], 1 - s[0])[0]
    out["criteria"][f"c2={c2}"] = {"spearman_cost_vs_fixability": float(r_fix), "spearman_cost_vs_baseline_error": float(r_base)}
    print(f"c2 = {c2}: Spearman(one-hop error, fixability) = {r_fix:+.2f} | Spearman(one-hop error, two-hop no-filler error) = {r_base:+.2f}")
rf = [out["criteria"][f"c2={c}"]["spearman_cost_vs_fixability"] for c in C2S]
rb = [out["criteria"][f"c2={c}"]["spearman_cost_vs_baseline_error"] for c in C2S]
out["criteria"]["cost_predicts_fixability"] = "supported" if max(rf) <= -0.8 else "not supported" if max(rf) > -0.5 else "in between"
out["criteria"]["cost_predicts_baseline"] = "supported" if min(rb) >= 0.8 else "not supported"
b = lambda w, c: out["two_hop"][w]["by_c1"][c]["k0"]
gap_words, gap_sym = b("words", 2) - b("words", 3), b("symbolic", 2) - b("symbolic", 3)
out["criteria"]["wording"] = {"gap_words_pts": 100 * gap_words, "gap_symbolic_pts": 100 * gap_sym,
                              "verdict": "lexical" if abs(gap_sym) <= 0.15 else "arithmetic" if gap_sym >= 0.30 else "in between"}
d4, d10 = b("words", 2) - b("words", 4), b("words", 2) - b("words", 10)
out["criteria"]["parity"] = {"x2_minus_x4_pts": 100 * d4, "x2_minus_x10_pts": 100 * d10,
                             "verdict": "parity" if abs(d4) <= 0.15 and abs(d10) <= 0.15 else "cost" if d4 >= 0.15 and d10 <= 0 else "neither criterion met"}
print("cost predicts fixability:", out["criteria"]["cost_predicts_fixability"], "| cost predicts baseline:", out["criteria"]["cost_predicts_baseline"])
print(f"wording: twice-vs-three-times gap {100 * gap_words:+.0f} pts in words, {100 * gap_sym:+.0f} pts symbolic -> {out['criteria']['wording']['verdict']}")
print(f"parity: x2 minus x4 = {100 * d4:+.0f} pts, x2 minus x10 = {100 * d10:+.0f} pts (no filler, words) -> {out['criteria']['parity']['verdict']}")
json.dump(out, open(ROOT / "data/results/followup_hopcost.json", "w"), indent=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x = np.arange(len(MULTS))
axes[0].bar(x, [100 * one["words"][c] for c in MULTS], color="C7")
axes[0].set_xticks(x)
axes[0].set_xticklabels([f"×{c}" for c in MULTS])
axes[0].set_ylabel("accuracy (%)")
axes[0].set_title("One-hop accuracy, no filler (cost of each multiplier)", fontsize=9)
for ax, c2 in zip(axes[1:], C2S):
    s = acc.xs(c2, level="c2")
    ax.bar(x - 0.2, 100 * s[0], 0.4, color="C7", label="no filler")
    ax.bar(x + 0.2, 100 * s[100], 0.4, color="C0", label="100 dots")
    ax.set_xticks(x)
    ax.set_xticklabels([f"×{c}" for c in MULTS])
    ax.set_xlabel("first-hop multiplier")
    ax.set_title(f"Two-hop accuracy, second hop ×{c2}", fontsize=9)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_hopcost.png", dpi=150)
print(f"\ntotal spend ${api.total_spend():.4f}")
