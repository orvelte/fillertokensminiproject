"""Follow-up analyses on existing data only (no API calls).

Run: python scripts/followup_existing_data.py
(1) Error-step analysis: classify every wrong answer in the Phase 3 M=300 samples by which
    step of the chain x -> c1*x -> y -> c2*y -> answer could have produced it, and compare
    no-filler vs filler.
(2) Cross-filler agreement: do dots k=100 and counting k=25 fix the same items?
Writes data/results/followup_existing_data.json.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[1]
COEF = {"twice": 2, "three times": 3, "four times": 4}
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
out = {}


def ap(c, op, k, v):
    return c * v + k if op == "plus" else c * v - k


def flip(op):
    return "minus" if op == "plus" else "plus"


def classify(a, it):
    """Name the first simple single-mistake story that reproduces wrong answer `a`."""
    if a == "unparsed":
        return "unparsed"
    w, ref, op1, k1 = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).groups()
    c1, k1, x, y = COEF[w], int(k1), it["chain"]["x"], it["chain"]["y"]
    c2, op2, k2 = it["coefficient"], it["operation"], it["constant"]
    others = [v for n, v in it["values"].items() if n not in (it["queried_term"],)]
    # question applied to the wrong variable (includes using x directly, i.e. skipping hop 1)
    if a in {ap(c2, op2, k2, v) for v in others}:
        return "binding: question applied to another variable"
    # hop-1 structural mistakes, final step done correctly
    y_alts = {ap(c1, flip(op1), k1, x), c1 * x} | {ap(c, op1, k1, x) for c in (2, 3) if c != c1}
    y_alts |= {ap(c1, op1, k1, v) for v in others if v != x}  # hop 1 read the wrong source variable
    if a in {ap(c2, op2, k2, v) for v in y_alts}:
        return "hop-1 structural (sign / coefficient / dropped constant / wrong source)"
    # final-step structural mistakes on the correct y
    if a in {ap(c2, flip(op2), k2, y), c2 * y, y, ap(1, op2, k2, y)} | {ap(c, op2, k2, y) for c in (2, 3) if c != c2}:
        return "final-step structural (sign / coefficient / dropped step)"
    # otherwise: could the final step have been done correctly on some wrong integer y'?
    rem = (a - k2) if op2 == "plus" else (a + k2)
    return "other, consistent with wrong y + correct final step" if rem % c2 == 0 else "other, final step itself wrong"


items = {it["idx"]: it for it in map(json.loads, open(ROOT / "data/tasks/eval.jsonl"))}
df = pd.DataFrame(json.loads(l) for l in open(ROOT / "data/results/phase3_M300.jsonl"))
df["ans"] = df.parsed.where(df.parsed.notna(), "unparsed")
df["ans"] = [int(a) if a != "unparsed" else a for a in df.ans]
per = pd.read_csv(ROOT / "data/results/phase3_M300_items.csv").set_index("idx")
p16 = df.pivot_table(index="idx", columns="cond", values="correct", aggfunc="mean")
d16 = p16["F_T1.0"] - p16["k0_T1.0"]  # all 16 no-filler samples; used only to describe items, not to test
kind = pd.Series(np.where(d16 >= 0.5, "flips up (>= +50 pts)",
                          np.where((p16["k0_T1.0"] <= 0.25) & (p16["F_T1.0"] <= 0.25), "stuck wrong (<= 25% both)", "other")),
                 index=p16.index)

# ---------------- (1) error-step analysis ----------------
wrong = df[~df.correct].copy()
wrong["etype"] = [classify(a, items[i]) for a, i in zip(wrong.ans, wrong.idx)]
wrong["diff"] = [a - items[i]["answer"] if a != "unparsed" else np.nan for a, i in zip(wrong.ans, wrong.idx)]
n_samples = df.groupby("cond").size()

print("=== (1a) error types, as % of ALL samples in each condition (300 items x 16) ===")
tab = (wrong.groupby(["etype", "cond"]).size().unstack(fill_value=0) / n_samples * 100)[["k0_T1.0", "k0_T0.4", "F_T1.0"]]
tab.loc["TOTAL wrong"] = tab.sum()
tab["filler - k0 (pts)"] = tab["F_T1.0"] - tab["k0_T1.0"]
tab["relative change"] = tab["F_T1.0"] / tab["k0_T1.0"] - 1
print(tab.round(2).to_string())
out["error_types_pct_of_samples"] = tab.round(4).to_dict()

print("\n=== (1b) residue test: of 'other' errors, share consistent with a correct final step ===")
print("chance level is 1/c2 if the wrong number were arbitrary")
oth = wrong[wrong.etype.str.startswith("other")].copy()
oth["c2"] = [items[i]["coefficient"] for i in oth.idx]
oth["valid"] = oth.etype.str.contains("consistent")
res = oth.groupby(["c2", "cond"]).valid.agg(["mean", "size"]).unstack("cond")
print(res.round(3).to_string())
out["residue_test"] = {f"c2={c}|{cond}": {"share_valid": float(g.valid.mean()), "n": int(len(g))}
                       for (c, cond), g in oth.groupby(["c2", "cond"])}

print("\n=== (1c) size of the error (answer - truth) among 'other' errors ===")
for cond in ["k0_T1.0", "F_T1.0"]:
    d = oth[oth.cond == cond]["diff"].astype(int)
    print(f"{cond}: n={len(d)} | multiple of 100: {np.mean(d % 100 == 0):.2f} | multiple of 10: {np.mean(d % 10 == 0):.2f}"
          f" | |diff|<=10: {np.mean(d.abs() <= 10):.2f} | median |diff|: {d.abs().median():.0f}"
          f" | top diffs: {Counter(d).most_common(8)}")
    out[f"diff_{cond}"] = {"n": int(len(d)), "mult100": float(np.mean(d % 100 == 0)), "mult10": float(np.mean(d % 10 == 0)),
                           "within10": float(np.mean(d.abs() <= 10)), "top": Counter(int(v) for v in d).most_common(8)}

print("\n=== (1d) no-filler error mix: items that flip up with filler vs items that stay wrong ===")
k0w = wrong[wrong.cond == "k0_T1.0"].assign(kind=lambda w: w.idx.map(kind))
mix = k0w[k0w.kind != "other"].groupby(["etype", "kind"]).size().unstack(fill_value=0)
mix = mix / mix.sum() * 100
print("items:", kind.value_counts().to_dict())
print(mix.round(1).to_string())
out["k0_error_mix_by_item_kind_pct"] = mix.round(3).to_dict()
out["item_kinds"] = kind.value_counts().to_dict()

print("\n=== (1e) how concentrated are no-filler answers? (16 samples per item, T=1) ===")
rows = []
for idx, d in df[df.cond.isin(["k0_T1.0", "F_T1.0"])].groupby("idx"):
    for cond, g in d.groupby("cond"):
        c = Counter(g.ans)
        truth = items[idx]["answer"]
        wrong_c = Counter({a: n for a, n in c.items() if a != truth})
        rows.append({"idx": idx, "cond": cond, "n_distinct": len(c),
                     "top_wrong_share": max(wrong_c.values()) / 16 if wrong_c else 0.0,
                     "top_wrong": max(wrong_c, key=wrong_c.get) if wrong_c else None})
conc = pd.DataFrame(rows).assign(kind=lambda r: r.idx.map(kind))
print(conc.groupby(["kind", "cond"])[["n_distinct", "top_wrong_share"]].mean().round(2).to_string())
piv = conc.pivot(index="idx", columns="cond", values="top_wrong")
stuck = kind[kind.str.startswith("stuck")].index
same = (piv.loc[stuck, "k0_T1.0"] == piv.loc[stuck, "F_T1.0"]).mean()
print(f"stuck-wrong items whose most common wrong answer is the same with and without filler: {same:.0%} (n={len(stuck)})")
out["stuck_same_top_wrong_answer"] = float(same)
out["concentration"] = {f"{k}|{c}": v for (k, c), v in conc.groupby(["kind", "cond"])[["n_distinct", "top_wrong_share"]].mean().round(3).to_dict("index").items()}

# ---------------- (2) cross-filler agreement ----------------
print("\n=== (2a) Phase 2 greedy, N=600: do dots-100 and counting-25 fix the same items? ===")
p2 = pd.DataFrame(json.loads(l) for l in open(ROOT / "data/results/phase2_full.jsonl"))
c = p2.pivot(index="idx", columns="cond", values="correct")
for name, base in [("wrong at k=0", ~c["dots_0"]), ("right at k=0", c["dots_0"])]:
    s = c[base]
    fixed_d, fixed_c = s["dots_100"] != s["dots_0"], s["counting_25"] != s["dots_0"]
    t = pd.crosstab(fixed_d, fixed_c).reindex(index=[False, True], columns=[False, True], fill_value=0)
    odds, p = fisher_exact(t.values)
    exp = fixed_d.sum() * fixed_c.sum() / len(s)
    print(f"items {name}: n={len(s)} | changed by dots: {fixed_d.sum()} | by counting: {fixed_c.sum()} | by both: {t.loc[True, True]}"
          f" (expected if independent: {exp:.1f}) | odds ratio {odds:.1f}, Fisher p={p:.2g}")
    out[f"cross_filler_{name}"] = {"n": int(len(s)), "dots": int(fixed_d.sum()), "counting": int(fixed_c.sum()),
                                   "both": int(t.loc[True, True]), "expected_independent": float(exp),
                                   "odds_ratio": float(odds), "fisher_p": float(p)}

print("\n=== (2b) independent check on the first 300 items: counting-25 greedy vs sampled dots-100 pass rate ===")
m = per.join(c["counting_25"].rename("counting_greedy")).join(c["dots_0"].rename("k0_greedy"))
low = m[m.p0 <= 0.25]  # items the model rarely gets right without filler (half-B estimate)
g = low.groupby("counting_greedy").pf.agg(["mean", "size"])
print("items with no-filler pass rate <= 25%: mean dots-100 pass rate by counting-25 greedy outcome")
print(g.round(3).to_string())
print("same split by the k=0 greedy outcome (noise reference):")
print(low.groupby("k0_greedy").pf.agg(["mean", "size"]).round(3).to_string())
out["low_p0_pf_by_counting_greedy"] = {str(k): {"mean_pf": float(v["mean"]), "n": int(v["size"])} for k, v in g.iterrows()}

json.dump(out, open(ROOT / "data/results/followup_existing_data.json", "w"), indent=1, default=str)
