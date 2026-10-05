"""Follow-up: (1) does an item's tipping point track arithmetic difficulty? (2) do items that stay
wrong give the same wrong answer every time?  Existing data only, no API calls.

Run: python scripts/followup_tipping.py
Tipping point = smallest filler length k in (0, 10, 25, 50, 100) at which the item's pass rate
(16 samples, T=1) is >= 75%; "never" if none. The 81 items not in the dose-response run were at
>= 15/16 at both k=0 and k=100 and count as tipping at 0.
Chain: x -> c1*x -> y = c1*x +/- k1 -> c2*y -> z = c2*y +/- k2.
Writes data/results/followup_tipping.json and scripts/followup_tipping.png.
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
COEF = {"twice": 2, "three times": 3}
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
KS = [0, 10, 25, 50, 100]
out = {}


def mul_carries(a, c):
    n = carry = 0
    while a:
        carry = (a % 10 * c + carry) // 10
        n += carry > 0
        a //= 10
    return n


def addsub_carries(a, b, op):
    n = carry = 0
    while a or b:
        da, db = a % 10, b % 10
        carry = int(da + db + carry >= 10) if op == "plus" else int(da - db - carry < 0)
        n += carry
        a, b = a // 10, b // 10
    return n


items = {it["idx"]: it for it in map(json.loads, open(ROOT / "data/tasks/eval.jsonl")) if it["idx"] < 300}
feat = []
for i, it in items.items():
    w, ref, op1, k1 = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).groups()
    c1, k1, ch, c2, op2, k2 = COEF[w], int(k1), it["chain"], it["coefficient"], it["operation"], it["constant"]
    carries = [mul_carries(ch["x"], c1), addsub_carries(ch["c1x"], k1, op1), mul_carries(ch["y"], c2), addsub_carries(ch["c2y"], k2, op2)]
    feat.append({"idx": i, "carries total": sum(carries), "carries, first hop": carries[0] + carries[1], "carries, second hop": carries[2] + carries[3],
                 "digits in y": len(str(ch["y"])), "digits in answer": len(str(ch["answer"])), "first product c1·x": ch["c1x"],
                 "second product c2·y": ch["c2y"], "c1 = 3": int(c1 == 3), "c2 = 3": int(c2 == 3),
                 "subtractions": int(op1 == "minus") + int(op2 == "minus"), "x": ch["x"], "y": ch["y"]})
feat = pd.DataFrame(feat).set_index("idx")
FEATURES = [c for c in feat.columns]

p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
p3 = p3[p3.cond.isin(["k0_T1.0", "F_T1.0"])].assign(k=lambda d: np.where(d.cond == "k0_T1.0", 0, 100))
dose = pd.DataFrame(json.loads(l) for l in open(RES / "followup_dose.jsonl"))
allk = pd.concat([p3[["idx", "k", "sample", "parsed", "answer", "correct"]], dose[["idx", "k", "sample", "parsed", "answer", "correct"]]])
rate = allk.pivot_table(index="idx", columns="k", values="correct", aggfunc="mean")
for k in (10, 25, 50):  # items skipped in the dose run were at ceiling at both ends
    rate[k] = rate[k].fillna(1.0)
rate = rate[KS]
tip = rate.apply(lambda r: next((k for k in KS if r[k] >= 0.75), np.inf), axis=1)
order = {0: 0, 10: 1, 25: 2, 50: 3, 100: 4, np.inf: 5}
d = feat.join(tip.rename("tip")).assign(rank=lambda x: x.tip.map(order))
labels = {0: "right without filler", 10: "tips at 10", 25: "tips at 25", 50: "tips at 50", 100: "tips at 100", np.inf: "never"}
d["group"] = d.tip.map(labels)
print("items by tipping point:", d.group.value_counts().reindex(labels.values()).to_dict())

# ---------------- 1. tipping point vs arithmetic difficulty ----------------
print("\n=== 1a. mean difficulty features by tipping point ===")
means = d.groupby("group")[FEATURES].mean().reindex(labels.values())
print(means.round(2).T.to_string())
out["feature_means_by_tip"] = means.round(3).to_dict("index")
out["counts"] = d.group.value_counts().to_dict()

print("\n=== 1b. Spearman correlation of each feature with the tipping rank ===")
wrong0 = d[d["rank"] >= 1]                 # not right without filler
flippers = d[(d["rank"] >= 1) & (d["rank"] <= 4)]  # tip at some filler length
out["spearman"] = {}
print(f"{'feature':24s} {'all 300 (0..never)':>22s} {'wrong at k=0: tip k..never':>28s} {'flippers only: tip k':>22s} {'flip vs never (MWU p)':>22s}")
for f in FEATURES:
    r_all, r_w, r_f = spearmanr(d[f], d["rank"]), spearmanr(wrong0[f], wrong0["rank"]), spearmanr(flippers[f], flippers["rank"])
    mw = mannwhitneyu(flippers[f], d[d["rank"] == 5][f])
    out["spearman"][f] = {"all": [float(r_all[0]), float(r_all[1])], "wrong_at_k0": [float(r_w[0]), float(r_w[1])],
                          "flippers": [float(r_f[0]), float(r_f[1])], "flip_vs_never_p": float(mw.pvalue),
                          "mean_flippers": float(flippers[f].mean()), "mean_never": float(d[d["rank"] == 5][f].mean())}
    print(f"{f:24s} {r_all[0]:+.2f} (p={r_all[1]:.2g}){'':6s} {r_w[0]:+.2f} (p={r_w[1]:.2g}){'':10s} {r_f[0]:+.2f} (p={r_f[1]:.2g}){'':6s} {mw.pvalue:.2g}")
print(f"n: all {len(d)}, wrong at k=0 {len(wrong0)}, flippers {len(flippers)}, never {int((d['rank'] == 5).sum())}")

# how much of the outcome do the features explain? (cross-validated logistic-free check: rank regression R^2)
def r2(sub, y):
    X = np.column_stack([np.ones(len(sub))] + [(sub[f] - sub[f].mean()) / (sub[f].std() + 1e-9) for f in FEATURES if f not in ("x", "y")])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    n, p = X.shape
    return 1 - (res.var() * n / (n - p)) / y.var()


out["adj_r2"] = {"right_without_filler_all_items": float(r2(d, (d["rank"] == 0).astype(float).values)),
                 "ever_flips_among_wrong_at_k0": float(r2(wrong0, (wrong0["rank"] <= 4).astype(float).values)),
                 "tip_rank_among_flippers": float(r2(flippers, flippers["rank"].astype(float).values))}
print("\nadjusted R^2 of a linear model on all difficulty features:", {k: round(v, 3) for k, v in out["adj_r2"].items()})

# ---------------- 2. do stay-wrong items repeat the same wrong answer? ----------------
print("\n=== 2. items that stay wrong: is it the same wrong answer every time? ===")
allk["ans"] = allk.parsed.where(allk.parsed.notna(), -10**9)
stay = rate[(rate[0] <= 0.25) & (rate[100] <= 0.25)].index  # same definition as before (n=91)


def top_wrong(g):
    truth = g.answer.iloc[0]
    c = Counter(a for a in g.ans if a != truth)
    a, n = c.most_common(1)[0] if c else (None, 0)
    return pd.Series({"top": a, "share": n / len(g), "distinct": g.ans.nunique()})


tw = allk[allk.idx.isin(stay)].groupby(["idx", "k"]).apply(top_wrong).unstack("k")
out["stay_wrong"] = {"n": int(len(stay)), "by_k": {}}
print(f"{len(stay)} items. Share of the 16 samples taken by the most common wrong answer:")
for k in KS:
    s = tw["share"][k].astype(float)
    out["stay_wrong"]["by_k"][k] = {"mean_share": float(s.mean()), "ge_75": float((s >= 0.75).mean()), "ge_50": float((s >= 0.5).mean()),
                                    "le_25": float((s <= 0.25).mean()), "mean_distinct": float(tw["distinct"][k].astype(float).mean())}
    print(f"  k={k:3d}: mean {s.mean():.0%} | items where it is >= 75% of samples: {(s >= 0.75).mean():.0%} | >= 50%: {(s >= 0.5).mean():.0%} | "
          f"<= 25%: {(s <= 0.25).mean():.0%} | distinct answers {tw['distinct'][k].astype(float).mean():.1f}")
det = tw[tw["share"][100].astype(float) >= 0.75]
print(f"\n{len(det)} items give one wrong answer in >= 75% of samples at k=100. For those:")
same = {k: float((det["top"][k] == det["top"][100]).mean()) for k in (0, 10, 25, 50)}
print("  that answer is also the most common wrong answer at k =", {k: f"{v:.0%}" for k, v in same.items()})
share_at = {k: float(np.mean([(allk[(allk.idx == i) & (allk.k == k)].ans == det["top"][100][i]).mean() for i in det.index])) for k in (0, 10, 25, 50)}
print("  mean share of samples equal to that answer at k =", {k: f"{v:.0%}" for k, v in share_at.items()})
out["stay_wrong"]["deterministic_at_100"] = {"n": int(len(det)), "same_top_at_k": same, "mean_share_of_that_answer_at_k": share_at}


def story(a, it):
    """Simple single-mistake explanations for a wrong answer."""
    w, ref, op1, k1 = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).groups()
    c1, k1, x, y = COEF[w], int(k1), it["chain"]["x"], it["chain"]["y"]
    c2, op2, k2 = it["coefficient"], it["operation"], it["constant"]
    ap = lambda c, op, k, v: c * v + k if op == "plus" else c * v - k
    fl = lambda op: "minus" if op == "plus" else "plus"
    if a in it["rivals"].values():
        return "binding"
    ys = {ap(c1, fl(op1), k1, x), c1 * x} | {ap(c, op1, k1, x) for c in (2, 3) if c != c1}
    if a in {ap(c2, op2, k2, v) for v in ys} | {ap(c2, fl(op2), k2, y), c2 * y, y} | {ap(c, op2, k2, y) for c in (2, 3) if c != c2}:
        return "structural (sign / coefficient / dropped step)"
    return "near miss, same parity as the answer" if (a - it["answer"]) % 2 == 0 else "near miss, other parity"


st = Counter(story(int(det["top"][100][i]), items[i]) for i in det.index)
errs = [int(det["top"][100][i]) - items[i]["answer"] for i in det.index]
print("  what the repeated wrong answer is:", dict(st))
print(f"  its distance from the truth: median {np.median(np.abs(errs)):.0f}, quartiles {np.percentile(np.abs(errs), [25, 75]).astype(int).tolist()}; "
      f"below the truth in {np.mean(np.array(errs) < 0):.0%}")
out["stay_wrong"]["deterministic_at_100"].update({"stories": dict(st), "median_abs_error": float(np.median(np.abs(errs)))})

# reference: how consistent is the RIGHT answer on items that flip?
flip = rate[(rate[100] - rate[0]) >= 0.5].index
print(f"reference, {len(flip)} flip items at k=100: mean pass rate {rate.loc[flip, 100].mean():.0%}; "
      f"items with >= 75% right: {(rate.loc[flip, 100] >= 0.75).mean():.0%}")
json.dump(out, open(RES / "followup_tipping.json", "w"), indent=1, default=str)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
order_l = list(labels.values())
for ax, f in zip(axes[:2], ["carries total", "second product c2·y"]):
    data = [d[d.group == g][f].values for g in order_l]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order_l) + 1))
    ax.set_xticklabels([f"{g}\n(n={len(v)})" for g, v in zip(order_l, data)], fontsize=7)
    ax.set_title(f"{f} by tipping point", fontsize=9)
for k, c in zip([0, 100], ["C7", "C0"]):
    axes[2].hist(tw["share"][k].astype(float), bins=np.linspace(0, 1, 9), alpha=0.6, color=c, label=f"k={k}")
axes[2].set_xlabel("share of 16 samples taken by the most common wrong answer")
axes[2].set_ylabel("items")
axes[2].set_title(f"Items that stay wrong (n={len(stay)})", fontsize=9)
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_tipping.png", dpi=150)

# ---------------- 1c. breakdown by the two multipliers (added after the first pass) ----------------
d["outcome"] = np.where(d["rank"] == 0, "right without filler", np.where(d["rank"] <= 4, "flips", "never"))
cols = ["right without filler", "flips", "never"]
t1 = pd.crosstab(d["c1 = 3"].map({0: "c1 = 2", 1: "c1 = 3"}), d.outcome)[cols]
by = d.join(rate).groupby(["c1 = 3", "c2 = 3"])[KS].mean().assign(n=d.groupby(["c1 = 3", "c2 = 3"]).size())
print("\n=== 1c. outcome by first-hop multiplier ===")
print(t1.to_string())
print("mean pass rate by (c1 = 3, c2 = 3) and filler length:")
print(by.round(2).to_string())
out["by_c1"] = t1.to_dict("index")
out["pass_rate_by_c1_c2"] = {f"c1={3 if a else 2},c2={3 if b else 2}": {str(k): float(v) for k, v in row.items()} for (a, b), row in by.iterrows()}
within = {}
for name, sub, g1, g2 in [("c1 = 3: flips vs never", d[d["c1 = 3"] == 1], "flips", "never"),
                          ("c1 = 2: right without filler vs flips", d[d["c1 = 3"] == 0], "right without filler", "flips")]:
    within[name] = {}
    print(f"within {name} (n = {(sub.outcome == g1).sum()} vs {(sub.outcome == g2).sum()}):")
    for f in ["carries total", "digits in y", "first product c1·x", "second product c2·y", "c2 = 3", "subtractions", "x"]:
        a, b = sub[sub.outcome == g1][f], sub[sub.outcome == g2][f]
        p = float(mannwhitneyu(a, b).pvalue)
        within[name][f] = {"mean_first": float(a.mean()), "mean_second": float(b.mean()), "p": p}
        print(f"  {f:22s} {a.mean():7.2f} vs {b.mean():7.2f} (MWU p={p:.2g})")
out["within_c1"] = within
json.dump(out, open(RES / "followup_tipping.json", "w"), indent=1, default=str)
