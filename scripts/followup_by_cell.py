"""Follow-up: redo the item-level checks within each (c1, c2) cell, and re-split earlier
aggregates by the first-hop multiplier c1. Existing data only, no API calls.

Run: python scripts/followup_by_cell.py
A. Per cell: is the filler effect bimodal across items (some flip fully, others do not move) or a
   uniform shift? Mostly-right / in-between / mostly-wrong table, predictive check against
   independent attempts (err_f = err_0^N), proportional error (err_f = c*err_0) and a uniform
   additive shift, and the independence slope.
B. By c1: error mix (binding / arithmetic slip / other), parity retention, slip size.
Data: Phase 3 M=300 (16 samples at k=0 and at dots k=100, T=1); greedy N=600 for replication.
Writes data/results/followup_by_cell.json and scripts/followup_by_cell.png.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
COEF = {"twice": 2, "three times": 3}
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")
GRID = np.linspace(0, 1, 201)
CELLS = [(2, 2), (2, 3), (3, 2), (3, 3)]
rng = np.random.default_rng(0)
out = {}

items = {it["idx"]: it for it in map(json.loads, open(ROOT / "data/tasks/eval.jsonl"))}
for it in items.values():
    it["c1"] = COEF[EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).group(1)]
    it["c2"] = it["coefficient"]

p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
p3["c1"], p3["c2"] = p3.idx.map(lambda i: items[i]["c1"]), p3.idx.map(lambda i: items[i]["c2"])
k0, kf = p3[p3.cond == "k0_T1.0"], p3[p3.cond == "F_T1.0"]
per = pd.DataFrame({"e0": 16 - k0.groupby("idx").correct.sum(), "ef": 16 - kf.groupby("idx").correct.sum(),
                    "e0a": 8 - k0[k0["sample"] < 8].groupby("idx").correct.sum(), "e0b": 8 - k0[k0["sample"] >= 8].groupby("idx").correct.sum()})
per["c1"], per["c2"] = [items[i]["c1"] for i in per.index], [items[i]["c2"] for i in per.index]
three = lambda e: np.array([(e <= 4).mean(), ((e > 4) & (e < 12)).mean(), (e >= 12).mean()])
fmt = lambda v: " / ".join(f"{x:.0%}" for x in v)


def npmle(e, n=16, iters=300):
    L = binom.pmf(e[:, None], n, GRID[None, :])
    pi = np.full(len(GRID), 1 / len(GRID))
    for _ in range(iters):
        post = L * pi
        post /= post.sum(1, keepdims=True)
        pi = post.mean(0)
    return pi, L


def ll(h, L0pi, ef):
    h = np.clip(h, 1e-9, 1 - 1e-9)[None, :]
    return float(np.log((L0pi * np.exp(ef[:, None] * np.log(h) + (16 - ef)[:, None] * np.log1p(-h))).sum(1) + 1e-300).sum())


def predict(h, post):
    """Predicted share of items ending mostly right / in between / mostly wrong."""
    c4, c11 = binom.cdf(4, 16, h), binom.cdf(11, 16, h)
    return np.array([(post @ c4).mean(), (post @ (c11 - c4)).mean(), (post @ (1 - c11)).mean()])


# ---------------- A. per cell ----------------
fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
print("=== A. within each (c1, c2) cell: no filler vs dots k=100, 16 samples per item ===")
for ax, (c1, c2) in zip(axes, CELLS):
    d = per[(per.c1 == c1) & (per.c2 == c2)]
    e0, ef = d.e0.values, d.ef.values
    delta = (e0 - ef) / 16  # change in pass rate
    cell = {"n": int(len(d)), "pass_k0": float(1 - e0.mean() / 16), "pass_k100": float(1 - ef.mean() / 16),
            "k0_three_way": three(e0).tolist(), "k100_three_way": three(ef).tolist()}
    print(f"\n--- c1 = {c1}, c2 = {c2}: {len(d)} items | pass rate {cell['pass_k0']:.2f} -> {cell['pass_k100']:.2f} ---")
    print(f"items mostly right / in between / mostly wrong: no filler {fmt(three(e0))} | filler {fmt(three(ef))}")
    # how the change is distributed across items
    bins = {"falls >= 25 pts": (delta <= -0.25).mean(), "within ±25 pts": (np.abs(delta) < 0.25).mean(),
            "rises 25-49 pts": ((delta >= 0.25) & (delta < 0.5)).mean(), "rises >= 50 pts": (delta >= 0.5).mean()}
    cell["delta_distribution"] = {k: float(v) for k, v in bins.items()}
    print("change in pass rate per item: " + " | ".join(f"{k}: {v:.0%}" for k, v in bins.items()))

    # models, fitted within the cell
    pi, L0 = npmle(e0)
    post = L0 * pi
    post /= post.sum(1, keepdims=True)
    L0pi = L0 * pi
    n_fit = minimize_scalar(lambda n: -ll(GRID ** n, L0pi, ef), bounds=(0.2, 30), method="bounded").x
    c_fit = minimize_scalar(lambda c: -ll(np.clip(c * GRID, 0, 1), L0pi, ef), bounds=(0.05, 1.5), method="bounded").x
    shift = (e0.mean() - ef.mean()) / 16
    gen = minimize(lambda t: -ll(np.clip(np.exp(t[0]) * GRID ** t[1], 0, 1), L0pi, ef), [0, 1], method="Nelder-Mead").x
    models = {"independent attempts": GRID ** n_fit, "proportional error": np.clip(c_fit * GRID, 0, 1), "uniform shift": np.clip(GRID - shift, 0, 1)}
    cell["fits"] = {"attempts_N": float(n_fit), "proportional_c": float(c_fit), "uniform_shift_pp": float(100 * shift),
                    "general_N": float(gen[1]), "general_c": float(np.exp(gen[0])),
                    "loglik": {k: ll(h, L0pi, ef) for k, h in models.items()}}
    print(f"fits: attempts N = {n_fit:.2f} | proportional c = {c_fit:.2f} | uniform shift = {100 * shift:+.0f} pts | "
          f"general: {np.exp(gen[0]):.2f}·err^{gen[1]:.2f} (steep N signals all-or-nothing items, not attempts)")
    print("log-likelihoods: " + " | ".join(f"{k} {v:.1f}" for k, v in cell["fits"]["loglik"].items()))

    cell["predictive_check"] = {}
    for gname, m in [("all items", np.ones(len(d), bool)), ("items wrong >= 75% without filler", e0 >= 12)]:
        if m.sum() < 8:
            continue
        row = {"n": int(m.sum()), "observed": three(ef[m]).tolist()}
        line = f"{gname} (n={m.sum()}): observed {fmt(three(ef[m]))}"
        for name, h in models.items():
            pr = predict(h, post[m])
            row[name] = pr.tolist()
            line += f" | {name} {fmt(pr)}"
        cell["predictive_check"][gname] = row
        print(line)

    # independence slope: split-half IV on smoothed log error rates, bootstrap over items
    def iv(dd):
        xa, xb, y = np.log((dd.e0a + .5) / 9), np.log((dd.e0b + .5) / 9), np.log((dd.ef + .5) / 17)
        return float(np.cov(y, xb)[0, 1] / np.cov(xa, xb)[0, 1])
    b = [iv(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)]
    cell["iv_slope"] = {"slope": iv(d), "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
    print(f"independence slope (split-half IV, log error): {iv(d):.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]")

    # is the spread of per-item change larger than a uniform shift plus sampling noise would give?
    q = np.clip(GRID - shift, 0, 1)
    sims = []
    for _ in range(2000):
        g = np.array([rng.choice(len(GRID), p=p_) for p_ in post])
        sims.append(np.std((rng.binomial(16, GRID[g]) - rng.binomial(16, q[g])) / 16))
    cell["delta_sd"] = {"observed": float(delta.std()), "uniform_shift_expected": float(np.mean(sims)),
                        "uniform_shift_95": [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]}
    print(f"SD of per-item change: observed {delta.std():.3f} | uniform shift + sampling noise would give "
          f"{np.mean(sims):.3f} [{np.percentile(sims, 2.5):.3f}, {np.percentile(sims, 97.5):.3f}]")
    out[f"c1={c1},c2={c2}"] = cell

    ax.hist(delta, bins=np.linspace(-1, 1, 17), color="C0")
    ax.axvline(delta.mean(), color="k", ls="--", lw=1)
    ax.set_title(f"c1 = {c1}, c2 = {c2} (n={len(d)}): {cell['pass_k0']:.2f} → {cell['pass_k100']:.2f}", fontsize=10)
    ax.set_xlabel("change in pass rate per item (filler − no filler)")
axes[0].set_ylabel("items")
fig.suptitle("Per-item change in pass rate with dots k=100, by first-hop (c1) and second-hop (c2) multiplier; dashed: cell mean", fontsize=10)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_by_cell.png", dpi=150)

# ---------------- B. earlier aggregates, split by c1 ----------------
ap = lambda c, op, k, v: c * v + k if op == "plus" else c * v - k


def etype(parsed, i):
    it = items[i]
    if parsed is None or pd.isna(parsed):
        return "other"
    a = int(parsed)
    if a == it["answer"]:
        return "correct"
    w, ref, op1, k1 = EXPR.match(dict(map(tuple, it["definitions"]))[it["queried_term"]]).groups()
    bind = set(it["rivals"].values()) | {ap(it["c2"], it["operation"], it["constant"], ap(COEF[w], op1, int(k1), v))
                                         for n, v in it["values"].items() if n not in (it["queried_term"], ref)}
    if a in bind:
        return "binding error"
    return "arithmetic slip" if abs(a - it["answer"]) <= 60 else "other"


def rel_ci(df, base, treat, t, n=2000):
    pc = df[df.cond.isin([base, treat])].assign(hit=lambda x: x.etype == t).groupby(["idx", "cond"]).hit.sum().unstack(fill_value=0)
    b, f = pc[base].values, pc[treat].values
    idx = rng.integers(0, len(b), size=(n, len(b)))
    bs = f[idx].sum(1) / np.maximum(b[idx].sum(1), 1) - 1
    return float(f.sum() / max(b.sum(), 1) - 1), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(b.sum()), int(f.sum())


print("\n\n=== B1. error mix by first-hop multiplier (sampled, 16 per item) ===")
p3["etype"] = [etype(a, i) for a, i in zip(p3.parsed, p3.idx)]
out["error_mix_by_c1"] = {}
for c1 in (2, 3):
    d = p3[(p3.c1 == c1) & p3.cond.isin(["k0_T1.0", "F_T1.0"])]
    n0, nf = (d.cond == "k0_T1.0").sum(), (d.cond == "F_T1.0").sum()
    print(f"c1 = {c1} ({d.idx.nunique()} items):")
    res = {}
    for t in ["binding error", "arithmetic slip", "other"]:
        r, lo, hi, b, f = rel_ci(d, "k0_T1.0", "F_T1.0", t)
        res[t] = {"no_filler_pct": 100 * b / n0, "filler_pct": 100 * f / nf, "rel_change": r, "ci": [lo, hi], "counts": [b, f]}
        print(f"  {t:16s} {100 * b / n0:5.1f}% -> {100 * f / nf:5.1f}% of samples | {r:+.0%} [{lo:+.0%}, {hi:+.0%}] ({b} -> {f})")
    w0, wf = (~d[d.cond == "k0_T1.0"].correct).sum(), (~d[d.cond == "F_T1.0"].correct).sum()
    res["all_wrong"] = {"no_filler_pct": 100 * w0 / n0, "filler_pct": 100 * wf / nf, "rel_change": wf / w0 - 1}
    res["mix_no_filler"] = {t: res[t]["counts"][0] / w0 for t in ["binding error", "arithmetic slip", "other"]}
    res["mix_filler"] = {t: res[t]["counts"][1] / wf for t in ["binding error", "arithmetic slip", "other"]}
    print(f"  all wrong        {100 * w0 / n0:5.1f}% -> {100 * wf / nf:5.1f}% | {wf / w0 - 1:+.0%} | share of errors (binding / slip / other): "
          f"no filler {fmt(res['mix_no_filler'].values())} | filler {fmt(res['mix_filler'].values())}")
    out["error_mix_by_c1"][c1] = res

print("\n=== B1b. same on the greedy 600-item run (dots k=100) ===")
p2 = pd.DataFrame(json.loads(l) for l in open(RES / "phase2_full.jsonl"))
p2["c1"] = p2.idx.map(lambda i: items[i]["c1"])
p2["etype"] = [etype(a, i) for a, i in zip(p2.parsed, p2.idx)]
out["error_mix_by_c1_greedy"] = {}
for c1 in (2, 3):
    d = p2[p2.c1 == c1]
    n = d.idx.nunique()
    row = {}
    for t in ["binding error", "arithmetic slip", "other"]:
        r, lo, hi, b, f = rel_ci(d, "dots_0", "dots_100", t)
        row[t] = {"counts": [b, f], "rel_change": r, "ci": [lo, hi]}
    out["error_mix_by_c1_greedy"][c1] = row
    print(f"c1 = {c1} ({n} items): " + " | ".join(f"{t} {v['counts'][0]} -> {v['counts'][1]} ({v['rel_change']:+.0%} [{v['ci'][0]:+.0%}, {v['ci'][1]:+.0%}])" for t, v in row.items()))

print("\n=== B2. parity retention by cell (non-binding wrong answers; share with the answer's parity) ===")
wrong = p3[p3.cond.isin(["k0_T1.0", "F_T1.0"]) & ~p3.correct & p3.parsed.notna() & (p3.etype != "binding error")].copy()
wrong["zerr"] = [int(a) - items[i]["answer"] for a, i in zip(wrong.parsed, wrong.idx)]
out["answer_parity_by_cell"] = {}
for c1, c2 in CELLS:
    line, row = f"c1 = {c1}, c2 = {c2}: ", {}
    for cond, lab in [("k0_T1.0", "no filler"), ("F_T1.0", "filler")]:
        d = wrong[(wrong.c1 == c1) & (wrong.c2 == c2) & (wrong.cond == cond)]
        row[lab] = {"n": int(len(d)), "answer_parity_kept": float((d.zerr % 2 == 0).mean()) if len(d) else None,
                    "divisible_by_c2": float((d.zerr % c2 == 0).mean()) if len(d) else None}
        line += f"{lab} {row[lab]['answer_parity_kept']:.1%} (n={len(d)}, {d.idx.nunique()} items)   " if len(d) else f"{lab} n=0   "
    note = "forced by the task" if c2 == 2 else "not forced"
    out["answer_parity_by_cell"][f"c1={c1},c2={c2}"] = row
    print(line + f"[{note}]")

print("\n=== B3. slip size by first-hop multiplier (|answer error|, non-binding wrong answers) ===")
out["slip_size_by_c1"] = {}
for c1 in (2, 3):
    row = {}
    for cond, lab in [("k0_T1.0", "no filler"), ("F_T1.0", "filler")]:
        a = wrong[(wrong.c1 == c1) & (wrong.cond == cond)].zerr.abs()
        row[lab] = {"n": int(len(a)), "median": float(a.median()), "q25": float(a.quantile(.25)), "q75": float(a.quantile(.75))}
    out["slip_size_by_c1"][c1] = row
    print(f"c1 = {c1}: " + " | ".join(f"{lab}: median {v['median']:.0f}, quartiles {v['q25']:.0f}-{v['q75']:.0f} (n={v['n']})" for lab, v in row.items()))
json.dump(out, open(RES / "followup_by_cell.json", "w"), indent=1)
