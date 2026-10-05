"""Follow-up: independence test. err_filler = c * err_0^N across items (existing data, no API calls).

Run: python scripts/followup_independence.py
N independent attempts with recognition of a correct one predict err_filler ~ err_0^N
(log-log slope N > 1). One computation whose failure filler makes less likely predicts
err_filler ~ c * err_0 (slope 1).
Per-item rates from 16 samples are noisy, which would flatten a naive slope, so:
  (main) latent-variable binomial model: each item has a true no-filler error rate q with a
         nonparametric distribution (grid NPMLE from the no-filler counts); filler errors are
         Binomial(16, min(1, c*q^N)); (c, N) by maximum likelihood, CI by bootstrap over items.
         Uses every item, including those at 0% or 100%.
  (check) split-half instrumental-variable slope on log error rates, dropping items at 0% or
         100% in either condition.
Data: Phase 3 M=300 (16 samples at k=0 and at dots k=100, T=1) and the dose-response run.
Writes data/results/followup_independence.json and scripts/followup_independence.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binom, chi2

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "results"
rng = np.random.default_rng(0)
GRID = np.linspace(0, 1, 201)
out = {}

p3 = pd.DataFrame(json.loads(l) for l in open(RES / "phase3_M300.jsonl"))
p3["err"] = ~p3.correct
k0 = p3[p3.cond == "k0_T1.0"]
cnt = lambda d: d.groupby("idx").err.agg(["sum", "size"])
e0, ef = cnt(k0), cnt(p3[p3.cond == "F_T1.0"])
e0a, e0b = cnt(k0[k0["sample"] < 8]), cnt(k0[k0["sample"] >= 8])


def npmle(e, n, iters=200):
    """Grid NPMLE of the distribution of per-item true error rates from binomial counts."""
    L = binom.pmf(e[:, None], n[:, None], GRID[None, :])
    pi = np.full(len(GRID), 1 / len(GRID))
    for _ in range(iters):
        post = L * pi
        post /= post.sum(1, keepdims=True)
        pi = post.mean(0)
    return pi, L


def loglik(theta, L0pi, ef_, nf_):
    logc, n_exp = theta
    h = np.clip(np.exp(logc) * GRID ** n_exp, 1e-9, 1 - 1e-9)[None, :]
    # binomial likelihood without the constant coefficient (it does not depend on the parameters)
    lf = ef_[:, None] * np.log(h) + (nf_ - ef_)[:, None] * np.log1p(-h)
    return float(np.log((L0pi * np.exp(lf)).sum(1) + 1e-300).sum())


def fit(e0_, n0_, ef_, nf_):
    pi, L0 = npmle(e0_, n0_)
    L0pi = L0 * pi
    best = max((minimize(lambda t: -loglik(t, L0pi, ef_, nf_), x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-3}) for x0 in ([-0.3, 1], [0, 2])),
               key=lambda r: -r.fun)
    slope1 = minimize(lambda t: -loglik([t[0], 1.0], L0pi, ef_, nf_), [-0.3], method="Nelder-Mead")
    pure = minimize(lambda t: -loglik([0.0, t[0]], L0pi, ef_, nf_), [1.5], method="Nelder-Mead")
    return {"c": float(np.exp(best.x[0])), "N": float(best.x[1]), "ll": float(-best.fun),
            "slope1_c": float(np.exp(slope1.x[0])), "slope1_ll": float(-slope1.fun),
            "pure_N": float(pure.x[0]), "pure_ll": float(-pure.fun)}


def fit_with_ci(a, b, n_boot=300):
    """a, b: per-item (sum, size) frames for no filler and filler, same index."""
    arr = [a["sum"].values, a["size"].values, b["sum"].values, b["size"].values]
    r = fit(*arr)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        boots.append(fit(*[x[i] for x in arr]))
    for key in ["N", "c", "pure_N", "slope1_c"]:
        v = [x[key] for x in boots]
        r[key + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    r["share_boot_N_above_1"] = float(np.mean([x["N"] > 1 for x in boots]))
    r["lr_p_slope1_vs_general"] = float(chi2.sf(2 * (r["ll"] - r["slope1_ll"]), 1))
    r["lr_p_pure_vs_general"] = float(chi2.sf(2 * (r["ll"] - r["pure_ll"]), 1))
    return r


def show(label, r):
    print(f"{label}\n  general: slope N = {r['N']:.2f} [{r['N_ci'][0]:.2f}, {r['N_ci'][1]:.2f}], c = {r['c']:.2f} [{r['c_ci'][0]:.2f}, {r['c_ci'][1]:.2f}]"
          f" | bootstrap share with N > 1: {r['share_boot_N_above_1']:.0%}")
    print(f"  slope fixed at 1 (err_f = c*err_0): c = {r['slope1_c']:.2f} [{r['slope1_c_ci'][0]:.2f}, {r['slope1_c_ci'][1]:.2f}], "
          f"log-lik {r['slope1_ll']:.1f} vs general {r['ll']:.1f} (LR p = {r['lr_p_slope1_vs_general']:.2g})")
    print(f"  pure attempts (err_f = err_0^N): N = {r['pure_N']:.2f} [{r['pure_N_ci'][0]:.2f}, {r['pure_N_ci'][1]:.2f}], "
          f"log-lik {r['pure_ll']:.1f} (LR p vs general = {r['lr_p_pure_vs_general']:.2g})")


print("=== main: latent-variable binomial model, dots k=100 vs no filler, all 300 items ===")
out["k100_all_items"] = fit_with_ci(e0, ef)
show("300 items, 16 + 16 samples", out["k100_all_items"])

# ---- check: split-half instrumental-variable slope on log error rates ----
print("\n=== check: split-half IV slope (x = log err on half A, instrument = half B) ===")
d = pd.DataFrame({"a": e0a["sum"], "b": e0b["sum"], "f": ef["sum"], "tot": e0["sum"]})
keep = d[(d.tot > 0) & (d.tot < 16) & (d.f > 0) & (d.f < 16)]


def iv(dd, smooth=0.5):
    xa, xb, y = np.log((dd.a + smooth) / (8 + 2 * smooth)), np.log((dd.b + smooth) / (8 + 2 * smooth)), np.log((dd.f + smooth) / (16 + 2 * smooth))
    return float(np.cov(y, xb)[0, 1] / np.cov(xa, xb)[0, 1]), float(np.polyfit(xa, y, 1)[0])


for label, dd in [("items not at 0% or 100% in either condition", keep), ("all items (smoothed logs)", d)]:
    s_iv, s_naive = iv(dd)
    b = np.array([iv(dd.iloc[rng.integers(0, len(dd), len(dd))])[0] for _ in range(2000)])
    out[f"iv|{label}"] = {"n": int(len(dd)), "iv_slope": s_iv, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "naive_slope": s_naive}
    print(f"{label}: n={len(dd)} | IV slope {s_iv:.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}] | naive OLS slope {s_naive:.2f}")

# ---- what the data look like: bin by half A, read no-filler error on half B ----
print("\n=== binned view (bins from half A; no-filler error read from half B, so no selection bias) ===")
d["bin"] = pd.cut(d.a, [-1, 0, 2, 5, 7, 8], labels=["0/8", "1–2/8", "3–5/8", "6–7/8", "8/8"])
g = d.groupby("bin", observed=True).agg(n=("a", "size"), err0=("b", lambda s: s.mean() / 8), errf=("f", lambda s: s.mean() / 16),
                                        flipped=("f", lambda s: (s <= 4).mean()), stuck=("f", lambda s: (s >= 12).mean()))
r = out["k100_all_items"]
g["pred_general"] = np.clip(r["c"] * g.err0 ** r["N"], 0, 1)
g["pred_slope1"] = np.clip(r["slope1_c"] * g.err0, 0, 1)
g["pred_pure"] = g.err0 ** r["pure_N"]
g["ratio"] = g.errf / g.err0
print(g.round(3).to_string())
print("(flipped = share of items in the bin with filler error <= 25%; stuck = share with filler error >= 75%)")
out["bins"] = g.round(4).reset_index().astype({"bin": str}).to_dict("records")

# ---- slope of the MEAN relation, from the bin means (model-free) ----
def bin_slope(dd):
    gg = dd.groupby("bin", observed=True).agg(x=("b", lambda v: v.mean() / 8), y=("f", lambda v: v.mean() / 16), n=("a", "size"))
    gg = gg[(gg.x > 0) & (gg.y > 0)]
    return float(np.polyfit(np.log(gg.x), np.log(gg.y), 1, w=np.sqrt(gg.n))[0])


bs = np.array([bin_slope(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(2000)])
out["bin_mean_loglog_slope"] = {"slope": bin_slope(d), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
print(f"\nlog-log slope through the five bin means: {bin_slope(d):.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]")
print("implied N per bin if err_f = err_0^N: " + ", ".join(f"{b_}: {np.log(y) / np.log(x):.1f}" for b_, x, y in zip(g.index, g.err0, g.errf)))
out["implied_N_per_bin"] = {str(b_): float(np.log(y) / np.log(x)) for b_, x, y in zip(g.index, g.err0, g.errf)}

# ---- predictive check: do the models reproduce the all-or-nothing split within items? ----
print("\n=== predictive check: share of items ending mostly right / in between / mostly wrong with filler ===")
pi, L0 = npmle(e0["sum"].values, e0["size"].values)
post = L0 * pi
post /= post.sum(1, keepdims=True)
obs_e = ef["sum"].values
grp = pd.cut(e0["sum"].values, [-1, 4, 11, 16], labels=["no-filler error <= 25%", "26-74%", ">= 75%"])
models = {"independent attempts": np.clip(GRID ** r["pure_N"], 0, 1), "one computation (slope 1)": np.clip(r["slope1_c"] * GRID, 0, 1),
          "general fit": np.clip(r["c"] * GRID ** r["N"], 0, 1)}
out["predictive_check"] = {}
for gname in grp.categories:
    m = np.asarray(grp == gname)
    o = obs_e[m]
    row = {"n": int(m.sum()), "observed": [float((o <= 4).mean()), float(((o > 4) & (o < 12)).mean()), float((o >= 12).mean())]}
    line = f"{gname:24s} n={m.sum():3d} | observed {row['observed'][0]:.0%} / {row['observed'][1]:.0%} / {row['observed'][2]:.0%}"
    for name, h in models.items():
        cdf4, cdf11 = binom.cdf(4, 16, h), binom.cdf(11, 16, h)
        pr = [float((post[m] @ cdf4).mean()), float((post[m] @ (cdf11 - cdf4)).mean()), float((post[m] @ (1 - cdf11)).mean())]
        row[name] = pr
        line += f" | {name}: {pr[0]:.0%} / {pr[1]:.0%} / {pr[2]:.0%}"
    out["predictive_check"][gname] = row
    print(line)

# ---- by filler length (219 items not at ceiling in both existing conditions) ----
print("\n=== by filler length (dose-response items only; the 81 excluded items had ~0 error at k=0 and k=100) ===")
dose = pd.DataFrame(json.loads(l) for l in open(RES / "followup_dose.jsonl"))
dose["err"] = ~dose.correct
sub = sorted(set(dose.idx))
for k in [10, 25, 50]:
    rk = fit_with_ci(e0.loc[sub], cnt(dose[dose.k == k]).loc[sub], n_boot=150)
    out[f"k{k}_dose_items"] = rk
    show(f"k={k}, {len(sub)} items", rk)
rk = fit_with_ci(e0.loc[sub], ef.loc[sub], n_boot=150)
out["k100_dose_items"] = rk
show(f"k=100, same {len(sub)} items", rk)
json.dump(out, open(RES / "followup_independence.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
jit = lambda v: v + rng.uniform(-0.012, 0.012, len(v))
ax.scatter(jit(d.b / 8), jit(d.f / 16), s=10, alpha=0.3, color="C7", label="items (no-filler error from half B)")
ax.plot(g.err0, g.errf, "ko-", ms=7, label="bin means (bins from half A)")
x = np.linspace(0.001, 1, 200)
ax.plot(x, np.clip(r["slope1_c"] * x, 0, 1), "C0--", label=f"one computation: {r['slope1_c']:.2f} × err₀")
ax.plot(x, x ** r["pure_N"], "C3--", label=f"independent attempts: err₀^{r['pure_N']:.2f}")
ax.plot(x, np.clip(r["c"] * x ** r["N"], 0, 1), "C2-", label=f"general fit: {r['c']:.2f} × err₀^{r['N']:.2f}")
ax.plot(x, x, color="k", lw=0.5)
ax.set_xlabel("error rate without filler")
ax.set_ylabel("error rate with dots k=100")
ax.set_title("Per-item error rates, V4 Flash, 300 items (T=1)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(Path(__file__).parent / "followup_independence.png", dpi=150)
