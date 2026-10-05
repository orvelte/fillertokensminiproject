"""Phase 3 analysis: per-item filler effect by group, maj@k fit, temperature control, T1-T3.

Run: python scripts/phase3_analysis.py --M 60
Reads data/results/phase3_M<M>.jsonl. Writes data/results/phase3_M<M>_summary.json,
data/results/phase3_M<M>_items.csv and scripts/phase3_delta_vs_p0_M<M>.png.

Sample split (avoids regression to the mean): of each item's 16 k=0, T=1 samples,
half A (samples 0-7) classifies the item; half B (samples 8-15) gives p0 and the maj@k
predictions. Unparsed responses count as wrong and as their own (non-correct) answer.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 0
N_BOOT = 10_000
MAJ_KS = [1, 3, 5, 9, 15]
MAJ_DRAWS = 2000
GROUPS = ["correct-modal", "tie", "present, non-modal", "absent"]
COLORS = {"correct-modal": "C0", "tie": "C7", "present, non-modal": "C1", "absent": "C3"}


def classify(answers, truth):
    """Group from half-A answers: is the correct answer the unique mode / tied / present / absent."""
    c = Counter(answers)
    n_true = c.get(truth, 0)
    if n_true == 0:
        return "absent"
    top_other = max([v for a, v in c.items() if a != truth], default=0)
    return "correct-modal" if n_true > top_other else "tie" if n_true == top_other else "present, non-modal"


def maj_at_k(answers, truth, k, rng):
    """P(majority of k draws with replacement is correct); ties broken uniformly at random."""
    vals = sorted(set(answers), key=str)
    codes = np.array([vals.index(a) for a in answers])
    draws = codes[rng.integers(0, len(codes), size=(MAJ_DRAWS, k))]
    counts = np.stack([(draws == i).sum(1) for i in range(len(vals))], axis=1).astype(float)
    counts += rng.random(counts.shape) * 0.5  # random tie-break
    return float((counts.argmax(1) == vals.index(truth)).mean()) if truth in vals else 0.0


def boot_ci(x, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--suffix", default="", help="task-set suffix, e.g. _hard")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame(json.loads(l) for l in open(ROOT / "data" / "results" / f"phase3_M{a.M}{a.suffix}.jsonl"))
    df["ans"] = df.parsed.where(df.parsed.notna(), "unparsed")

    rows = []
    for idx, d in df.groupby("idx"):
        truth = d.answer.iloc[0]
        k0 = d[d.cond == "k0_T1.0"].sort_values("sample")
        half_a, half_b = list(k0.ans[:8]), list(k0.ans[8:])
        row = {"idx": idx, "group": classify(half_a, truth),
               "p0": float(np.mean([x == truth for x in half_b])),
               "pf": d[d.cond == "F_T1.0"].correct.mean(),
               "pT0.7": d[d.cond == "k0_T0.7"].correct.mean(),
               "pT0.4": d[d.cond == "k0_T0.4"].correct.mean()}
        for k in MAJ_KS:
            row[f"maj@{k}"] = maj_at_k(half_b, truth, k, rng)
        rows.append(row)
    it = pd.DataFrame(rows)
    it["delta"] = it.pf - it.p0
    it.to_csv(ROOT / "data" / "results" / f"phase3_M{a.M}{a.suffix}_items.csv", index=False)
    preds = [f"maj@{k}" for k in MAJ_KS] + ["pT0.7", "pT0.4"]
    wrong_modal = it.group.isin(["present, non-modal", "absent"])

    out = {"M": len(it), "parse_rate": float(df.parsed.notna().mean()),
           "parse_rate_by_cond": df.groupby("cond").parsed.apply(lambda s: float(s.notna().mean())).to_dict()}

    # ---- group means of delta (filler) and of each predictor's delta, bootstrap over items ----
    out["groups"] = {}
    print(f"\n{'group':20s} {'n':>4s} {'p0':>6s} {'pf':>6s}  delta filler [95% CI]      " + "  ".join(f"{p:>7s}" for p in preds))
    for g in GROUPS + ["wrong-modal (non-modal + absent)", "all"]:
        m = wrong_modal if g.startswith("wrong-modal") else (it.group == g) if g != "all" else pd.Series(True, index=it.index)
        s = it[m]
        mean, lo, hi = boot_ci(s.delta, rng)
        entry = {"n": int(len(s)), "p0": float(s.p0.mean()) if len(s) else None, "pf": float(s.pf.mean()) if len(s) else None,
                 "delta": mean, "ci": [lo, hi], "half_width": (hi - lo) / 2,
                 "pred_delta": {p: float((s[p] - s.p0).mean()) if len(s) else None for p in preds}}
        for p in ["pT0.7", "pT0.4"]:
            entry[f"ci_{p}"] = list(boot_ci(s[p] - s.p0, rng)[1:])
        out["groups"][g] = entry
        if len(s):
            print(f"{g[:20]:20s} {len(s):4d} {s.p0.mean():6.3f} {s.pf.mean():6.3f}  {mean:+.3f} [{lo:+.3f}, {hi:+.3f}]   "
                  + "  ".join(f"{entry['pred_delta'][p]:+7.3f}" for p in preds))

    # ---- which predictor matches pf: mean gap and per-item error ----
    out["fit"] = {p: {"mean_pred": float(it[p].mean()), "mean_gap_vs_pf": float((it[p] - it.pf).mean()),
                      "per_item_rmse": float(np.sqrt(((it[p] - it.pf) ** 2).mean())),
                      "per_item_mae": float((it[p] - it.pf).abs().mean())} for p in preds + ["p0"]}
    print(f"\nmean pf = {it.pf.mean():.3f}, mean p0 = {it.p0.mean():.3f}")
    print(f"{'predictor':10s} {'mean':>6s} {'gap vs pf':>10s} {'item RMSE':>10s} {'item MAE':>9s}")
    for p, f in out["fit"].items():
        print(f"{p:10s} {f['mean_pred']:6.3f} {f['mean_gap_vs_pf']:+10.3f} {f['per_item_rmse']:10.3f} {f['per_item_mae']:9.3f}")
    best = {k: min(MAJ_KS, key=lambda kk: abs(it.loc[i, f"maj@{kk}"] - it.loc[i, "pf"])) for i, k in enumerate(it.idx)}
    out["per_item_best_maj_k"] = {str(k): int(v) for k, v in Counter(best.values()).items()}
    print("per-item best maj@k counts:", dict(sorted(Counter(best.values()).items())))

    # ---- headline fractions ----
    wm, pn = it[wrong_modal], it[it.group == "present, non-modal"]
    out["headline"] = {
        "wrong_modal_n": int(len(wm)), "wrong_modal_frac_worse": float((wm.delta < 0).mean()) if len(wm) else None,
        "wrong_modal_frac_better": float((wm.delta > 0).mean()) if len(wm) else None,
        "present_nonmodal_n": int(len(pn)), "present_nonmodal_frac_improve": float((pn.delta > 0).mean()) if len(pn) else None,
        "present_nonmodal_frac_worse": float((pn.delta < 0).mean()) if len(pn) else None}
    print("\nheadline:", json.dumps(out["headline"]))

    # ---- testability checks ----
    mid = it.p0.between(0.1, 0.9)
    need = 60 * len(it) / 300  # T1 threshold scaled to this M
    n_cm, n_wm = int((mid & (it.group == "correct-modal")).sum()), int((mid & wrong_modal).sum())
    mean, lo, hi = boot_ci(it.delta, rng)
    hw = {g: out["groups"][g]["half_width"] for g in ["correct-modal", "present, non-modal", "absent", "wrong-modal (non-modal + absent)"]}
    out["tests"] = {
        "T1": {"correct_modal_mid_p0": n_cm, "wrong_modal_mid_p0": n_wm, "needed_each_at_this_M": need,
               "projected_at_300": {"correct_modal": n_cm * 300 / len(it), "wrong_modal": n_wm * 300 / len(it)},
               "pass": bool(n_cm >= need and n_wm >= need)},
        "T2": {"mean_delta": mean, "ci": [lo, hi], "pass": bool(mean > 0)},
        "T3": {"half_widths": hw, "pass": bool(all(h < 0.05 for h in hw.values() if h == h)),
               "M_needed_for_5pp": {g: (len(it) * (h / 0.05) ** 2 if h == h else None) for g, h in hw.items()}}}
    print("tests:", json.dumps(out["tests"], indent=1))
    json.dump(out, open(ROOT / "data" / "results" / f"phase3_M{a.M}{a.suffix}_summary.json", "w"), indent=1)

    # ---- main plot: delta vs p0 by group ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for g in GROUPS:
        s = it[it.group == g]
        if len(s):
            ax.scatter(s.p0 + rng.uniform(-0.02, 0.02, len(s)), s.delta + rng.uniform(-0.01, 0.01, len(s)),
                       s=18, alpha=0.6, color=COLORS[g], label=f"{g} (n={len(s)})")
            e = out["groups"][g]
            ax.errorbar([s.p0.mean()], [e["delta"]], yerr=[[e["delta"] - e["ci"][0]], [e["ci"][1] - e["delta"]]],
                        marker="D", ms=9, color=COLORS[g], mec="k", capsize=4, zorder=5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("p0: no-filler pass rate (half B, T=1)")
    ax.set_ylabel("Δ = pf − p0")
    ax.set_title(f"Filler effect per item by half-A group, M={len(it)} (diamonds: group mean, 95% CI)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f"phase3_delta_vs_p0_M{a.M}{a.suffix}.png", dpi=150)


if __name__ == "__main__":
    main()
