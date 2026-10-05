"""Phase 2: uplift screen (greedy, paired).

Run: python scripts/phase2.py pilot                          # 150 items x 6 conditions
     python scripts/phase2.py full --conds dots_50 dots_100  # k=0 + chosen conditions, N=600
     python scripts/phase2.py pilot --n-shot 5               # 5-shot diagnostic
Writes data/results/phase2_<stage>.jsonl, data/results/phase2_<stage>_table.csv and
scripts/phase2_uplift_<stage>.png.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import N_FEW_SHOT, build_messages, parse_answer

PILOT_CONDS = ["dots_0", "dots_10", "dots_25", "dots_50", "dots_100", "counting_25"]
N_PILOT, N_FULL = 150, 600
USD_PER_PROMPT_TOKEN = 0.14e-6  # no-cache upper bound, for the pre-run projection


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "full"])
    ap.add_argument("--conds", nargs="*", default=None, help="conditions besides dots_0 (full stage)")
    ap.add_argument("--n-shot", type=int, default=N_FEW_SHOT)
    a = ap.parse_args()
    conds = PILOT_CONDS if a.stage == "pilot" else ["dots_0"] + a.conds
    few_shot, items = load("fewshot.jsonl"), load("eval.jsonl")
    items = items[:N_PILOT if a.stage == "pilot" else N_FULL]
    stage = a.stage + (f"_{a.n_shot}shot" if a.n_shot != N_FEW_SHOT else "")

    reqs, meta = [], []
    for c in conds:
        kind, k = c.split("_")
        for it in items:
            reqs.append({"messages": build_messages(few_shot, it, kind, int(k), n_shot=a.n_shot)})
            meta.append((c, it))
    projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN  # generous chars/token
    print(f"{len(reqs)} calls, projected <= ${projected:.2f}")
    resps = api.chat_many(reqs, temperature=0.0, max_tokens=10, tag=f"phase2_{stage}", projected_usd=projected)

    rows = []
    with open(ROOT / "data" / "results" / f"phase2_{stage}.jsonl", "w") as f:
        for (c, it), r in zip(meta, resps):
            text = api.text_of(r)
            ans = parse_answer(text)
            row = {"idx": it["idx"], "cond": c, "response": text, "parsed": ans, "answer": it["answer"],
                   "correct": ans == it["answer"], "prompt_tokens": r["usage"]["prompt_tokens"],
                   "reasoning": api.has_reasoning(r)}
            rows.append(row)
            f.write(json.dumps({**row, "request": {"temperature": 0.0, "max_tokens": 10, "n_shot": a.n_shot},
                                "usage": r["usage"]}) + "\n")
    df = pd.DataFrame(rows)
    assert not df.reasoning.any(), "reasoning tokens found"

    correct = df.pivot(index="idx", columns="cond", values="correct")
    base_tok = df[df.cond == "dots_0"].prompt_tokens.mean()
    table = []
    for c in conds:
        d = df[df.cond == c]
        w2r = int((~correct["dots_0"] & correct[c]).sum())
        r2w = int((correct["dots_0"] & ~correct[c]).sum())
        p = binomtest(w2r, w2r + r2w, 0.5).pvalue if c != "dots_0" and w2r + r2w else float("nan")
        acc = d.correct.mean()
        table.append({"cond": c, "n": len(d), "acc": acc, "se": (acc * (1 - acc) / len(d)) ** 0.5,
                      "gain_pp": 100 * (acc - correct["dots_0"].mean()), "wrong_to_right": w2r,
                      "right_to_wrong": r2w, "mcnemar_p": p, "parse_rate": d.parsed.notna().mean(),
                      "filler_tokens_per_turn": (d.prompt_tokens.mean() - base_tok) / (a.n_shot + 1)})
    t = pd.DataFrame(table)
    t.to_csv(ROOT / "data" / "results" / f"phase2_{stage}_table.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    fig, ax = plt.subplots(figsize=(6, 4))
    dots = t[t.cond.str.startswith("dots")].assign(k=lambda x: x.cond.str.split("_").str[1].astype(int)).sort_values("k")
    ax.errorbar(range(len(dots)), 100 * dots.acc, yerr=100 * dots.se, marker="o", capsize=3, label="dots")
    ax.set_xticks(range(len(dots)))
    ax.set_xticklabels(dots.k)
    for _, r in t[t.cond.str.startswith("counting")].iterrows():
        k = int(r.cond.split("_")[1])
        x = list(dots.k).index(k) if k in list(dots.k) else len(dots)
        ax.errorbar([x + 0.08], [100 * r.acc], yerr=[100 * r.se], marker="s", capsize=3, color="C1", label=f"counting (k={k})")
    ax.set_xlabel("filler units k")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"V4 Flash, system of equations, {a.n_shot}-shot, greedy, N={len(items)} (±1 SE)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f"phase2_uplift_{stage}.png", dpi=150)
    print(f"total spend ${api.total_spend():.4f}")


if __name__ == "__main__":
    main()
