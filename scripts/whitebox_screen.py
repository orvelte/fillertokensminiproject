"""Uplift screen on smaller lens-equipped, standard-attention models (API only).

Run: python scripts/whitebox_screen.py [--suffix _easy]
Phase 2-style pilot: 150 items, 10-shot, greedy, k=0 vs dots k=25 vs dots k=100, thinking off.
Each model is pinned to the upstream host that answers a first probe call.
Writes data/results/whitebox_screen<suffix>.jsonl / .csv.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import build_messages, parse_answer

# OpenRouter id -> does the route accept the reasoning toggle (models with J-lenses in camilablank/workspace-lenses)
MODELS = {"qwen/qwen3.5-9b": True, "qwen/qwen3.5-27b": True, "qwen/qwen3.6-27b": True,
          "google/gemma-3-27b-it": False, "qwen/qwen3.6-35b-a3b": True,
          "qwen/qwen3.5-122b-a10b": True}  # 122B is the largest lens-equipped model after V4 Flash
CONDS = ["dots_0", "dots_25", "dots_100"]
N = 150

ap = argparse.ArgumentParser()
ap.add_argument("--suffix", default="", help="task-set suffix, e.g. _easy")
ap.add_argument("--models", nargs="*", default=list(MODELS)[:5])
a = ap.parse_args()
load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
few_shot, items = load(f"fewshot{a.suffix}.jsonl"), load(f"eval{a.suffix}.jsonl")[:N]

rows, table = [], []
for model in a.models:
    common = {"model": model, "thinking": not MODELS[model], "temperature": 0.0, "max_tokens": 10, "tag": f"whitebox_screen{a.suffix}"}
    try:
        probe = api.chat(build_messages(few_shot, items[0], "dots", 0), extra={"provider": {"allow_fallbacks": True}}, **common)
    except Exception as e:
        print(f"{model}: probe failed: {str(e)[:200]}")
        continue
    host = probe.get("provider")
    common["extra"] = {"provider": {"order": [host], "allow_fallbacks": False}}
    reqs, meta = [], []
    for c in CONDS:
        kind, k = c.split("_")
        for it in items:
            reqs.append({"messages": build_messages(few_shot, it, kind, int(k))})
            meta.append((c, it))
    try:
        resps = api.chat_many(reqs, projected_usd=0.5, **common)
    except Exception as e:
        print(f"{model} ({host}): run failed: {str(e)[:200]}")
        continue
    for (c, it), r in zip(meta, resps):
        text = api.text_of(r)
        ans = parse_answer(text)
        rows.append({"model": model, "host": host, "idx": it["idx"], "cond": c, "response": text, "parsed": ans,
                     "answer": it["answer"], "correct": ans == it["answer"], "reasoning": api.has_reasoning(r)})
    d = pd.DataFrame([r for r in rows if r["model"] == model])
    c = d.pivot(index="idx", columns="cond", values="correct")
    row = {"model": model, "host": host, "reasoning_calls": int(d.reasoning.sum()), "parse_rate": d.parsed.notna().mean(),
           "acc_k0": c["dots_0"].mean()}
    for cond in CONDS[1:]:
        up, down = int((~c["dots_0"] & c[cond]).sum()), int((c["dots_0"] & ~c[cond]).sum())
        row.update({f"acc_{cond}": c[cond].mean(), f"gain_{cond}_pp": 100 * (c[cond].mean() - c["dots_0"].mean()),
                    f"flips_{cond}": f"{up}/{down}", f"p_{cond}": binomtest(up, up + down, 0.5).pvalue if up + down else float("nan")})
    table.append(row)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

with open(ROOT / "data/results" / f"whitebox_screen{a.suffix}.jsonl", "a") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
t = pd.DataFrame(table)
t.to_csv(ROOT / "data/results" / f"whitebox_screen{a.suffix}.csv", mode="a", index=False)
print("\n" + t.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
print(f"total spend ${api.total_spend():.4f}")
