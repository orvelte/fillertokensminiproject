"""Phase 3 data collection: K=16 samples per item per condition.

Run: python scripts/phase3.py --M 60     # pilot
     python scripts/phase3.py --M 300    # full (pilot calls are cache hits)
Conditions: k=0 at T=1.0 / 0.7 / 0.4 and F* at T=1.0. Writes data/results/phase3_M<M>.jsonl.
No API seed is sent; samples are distinguished by the client-side `sample` index (cache key).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import build_messages, parse_answer

FSTAR = ("dots", 100)
K = 16
# (label, filler kind, k, temperature)
CONDS = [("k0_T1.0", "dots", 0, 1.0), ("F_T1.0", *FSTAR, 1.0), ("k0_T0.7", "dots", 0, 0.7), ("k0_T0.4", "dots", 0, 0.4)]
USD_PER_PROMPT_TOKEN = 0.14e-6


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, required=True)
    ap.add_argument("--suffix", default="", help="task-set suffix, e.g. _hard")
    a = ap.parse_args()
    few_shot, items = load(f"fewshot{a.suffix}.jsonl"), load(f"eval{a.suffix}.jsonl")[:a.M]

    reqs, meta = [], []
    for label, kind, k, temp in CONDS:
        for it in items:
            msgs = build_messages(few_shot, it, kind, k)
            for s in range(K):
                reqs.append({"messages": msgs, "temperature": temp, "sample": s})
                meta.append((label, temp, s, it))
    projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
    print(f"{len(reqs)} calls, projected <= ${projected:.2f} if nothing is cached", flush=True)
    resps = api.chat_many(reqs, max_tokens=10, tag=f"phase3_M{a.M}{a.suffix}", projected_usd=projected)

    n_reason = 0
    with open(ROOT / "data" / "results" / f"phase3_M{a.M}{a.suffix}.jsonl", "w") as f:
        for (label, temp, s, it), r in zip(meta, resps):
            text = api.text_of(r)
            ans = parse_answer(text)
            n_reason += api.has_reasoning(r)
            f.write(json.dumps({"idx": it["idx"], "cond": label, "temperature": temp, "sample": s,
                                "response": text, "parsed": ans, "answer": it["answer"],
                                "correct": ans == it["answer"], "max_tokens": 10, "usage": r["usage"]}) + "\n")
    print(f"done; calls with reasoning: {n_reason}; total spend ${api.total_spend():.4f}")


if __name__ == "__main__":
    main()
