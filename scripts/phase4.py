"""Phase 4 (optional): filler-type ladder, matched on token count (behavioural only).

Run: python scripts/phase4.py pilot     # 100 items x 9 new conditions
     python scripts/phase4.py full      # N=600 (pilot calls are cache hits)
Every filler is built to 100 tokens (V4 Flash tokenizer, cache/tokenizer/tokenizer.json) and the
same filler string appears in all 10 few-shot examples and the test item. k=0 and dots (100
tokens) are reused from Phase 2. Greedy, paired. Surprisal is deferred: no prompt-logprob route.
Writes data/tasks/phase4_fillers.json, data/results/phase4_<stage>.jsonl,
data/results/phase4_<stage>_table.csv and scripts/phase4_ladder_<stage>.png.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import N_FEW_SHOT, SYSTEM, parse_answer, user_turn

TARGET, SEED = 100, 20261004
USD_PER_PROMPT_TOKEN = 0.14e-6
tok = Tokenizer.from_file(str(ROOT / "cache/tokenizer/tokenizer.json"))
ntok = lambda s: len(tok.encode(" " + s, add_special_tokens=False).ids)  # as it appears after "Filler:"

ONES = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
words_for = lambda n: ONES[n - 1] if n < 20 else TENS[n // 10 - 2] + ("" if n % 10 == 0 else " " + ONES[n % 10 - 1])

GETTYSBURG = ("Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, "
              "and dedicated to the proposition that all men are created equal. Now we are engaged in a great civil war, testing "
              "whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great "
              "battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those "
              "who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.")
P_TEXT = ("The town library opens early on weekdays and stays quiet until the afternoon. People come in to read the newspaper, "
          "return books, and ask about new arrivals. The staff keep the shelves tidy and help visitors find what they need, "
          "and in the evening they switch off the lamps and lock the front door.")
Q_TEXT = ("A small bakery sits on the corner near the bus stop. In the morning the owner arranges bread and pastries in the "
          "window while regulars wait outside. By noon most of the loaves are gone and the shop grows calm again, so the "
          "owner sweeps the floor and starts preparing dough for the next day.")


def fit(units, target=TARGET):
    """Longest space-joined prefix of `units` that is at most `target` tokens."""
    out = []
    for u in units:
        if ntok(" ".join(out + [u])) > target:
            break
        out.append(u)
    return " ".join(out)


def build_fillers():
    rng = random.Random(SEED)
    p, q = fit(P_TEXT.split(), TARGET // 2), fit(Q_TEXT.split(), TARGET // 2)
    pq_words = (p + " " + q).split()
    shuffled = pq_words[:]
    rng.shuffle(shuffled)
    vocab = sorted(k[1:] for k in tok.get_vocab() if k.startswith("Ġ") and len(k) >= 4 and k[1:].isascii() and k[1:].isalpha() and k[1:].islower())
    f = {
        "dots": " ".join(["."] * TARGET),
        "repeated word": fit(["the"] * 300),
        "counting, words": fit([words_for(n) for n in range(1, 100)]),
        "counting, digits": fit([str(n) for n in range(1, 300)]),
        "short cycle": fit(["red", "blue", "green"] * 100),
        "memorized text": fit(GETTYSBURG.split()),
        "prose P+Q": fit(pq_words),
        "prose P+P": fit((p + " " + p).split()),
        "shuffled prose": fit(shuffled),
        "random vocabulary": fit(rng.sample(vocab, 150)),
    }
    return f, {k: ntok(v) for k, v in f.items()}


def messages(few_shot, item, filler):
    msgs = [{"role": "system", "content": SYSTEM}]
    for fs in few_shot[:N_FEW_SHOT]:
        msgs.append({"role": "user", "content": user_turn(fs, filler)})
        msgs.append({"role": "assistant", "content": str(fs["answer"])})
    msgs.append({"role": "user", "content": user_turn(item, filler)})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["pilot", "full"])
    a = ap.parse_args()
    n = 100 if a.stage == "pilot" else 600
    load = lambda name: [json.loads(l) for l in open(ROOT / "data" / "tasks" / name)]
    few_shot, items = load("fewshot.jsonl"), load("eval.jsonl")[:n]
    fillers, counts = build_fillers()
    json.dump({"target_tokens": TARGET, "seed": SEED, "token_counts": counts, "fillers": fillers},
              open(ROOT / "data/tasks/phase4_fillers.json", "w"), indent=1)
    new = [c for c in fillers if c != "dots"]

    reqs, meta = [], []
    for c in new:
        for it in items:
            reqs.append({"messages": messages(few_shot, it, fillers[c])})
            meta.append((c, it))
    projected = sum(len(json.dumps(r["messages"])) / 2.5 for r in reqs) * USD_PER_PROMPT_TOKEN
    print(f"{len(reqs)} calls, projected <= ${projected:.2f}", flush=True)
    resps = api.chat_many(reqs, temperature=0.0, max_tokens=10, tag=f"phase4_{a.stage}", projected_usd=projected)

    rows = []
    with open(ROOT / "data/results" / f"phase4_{a.stage}.jsonl", "w") as f:
        for (c, it), r in zip(meta, resps):
            text = api.text_of(r)
            ans = parse_answer(text)
            row = {"idx": it["idx"], "cond": c, "response": text, "parsed": ans, "answer": it["answer"],
                   "correct": ans == it["answer"], "violation": ans is None or api.has_reasoning(r),
                   "prompt_tokens": r["usage"]["prompt_tokens"]}
            rows.append(row)
            f.write(json.dumps({**row, "temperature": 0.0, "max_tokens": 10, "usage": r["usage"]}) + "\n")
    # reuse Phase 2 for no filler and dots (100 tokens)
    for l in open(ROOT / "data/results/phase2_full.jsonl"):
        r = json.loads(l)
        if r["idx"] < n and r["cond"] in ("dots_0", "dots_100"):
            rows.append({"idx": r["idx"], "cond": "no filler" if r["cond"] == "dots_0" else "dots", "response": r["response"],
                         "parsed": r["parsed"], "answer": r["answer"], "correct": r["correct"], "violation": r["parsed"] is None,
                         "prompt_tokens": r["prompt_tokens"]})
    df = pd.DataFrame(rows)
    c = df.pivot(index="idx", columns="cond", values="correct")
    base_tok = df[df.cond == "no filler"].prompt_tokens.mean()

    def mcnemar(x, y):
        up, down = int((~c[x] & c[y]).sum()), int((c[x] & ~c[y]).sum())
        return up, down, (binomtest(up, up + down, 0.5).pvalue if up + down else float("nan"))

    table = []
    for cond in ["no filler", "dots"] + new:
        d = df[df.cond == cond]
        up, down, p = mcnemar("no filler", cond) if cond != "no filler" else (0, 0, float("nan"))
        _, _, p_dots = mcnemar("dots", cond) if cond not in ("no filler", "dots") else (0, 0, float("nan"))
        ok = d[~d.violation]
        table.append({"cond": cond, "acc": d.correct.mean(), "se": (d.correct.mean() * (1 - d.correct.mean()) / len(d)) ** 0.5,
                      "gain_pp": 100 * (d.correct.mean() - c["no filler"].mean()), "wrong_to_right": up, "right_to_wrong": down,
                      "p_vs_no_filler": p, "p_vs_dots": p_dots, "violations": int(d.violation.sum()),
                      "acc_format_following": ok.correct.mean(),
                      "filler_tokens_in_prompt": (d.prompt_tokens.mean() - base_tok) / (N_FEW_SHOT + 1)})
    t = pd.DataFrame(table)
    t.to_csv(ROOT / "data/results" / f"phase4_{a.stage}_table.csv", index=False)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nmatched pairs (first vs second; up = second right where first wrong):")
    pairs = {}
    for x, y in [("shuffled prose", "prose P+Q"), ("prose P+Q", "prose P+P"), ("counting, digits", "counting, words")]:
        up, down, p = mcnemar(x, y)
        pairs[f"{x} vs {y}"] = {"acc_first": float(c[x].mean()), "acc_second": float(c[y].mean()), "up": up, "down": down, "p": float(p)}
        print(f"  {x} {c[x].mean():.1%} vs {y} {c[y].mean():.1%}: up {up}, down {down}, McNemar p={p:.2g}")
    json.dump(pairs, open(ROOT / "data/results" / f"phase4_{a.stage}_pairs.json", "w"), indent=1)

    t2 = t.sort_values("acc")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["C7" if x == "no filler" else "C0" for x in t2.cond]
    ax.barh(t2.cond, 100 * t2.acc, xerr=100 * t2.se, capsize=3, color=colors)
    for y, (acc, se) in enumerate(zip(t2.acc, t2.se)):
        ax.text(100 * (acc + se) + 1, y, f"{100 * acc:.1f}%", va="center", fontsize=9)
    ax.axvline(100 * c["no filler"].mean(), color="k", ls="--", lw=0.8)
    ax.set_xlim(0, 80)
    ax.set_xlabel("accuracy (%), greedy (±1 SE)")
    ax.set_title(f"Filler types matched at about {TARGET} tokens, V4 Flash, N={n}", fontsize=10)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f"phase4_ladder_{a.stage}.png", dpi=150)
    print(f"total spend ${api.total_spend():.4f}")


if __name__ == "__main__":
    main()
