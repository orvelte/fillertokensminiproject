"""Phase 1: generate the system-of-equations task and calibrate difficulty.

Run: python scripts/phase1.py [--max-coef 3] [--const-max 50] [--num-terms 5]
Generates 10 few-shot + 600 eval items with Brauer's generator (fixed seed), adds the full
chain / distractor values / binding-error rivals, then pilots the first 100 eval items
(greedy, k=0). Writes data/tasks/{fewshot,eval}.jsonl, data/tasks/settings.json and
data/results/phase1_pilot_<settings>.jsonl.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "vendor/filler-token-reasoning/scripts"))
import api
from data.generate_varbind_dataset import COEF_WORDS, generate_unique_problems, load_real_words
from prompts import N_FEW_SHOT, build_messages, parse_answer

SEED = 20261004
N_EVAL = 600
N_PILOT = 100
VAL_MIN, VAL_MAX, CONST_MIN, CHAIN_LEN = 10, 99, 1, 1  # Brauer defaults, not calibration knobs

COEF_NUM = {w: n for n, w in COEF_WORDS.items()}
EXPR = re.compile(r"^(.+) the number for (\w+) (plus|minus) (\d+)$")


def apply(coef, op, const, v):
    return coef * v + const if op == "plus" else coef * v - const


def enrich(p, idx):
    """Add resolved values, the chain {x, c1x, y, c2y, answer}, distractors and rival answers."""
    values, parsed = {}, {}
    for name, d in p["definitions"]:  # definitions are in topological order
        if isinstance(d, int):
            values[name] = d
        else:
            w, ref, op, const = EXPR.match(d).groups()
            parsed[name] = (COEF_NUM[w], ref, op, int(const))
            values[name] = apply(COEF_NUM[w], op, int(const), values[ref])
    q = p["queried_term"]
    c1, x_name, _, _ = parsed[q]
    c2, op2, k2 = p["coefficient"], p["operation"], p["constant"]
    y = values[q]
    assert y == p["queried_value"] and apply(c2, op2, k2, y) == p["answer"]
    return {
        "idx": idx, **p,
        "values": values,
        "x_name": x_name,
        "chain": {"x": values[x_name], "c1x": c1 * values[x_name], "y": y, "c2y": c2 * y, "answer": p["answer"]},
        "distractors": {n: v for n, v in values.items() if n not in (q, x_name)},
        # binding errors: the question's operation applied to each other variable
        "rivals": {n: apply(c2, op2, k2, v) for n, v in values.items() if n != q},
    }


def similar_names(p, rng, words):
    """Rename an item's variables so they share a consonant-vowel prefix (zab, zad, zaf, ...)."""
    old = [n for n, _ in p["definitions"]]
    cons, vowels = "bcdfghjklmnpqrstvwxyz", "aeiou"
    while True:
        prefix = rng.choice(cons) + rng.choice(vowels)
        ends = [c for c in cons if prefix + c not in words]
        if len(ends) >= len(old):
            break
    m = dict(zip(old, (prefix + c for c in rng.sample(ends, len(old)))))
    sub = lambda t: re.sub(r"\b(" + "|".join(old) + r")\b", lambda g: m[g.group(1)], t)
    return {**p, "definitions": [[m[n], d if isinstance(d, int) else sub(d)] for n, d in p["definitions"]],
            "queried_term": m[p["queried_term"]], "question": sub(p["question"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-coef", type=int, default=3)
    ap.add_argument("--const-max", type=int, default=50)
    ap.add_argument("--num-terms", type=int, default=5)
    ap.add_argument("--suffix", default="", help="e.g. _hard: write fewshot_hard.jsonl etc. instead of the frozen set")
    ap.add_argument("--similar-names", action="store_true", help="variables in an item share a 2-letter prefix")
    a = ap.parse_args()
    settings = {"seed": SEED, "max_coef": a.max_coef, "const_min": CONST_MIN, "const_max": a.const_max,
                "num_terms": a.num_terms, "chain_len": CHAIN_LEN, "val_min": VAL_MIN, "val_max": VAL_MAX,
                "n_few_shot": N_FEW_SHOT, "n_eval": N_EVAL, "similar_names": a.similar_names}
    name = f"coef{a.max_coef}_const{a.const_max}_terms{a.num_terms}{a.suffix}"

    assert a.suffix or not (ROOT / "data/tasks/eval.jsonl").exists(), "frozen eval set exists; pass --suffix"
    words = load_real_words()
    assert words, "no word list found"
    problems = generate_unique_problems(N_FEW_SHOT + N_EVAL, random.Random(SEED), words, a.num_terms, CHAIN_LEN,
                                        VAL_MIN, VAL_MAX, CONST_MIN, a.const_max, a.max_coef)
    if a.similar_names:
        name_rng = random.Random(SEED + 1)  # separate stream: does not disturb the generator
        problems = [similar_names(p, name_rng, words) for p in problems]
    few_shot = [enrich(p, i) for i, p in enumerate(problems[:N_FEW_SHOT])]
    items = [enrich(p, i) for i, p in enumerate(problems[N_FEW_SHOT:])]

    tasks = ROOT / "data" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    for fname, rows in [(f"fewshot{a.suffix}.jsonl", few_shot), (f"eval{a.suffix}.jsonl", items)]:
        with open(tasks / fname, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump(settings, open(tasks / f"settings{a.suffix}.json", "w"), indent=2)

    # ---- pilot: first 100 eval items, greedy, k=0 ----
    pilot = items[:N_PILOT]
    resps = api.chat_many([{"messages": build_messages(few_shot, it, "dots", 0)} for it in pilot],
                          temperature=0.0, max_tokens=10, tag="phase1", projected_usd=0.05)
    out = ROOT / "data" / "results" / f"phase1_pilot_{name}.jsonl"
    n_ok = n_parse = n_rival = n_close = 0
    with open(out, "w") as f:
        for it, r in zip(pilot, resps):
            text = api.text_of(r)
            ans = parse_answer(text)
            n_ok += ans == it["answer"]
            n_parse += ans is not None
            n_rival += ans in it["rivals"].values() and ans != it["answer"]
            n_close += ans is not None and ans != it["answer"] and ans not in it["rivals"].values() and abs(ans - it["answer"]) <= 60
            f.write(json.dumps({"idx": it["idx"], "request": {"cond": "dots_0", "temperature": 0.0, "max_tokens": 10},
                                "response": text, "parsed": ans, "answer": it["answer"],
                                "correct": ans == it["answer"], "usage": r.get("usage")}) + "\n")
    print(f"settings: {settings}")
    print(f"pilot k=0 greedy accuracy: {n_ok}/{N_PILOT} | parsed {n_parse}/{N_PILOT} | "
          f"binding-error rival answers {n_rival} | other wrong within 60 of truth {n_close} | reasoning calls {sum(api.has_reasoning(r) for r in resps)}")
    print(f"total spend ${api.total_spend():.4f}; pilot saved to {out}")


if __name__ == "__main__":
    main()
