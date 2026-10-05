"""Phase 0 verification checks V1-V4 + token measurements for the cost model.

Run: python scripts/phase0.py
Uses Brauer's shipped varbind dataset as stand-in items (Phase 1 generates the real set).
"""
import json
import sys
from collections import Counter
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
import api
from prompts import build_messages, parse_answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "results" / "phase0.jsonl"
TAG = "phase0"

ds = json.load(open(ROOT / "vendor/filler-token-reasoning/data/chained_var_binding_dataset.json"))
few_shot = ds["few_shot_examples"] + ds["examples"][:2]  # 10 stand-in few-shot
items = ds["examples"][2:]

records = []


def run(reqs, check):
    resps = api.chat_many([{k: v for k, v in r.items() if not k.startswith("_")} for r in reqs], tag=TAG)
    for req, resp in zip(reqs, resps):
        for i in range(len(resp["choices"])):
            records.append({"check": check, "item": req["_item"], "cond": req["_cond"],
                            "temperature": req.get("temperature", 0.0), "sample": req.get("sample", 0),
                            "text": api.text_of(resp, i), "parsed": parse_answer(api.text_of(resp, i)),
                            "answer": items[req["_item"]]["answer"], "usage": resp.get("usage"),
                            "finish": resp["choices"][i].get("finish_reason"), "cached_call": resp["_cached"]})
    return resps


def req(i, kind="dots", k=0, **kw):
    return {"messages": build_messages(few_shot, items[i], kind, k), "max_tokens": 10,
            "_item": i, "_cond": f"{kind}_{k}", **kw}


# ---- V1: no hidden reasoning (20 calls: 10 at k=0, 10 at dots k=25) ----
v1 = run([req(i, k=0) for i in range(10)] + [req(i, k=25) for i in range(10)], "V1")
n_reason = sum(api.has_reasoning(r) for r in v1)
print(f"V1 calls with reasoning tokens or reasoning_content: {n_reason}/20 -> {'PASS' if n_reason == 0 else 'FAIL'}")

# ---- V2: answer format (40 items x {k=0, dots 25, counting 25}, greedy, max_tokens=10) ----
v2_reqs = [req(i, kind, k) for kind, k in [("dots", 0), ("dots", 25), ("counting", 25)] for i in range(40)]
v2 = run(v2_reqs, "V2")
texts = [api.text_of(r) for r in v2]
parsed = [parse_answer(t) for t in texts]
rate = sum(p is not None for p in parsed) / len(parsed)
print(f"V2 parse rate: {rate:.1%} ({len(parsed)} responses) -> {'PASS' if rate >= 0.95 else 'FAIL'}")
print("   unparsed examples:", [t for t, p in zip(texts, parsed) if p is None][:5])
print("   reasoning in any V2 call:", sum(api.has_reasoning(r) for r in v2))
for c in ["dots_0", "dots_25", "counting_25"]:
    rs = [(q, p) for q, p in zip(v2_reqs, parsed) if q["_cond"] == c]
    acc = sum(p == items[q["_item"]]["answer"] for q, p in rs) / len(rs)
    us = [r["usage"] for q, r in zip(v2_reqs, v2) if q["_cond"] == c]
    print(f"   {c}: acc {acc:.0%} (n=40, stand-in items) | mean prompt tok {sum(u['prompt_tokens'] for u in us)/len(us):.0f}"
          f" | mean cached tok {sum(u['prompt_tokens_details']['cached_tokens'] for u in us)/len(us):.0f}"
          f" | mean completion tok {sum(u['completion_tokens'] for u in us)/len(us):.1f}")

# ---- V3: sampling ----
wrong = [q["_item"] for q, p in zip(v2_reqs, parsed) if q["_cond"] == "dots_0" and p != items[q["_item"]]["answer"]][:5]
v3 = run([req(i, k=0, temperature=1.0, sample=s) for i in wrong for s in range(8)], "V3_T1")
for j, i in enumerate(wrong):
    ans = Counter(parse_answer(api.text_of(r)) for r in v3[j * 8:(j + 1) * 8])
    print(f"V3 T=1 item {i} (truth {items[i]['answer']}): {dict(ans)}")
n_varied = sum(len({api.text_of(r) for r in v3[j * 8:(j + 1) * 8]}) > 1 for j in range(len(wrong)))
print(f"V3 temperature honored: {n_varied}/{len(wrong)} hard items give varied answers at T=1 -> {'PASS' if n_varied else 'FAIL'}")
v3g = run([req(i, k=0, temperature=0.0, sample=s) for i in wrong for s in range(4)], "V3_T0")
n_det = sum(len({api.text_of(r) for r in v3g[j * 4:(j + 1) * 4]}) == 1 for j in range(len(wrong)))
print(f"V3 greedy repeatable: {n_det}/{len(wrong)} items give identical answers over 4 repeats at T=0")

try:
    rn = run([req(wrong[0], k=0, temperature=1.0, n=8)], "V3_n8")[0]
    print(f"V3 n=8: returned {len(rn['choices'])} choices; answers {[api.text_of(rn, i) for i in range(len(rn['choices']))]}; usage {rn['usage']}")
except Exception as e:
    print("V3 n=8 not supported:", str(e)[:300])
cached = [r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in v3 if not r["_cached"]]
if cached:
    print(f"V3 prefix caching: {sum(c > 0 for c in cached)}/{len(cached)} paid T=1 calls report cached_tokens>0 (max {max(cached)})")

# ---- V4: prompt-token logprobs ----
cfg = api.PROVIDERS[api.DEFAULT_PROVIDER]
hdr = {"Authorization": f"Bearer {api.ENV[cfg['key_env']]}"}
try:
    r = api.chat(req(0, k=0)["messages"], max_tokens=10, extra={"logprobs": True, "top_logprobs": 5}, tag=TAG)
    lp = r["choices"][0].get("logprobs")
    print("V4 output-token logprobs (chat):", "yes" if lp else "no", json.dumps(lp)[:300] if lp else "")
except Exception as e:
    print("V4 output-token logprobs (chat): error", str(e)[:300])
for name, body in [
    ("completions echo+logprobs", {"url": cfg["url"].replace("/chat/completions", "/completions"),
                                   "json": {"model": cfg["model"], "prompt": "Filler: . . . . .\n\nAnswer:", "max_tokens": 1,
                                            "echo": True, "logprobs": 1, "temperature": 0}}),
    ("chat echo+logprobs", {"url": cfg["url"], "json": {"model": cfg["model"], "messages": req(0, k=5)["messages"][-1:],
                                                        "max_tokens": 1, "echo": True, "logprobs": True, "temperature": 0,
                                                        **cfg["no_thinking"]}}),
]:
    r = requests.post(body["url"], headers=hdr, json=body["json"], timeout=60)
    print(f"V4 {name}: HTTP {r.status_code}: {r.text[:700]}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
api.write_costs_md()
print(f"\nsaved {len(records)} records to {OUT}; total spend ${api.total_spend():.4f}")
