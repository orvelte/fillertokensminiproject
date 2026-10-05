"""Cached, budget-capped chat-completions client (OpenRouter, Fireworks or official DeepSeek).

Usage:
    from api import chat, chat_many
    r = chat(messages, temperature=0, max_tokens=10)            # one call
    rs = chat_many([dict(messages=m, temperature=1.0, sample=i) ...])  # concurrent

Every call is cached in cache/api_cache.sqlite keyed by sha256 of the full request
(provider, model, messages, temperature, max_tokens, seed, n, extra, sample).
`sample` is a client-side index only: it makes repeated T>0 draws of the same prompt
distinct cache entries and is never sent to the API. Cache hits cost nothing.
Spend is tracked from `usage` in the same sqlite file; notes/costs.md holds the running
total; a call is refused once spend reaches BUDGET_USD.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DB = ROOT / "cache" / "api_cache.sqlite"
COSTS_MD = ROOT / "notes" / "costs.md"

# USD per 1M tokens: (input cache-miss, input cache-hit, output). See notes/costs.md for sources.
PROVIDERS = {
    # DeepSeek V4 Flash 0423 = HF deepseek-ai/DeepSeek-V4-Flash, the checkpoint the J-lens was fit on.
    # Pinned to one upstream host so quantization never changes between calls.
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "key_env": "OPENROUTER_API_KEY",
        "model": "deepseek/deepseek-v4-flash",
        "price": (0.14, 0.07, 0.28),  # Parasail fp8; actual `usage.cost` is used when returned
        "no_thinking": {"reasoning": {"enabled": False}},
        "extra": {"provider": {"order": ["Parasail"], "allow_fallbacks": False}},
    },
    "fireworks": {
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",
        "key_env": "FIREWORKS_API_KEY",
        "model": "accounts/fireworks/models/deepseek-v4p1-flash",
        "price": (0.22, 0.007, 0.66),
        "no_thinking": {"thinking": {"type": "disabled"}},
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-flash",
        "price": (0.30, 0.006, 1.20),  # peak-hour prices (conservative)
        "no_thinking": {"thinking": {"type": "disabled"}},
    },
}
DEFAULT_PROVIDER = "openrouter"
MAX_CONCURRENCY = 16
MAX_RETRIES = 6


class BudgetExceeded(RuntimeError):
    pass


def _load_env():
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = _load_env()
BUDGET_USD = float(ENV["BUDGET_USD"])

_lock = threading.Lock()
_local = threading.local()


def _db():
    if not hasattr(_local, "db"):
        CACHE_DB.parent.mkdir(exist_ok=True)
        db = sqlite3.connect(CACHE_DB, timeout=60)
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, request TEXT, response TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS spend (ts REAL, provider TEXT, tag TEXT, prompt INT, "
                   "cached INT, completion INT, reasoning INT, usd REAL)")
        db.commit()
        _local.db = db
    return _local.db


def total_spend():
    return _db().execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]


def usage_cost(usage, provider):
    if usage.get("cost") is not None:  # OpenRouter reports the billed amount directly
        return float(usage["cost"])
    p_in, p_hit, p_out = PROVIDERS[provider]["price"]
    prompt = usage.get("prompt_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0)
    return ((prompt - cached) * p_in + cached * p_hit + completion * p_out) / 1e6


def write_costs_md():
    """Rewrite the auto-generated running-total block at the top of notes/costs.md."""
    rows = _db().execute("SELECT tag, COUNT(*), SUM(prompt), SUM(cached), SUM(completion), SUM(reasoning), "
                         "SUM(usd) FROM spend GROUP BY tag ORDER BY MIN(ts)").fetchall()
    total = sum(r[6] for r in rows)
    lines = ["<!-- AUTO:BEGIN (rewritten by scripts/api.py; do not edit) -->",
             f"## Running total: ${total:.4f} of ${BUDGET_USD:.2f} budget", "",
             "| tag | paid calls | prompt tok | cached tok | completion tok | reasoning tok | USD |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {t} | {n} | {p} | {c} | {o} | {r} | {u:.4f} |" for t, n, p, c, o, r, u in rows]
    lines.append("<!-- AUTO:END -->")
    block = "\n".join(lines)
    old = COSTS_MD.read_text() if COSTS_MD.exists() else "# Costs\n\n"
    if "<!-- AUTO:BEGIN" in old:
        pre, rest = old.split("<!-- AUTO:BEGIN", 1)
        post = rest.split("<!-- AUTO:END -->", 1)[1]
        new = pre + block + post
    else:
        new = old.rstrip("\n") + "\n\n" + block + "\n"
    COSTS_MD.parent.mkdir(exist_ok=True)
    COSTS_MD.write_text(new)


def chat(messages, temperature=0.0, max_tokens=10, seed=None, n=1, extra=None, sample=0,
         provider=DEFAULT_PROVIDER, model=None, tag="misc", thinking=False):
    """One chat completion. Returns the raw response JSON plus `_cached` (bool)."""
    cfg = PROVIDERS[provider]
    body = {"model": model or cfg["model"], "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "n": n}
    if seed is not None:
        body["seed"] = seed
    if not thinking:
        body.update(cfg["no_thinking"])
    body.update(cfg.get("extra", {}))
    body.update(extra or {})
    request = {"provider": provider, "sample": sample, **body}
    key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    row = _db().execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    if row:
        return {**json.loads(row[0]), "_cached": True}

    if total_spend() >= BUDGET_USD:
        raise BudgetExceeded(f"spend ${total_spend():.4f} >= BUDGET_USD ${BUDGET_USD:.2f}")

    headers = {"Authorization": f"Bearer {ENV[cfg['key_env']]}", "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(cfg["url"], headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                break
            if r.status_code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"API error {r.status_code}: {r.text[:500]}")
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
        time.sleep(min(60, 2 ** attempt))
    else:
        raise RuntimeError(f"API failed after {MAX_RETRIES} retries: {r.status_code} {r.text[:300]}")

    resp = r.json()
    usage = resp.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    with _lock:
        db = _db()
        db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                   (key, json.dumps(request, ensure_ascii=False), json.dumps(resp, ensure_ascii=False)))
        db.execute("INSERT INTO spend VALUES (?,?,?,?,?,?,?,?)",
                   (time.time(), provider, tag, usage.get("prompt_tokens", 0), cached,
                    usage.get("completion_tokens", 0), reasoning, usage_cost(usage, provider)))
        db.commit()
    return {**resp, "_cached": False}


def chat_many(requests_list, concurrency=MAX_CONCURRENCY, projected_usd=None, **common):
    """Run many chat() calls concurrently, preserving order. Each item is a dict of chat() kwargs.

    If `projected_usd` is given, refuse to start when it would take spend past BUDGET_USD.
    """
    if projected_usd is not None and total_spend() + projected_usd > BUDGET_USD:
        raise BudgetExceeded(f"projected ${projected_usd:.2f} + spent ${total_spend():.2f} "
                             f"> BUDGET_USD ${BUDGET_USD:.2f}")
    try:
        with ThreadPoolExecutor(min(concurrency, MAX_CONCURRENCY)) as ex:
            return list(ex.map(lambda kw: chat(**{**common, **kw}), requests_list))
    finally:
        write_costs_md()


def text_of(resp, i=0):
    return resp["choices"][i]["message"].get("content") or ""


def has_reasoning(resp):
    """True if the response shows any sign of hidden reasoning."""
    usage = resp.get("usage") or {}
    tok = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
    return tok > 0 or any(c["message"].get("reasoning_content") or c["message"].get("reasoning")
                          or c["message"].get("reasoning_details") for c in resp["choices"])
