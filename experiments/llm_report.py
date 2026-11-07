"""What an LLM log cost and how reliable it was.

    python3 experiments/llm_report.py game.jsonl

Dollar figures use Claude API list prices in USD per million tokens. Codex has none here because it
runs on a subscription.
"""

import json
import sys
from collections import Counter, defaultdict

PRICES = {"claude-opus-5-5": {"input": 4.0, "write": 5.0, "read": 0.20, "output": 20.0}}


def cost_of(row: dict) -> float:
    prices = PRICES.get(row.get("model", ""))
    usage = row.get("usage") or {}
    if not prices:
        return 0.0
    return (
        usage.get("input_tokens", 0) * prices["input"]
        + usage.get("cache_creation_input_tokens", 0) * prices["write"]
        + usage.get("cache_read_input_tokens", 0) * prices["read"]
        + usage.get("output_tokens", 0) * prices["output"]
    ) / 1e6


def summarize(rows: list[dict]) -> dict:
    out = {}
    by_bot = defaultdict(list)
    for row in rows:
        by_bot[row["bot"]].append(row)
    for bot, mine in sorted(by_bot.items()):
        called = [r for r in mine if r.get("seconds") is not None]
        models = Counter(r["model"] for r in called if r.get("model"))
        out[bot] = {
            "statuses": dict(Counter(r["status"] for r in mine)),
            "retried_moves": sum(r.get("tries", 1) > 1 for r in mine),
            "mean_s": round(sum(r["seconds"] for r in called) / len(called), 1) if called else None,
            "max_s": max((r["seconds"] for r in called), default=None),
            "models": dict(models),
            "tokens": {k: sum((r.get("usage") or {}).get(k) or 0 for r in called) for k in sorted({k for r in called for k in (r.get("usage") or {})})},
            "cost_usd": round(sum(cost_of(r) for r in called), 3),
            "errors": [e[:100] for r in mine for e in (r.get("errors") or [])][-3:],
        }
    return out


if __name__ == "__main__":
    print(json.dumps(summarize([json.loads(line) for line in open(sys.argv[1])]), indent=1))
