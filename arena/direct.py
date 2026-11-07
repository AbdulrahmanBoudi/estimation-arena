"""Direct games: every rated bot against every other, and what they say about the standings.

Benchmarks alone can't rank the bots against each other, because skill in a card game need not
be transitive. docs/fair-ranking.md gives the reasoning and its sources. Reading these files
needs no engine; arena.duels plays them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from math import comb
from pathlib import Path
from typing import Any

DIRECT_DIR = Path("results") / "direct"
SIGNIFICANCE = 0.05


def match_file(root: Path, a: str, b: str) -> Path:
    """One file per pair of bots, whichever way round they are named."""
    first, second = sorted((a, b))
    return root / DIRECT_DIR / f"{first}.vs.{second}.json"


def load_matches(root: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((root / DIRECT_DIR).glob("*.vs.*.json"))]


def current(matches: Iterable[dict[str, Any]], results: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The matches both of whose bots are still as they were: a changed bot or engine voids them."""
    rated = {r["bot"]: r for r in results}
    kept = []
    for match in matches:
        bots = (rated.get(match["a"]), rated.get(match["b"]))
        if all(r and r.get("engine") == match["engine"] for r in bots) and (
            bots[0]["bot_hash"], bots[1]["bot_hash"]
        ) == (match["hash_a"], match["hash_b"]):  # fmt: skip
            kept.append(match)
    return kept


def record(match: dict[str, Any]) -> tuple[int, int, int]:
    """Sets won by ``a``, drawn, and won by ``b``, under weighted rank."""
    a = f"sub:{match['a']}"
    wins = draws = losses = 0
    for pair in match["position_pairs"]:
        mine, theirs = (pair["a_wins"], pair["b_wins"]) if pair["a"] == a else (pair["b_wins"], pair["a_wins"])
        wins, draws, losses = wins + mine, draws + pair["draws"], losses + theirs
    return wins, draws, losses


def is_clear(wins: int, losses: int) -> bool:
    """Whether a result is unlikely between equal bots: a two-sided sign test, draws set aside."""
    played = wins + losses
    if not played:
        return False
    tail = sum(comb(played, k) for k in range(min(wins, losses) + 1)) / 2**played
    return min(1.0, 2 * tail) < SIGNIFICANCE


def table(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The matches for a page: who, the record, and whether it is clear."""
    rows = []
    for match in matches:
        wins, draws, losses = record(match)
        rows.append({"a": f"sub:{match['a']}", "b": f"sub:{match['b']}", "a_wins": wins, "draws": draws,
                     "b_wins": losses, "sets": wins + draws + losses, "clear": is_clear(wins, losses)})  # fmt: skip
    return rows


def runoff(keys: Sequence[str], pairs: Iterable[Any]) -> list[str]:
    """Bots from winner to first out: repeatedly drop the one scoring least against those left.

    A bot can't rise by beating up weak bots, because weak bots leave first. ``keys`` must be in
    standings order, which breaks ties: the lower of two equal bots goes first.
    """
    alive = list(keys)
    records = list(pairs)
    out = []
    while len(alive) > 1:
        points = dict.fromkeys(alive, 0.0)
        sets = dict.fromkeys(alive, 0)
        for p in records:
            if p.a in points and p.b in points:
                total = p.a_wins + p.draws + p.b_wins
                points[p.a] += p.a_wins + p.draws / 2
                points[p.b] += p.b_wins + p.draws / 2
                sets[p.a] += total
                sets[p.b] += total
        scores = {k: points[k] / sets[k] if sets[k] else 0.0 for k in alive}
        lowest = min(scores.values())
        loser = next(k for k in reversed(alive) if scores[k] == lowest)
        alive.remove(loser)
        out.append(loser)
    return alive + out[::-1]
