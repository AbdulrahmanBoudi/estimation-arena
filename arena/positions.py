"""Final positions in a Bola: King, Sub-King, Sub-Kooz, Kooz (last), the measure the leaderboard ranks by.

A game's position reward is King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1: the two ends count most. Tied seats share
the positions they occupy, so the four rewards always sum to 0. A set goes to the bot with the
higher mean position reward over its seats and rotations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from estimation.arena.duplicate import SetResult, slot_at

NAMES = ("King", "Sub-King", "Sub-Kooz", "Kooz")
REWARDS = (1.0, 0.2, -0.2, -1.0)


def position_shares(scores: Sequence[int], seat: int) -> list[float]:
    """The share of each of the four positions ``seat`` holds, ties split evenly."""
    above = sum(s > scores[seat] for s in scores)
    tied = sum(s == scores[seat] for s in scores)
    return [1 / tied if above <= p < above + tied else 0.0 for p in range(4)]


def position_reward(scores: Sequence[int], seat: int) -> float:
    return sum(share * r for share, r in zip(position_shares(scores, seat), REWARDS))


def _per_set(result: SetResult, lineup: Sequence[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {b: {"shares": [0.0] * 4, "games": 0} for b in dict.fromkeys(lineup)}
    for rotation, scores in enumerate(result.scores):
        for seat in range(4):
            side = out[lineup[slot_at(seat, rotation)]]
            side["shares"] = [a + b for a, b in zip(side["shares"], position_shares(scores, seat))]
            side["games"] += 1
    for side in out.values():
        side["shares"] = [x / side["games"] for x in side["shares"]]
        side["reward"] = sum(s * r for s, r in zip(side["shares"], REWARDS))
    return out


def summarize(results: Iterable[SetResult], lineup: Sequence[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Each bot's position shares and mean reward with its 95% interval over sets, and the
    set-by-set pair records by position reward."""
    sets = [_per_set(r, lineup) for r in results]
    bots = list(dict.fromkeys(lineup))
    summary: dict[str, dict[str, Any]] = {}
    for b in bots:
        rewards = [s[b]["reward"] for s in sets]
        n = len(rewards)
        mean = sum(rewards) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in rewards) / (n - 1)) if n > 1 else 0.0
        half = 1.96 * sd / math.sqrt(n)
        summary[b] = {
            "shares": [sum(s[b]["shares"][p] for s in sets) / n for p in range(4)],
            "reward": {"mean": mean, "low": mean - half, "high": mean + half, "sets": n},
        }
    pairs = []
    for i, a in enumerate(bots):
        for b in bots[i + 1 :]:
            wins = [0, 0, 0]
            for s in sets:
                diff = s[a]["reward"] - s[b]["reward"]
                wins[0 if diff > 1e-9 else 2 if diff < -1e-9 else 1] += 1
            pairs.append({"a": a, "b": b, "a_wins": wins[0], "draws": wins[1], "b_wins": wins[2]})
    return summary, pairs
