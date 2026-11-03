"""Sample moves for the leaderboard: every bot plays the same deal, a few of its card choices.

    uv run python -m arena.samples        # writes site/samples.json

Seat 0 is the bot, seats 1-3 HeuristicBot, deal ``SEED``. From the first round, three of seat
0's card decisions are kept (its first, a middle one, its last with a choice): its own view
(hand, trick so far, trump, estimate) and the card it played. Bots that can say why add scores
per card: ``explain(obs, cards)`` for submissions that define it, the search values for PIMC-type
anchors. Anchors whose files are missing (RL v2, Hybrid without runs/) are skipped, and samples
already in the file for them are kept.
"""

from __future__ import annotations

import json
from typing import Any

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot, PIMCBot
from estimation.bots.pimc import best_index
from estimation.engine import Bola, Phase, cards_of
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG

from .bots import ROOT, SUB_PREFIX, list_submissions, maker
from .run import load_anchors

SEED = 31_337
OUT = ROOT / "site" / "samples.json"


def sample(spec: str) -> dict[str, Any]:
    bot = maker(spec, SEED, DEFAULT_CONFIG)
    others = HeuristicBot()
    bola = Bola(SEED)
    frames: list[dict[str, Any]] = []
    while not bola.is_over and bola.round_no == 1:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        if seat != 0:
            bola.step(others.act(obs, legal))
            continue
        if obs.phase is not Phase.PLAY or len(legal) == 1:
            bola.step(bot.act(obs, legal))
            continue
        cards = [a - PLAY_BASE for a in legal]
        scores: dict[str, dict[int, float]] = {}
        if hasattr(bot, "explain"):
            chosen, scores = bot.explain(obs, cards)
        elif isinstance(bot, PIMCBot):
            cand, totals = bot.search(obs, cards)
            chosen = cand[best_index(totals)]
            n = getattr(bot.config, "samples", 1)
            scores = {"search": {c: t / n for c, t in zip(cand, totals)}}
        else:
            chosen = bot.act(obs, legal) - PLAY_BASE
        frames.append({
            "trump": obs.trump,
            "estimate": obs.estimates[0],
            "won": obs.tricks_won[0],
            "hand": cards_of(obs.hand),
            "trick": [[s, c] for s, c in obs.trick],
            "legal": cards,
            "chosen": chosen,
            "scores": {k: {str(c): round(v, 3) for c, v in d.items()} for k, d in scores.items()},
        })
        bola.step(PLAY_BASE + chosen)
    if hasattr(bot, "close"):
        bot.close()
    pick = sorted({0, len(frames) // 2, len(frames) - 1}) if frames else []
    return {"seed": SEED, "frames": [frames[i] for i in pick]}


def main() -> int:
    old = json.loads(OUT.read_text()) if OUT.is_file() else {}
    specs = [SUB_PREFIX + n for n in list_submissions()] + [a["spec"] for a in load_anchors()]
    out = dict(old)
    for spec in specs:
        out[spec] = sample(spec)
        print(f"{spec}: {len(out[spec]['frames'])} moves")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
