"""Rebuild a watchable game from an LLM match's move log, without paying for another game.

    python -m experiments.llm_game_from_log --log match.jsonl --bot llm-claude-api --opponent heuristic \
        --seed 2000 --result results/early/llm-claude-api.json

The log holds the LLM's moves, and the seeds give the deals and the opponent's moves. It writes the
first game of the set to site/games, and refuses unless it reproduces that game's real final scores.
Only for practice seeds: it publishes the cards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from estimation.arena.duplicate import bot_seed, deal_seed, slot_at
from estimation.engine.config import DEFAULT_CONFIG

from arena.bots import SUB_PREFIX, load_manifest, maker
from arena.direct import game_file
from arena.positions import decode_scores
from arena.replay import GAMES_DIR, record_game
from arena.leaderboard import SITE_DIR
from arena.run import bot_hash, engine_version


class Logged:
    """Plays back this seat's moves from the log, checking each was made in the same position."""

    def __init__(self, rows: list[dict], seat: int) -> None:
        self.rows, self.seat = rows, seat

    def act(self, view, legal: list[int]) -> int:
        row = self.rows.pop(0)
        if row["seat"] != self.seat or row["round"] != view.round_no or row["action"] not in legal:
            raise ValueError(f"the log doesn't match the game at round {view.round_no}, seat {self.seat}")
        return row["action"]


def rebuild(rows: list[dict], bot: str, opponent: str, seed: int, rotation: int = 0, index: int = 0) -> dict:
    me = SUB_PREFIX + bot
    line = (me, opponent, me, opponent)
    seats = [line[slot_at(seat, rotation)] for seat in range(4)]
    bots = [Logged(rows, seat) if seats[seat] == me else maker(seats[seat], bot_seed(seed, index, rotation, seat), DEFAULT_CONFIG)
            for seat in range(4)]  # fmt: skip
    return record_game(seats, deal_seed(seed, index), bots=bots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--bot", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--opponent-name", default="Heuristic")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SITE_DIR)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in args.log.read_text().splitlines()]
    game = rebuild(rows, args.bot, args.opponent, args.seed)
    result = json.loads(args.result.read_text())
    assert result["seed"] == args.seed
    versus = next(v for v in result["versus"] if v["spec"] == args.opponent)
    real = list(decode_scores(versus["scores"])[0].scores[0])
    if game["final"] != real:
        raise SystemExit(f"rebuilt scores {game['final']} differ from the real {real}")
    game |= {"engine": engine_version(), "versions": {SUB_PREFIX + args.bot: bot_hash(load_manifest(args.bot)), args.opponent: args.opponent}}
    path = args.out / GAMES_DIR / game_file(args.bot, args.opponent_name)
    path.write_text(json.dumps(game, separators=(",", ":")) + "\n")
    print(path, game["final"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
