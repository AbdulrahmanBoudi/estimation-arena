"""One game between LLM bots, on practice deals: never the ladder seed.

    python -m experiments.llm_match --out result.json --rounds 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from estimation.engine.bola import Bola
from estimation.engine.config import DEFAULT_CONFIG

from arena.bots import maker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seats", default="sub:llm-claude,sub:llm-codex,sub:llm-claude,sub:llm-codex")
    parser.add_argument("--seed", type=int, default=2000, help="practice deals")
    parser.add_argument("--rounds", type=int, default=0, help="stop after this many rounds (0: the whole game)")
    args = parser.parse_args(argv)
    seats = args.seats.split(",")
    assert len(seats) == 4
    bots = [maker(spec, args.seed + seat, DEFAULT_CONFIG) for seat, spec in enumerate(seats)]
    game = Bola(args.seed, DEFAULT_CONFIG)
    rounds, last = [], 1
    while not game.is_over:
        view = game.observation(game.current_player)
        if view.round_no != last:
            rounds.append({"round": last, "scores": list(view.scores)})
            print(rounds[-1], flush=True)
            args.out.write_text(json.dumps({"seats": seats, "seed": args.seed, "rounds": rounds}, indent=1))
            if args.rounds and last >= args.rounds:
                break
            last = view.round_no
        game.step(bots[game.current_player].act(view, game.legal_actions()))
    final = list(game.observation(0).scores)
    args.out.write_text(json.dumps({"seats": seats, "seed": args.seed, "rounds": rounds, "final": final, "finished": game.is_over}, indent=1))
    print("final", final, "finished", game.is_over)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
