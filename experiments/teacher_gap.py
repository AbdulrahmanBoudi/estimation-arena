"""How far is each card policy from the teacher? Regret under PIMC-100's values, same positions.

    PYTHONPATH=. uv run python experiments/teacher_gap.py --games 40 --workers 5

Plays PIMC-100 games and, at every card decision with a choice, also asks PIMC-30 (the
leaderboard's PIMC anchor) and HeuristicBot. A student must get its regret below PIMC-30's to
play better cards than the anchor.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import estimation.bots  # noqa: F401
from estimation.bots import HeuristicBot, PIMCBot, PIMCConfig
from estimation.bots.pimc import best_index
from estimation.engine import Bola, Phase, play_action
from estimation.engine.actions import PLAY_BASE


def game(g: int) -> list[tuple[float, float, int]]:
    seed = 20_000_000 + g
    teacher, anchor, h = PIMCBot(seed, PIMCConfig(samples=100)), PIMCBot(seed + 1), HeuristicBot()
    bola, out = Bola(seed), []
    while not bola.is_over:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        if obs.phase is Phase.PLAY and len(legal) > 1:
            cards = [a - PLAY_BASE for a in legal]
            cand, totals = teacher.search(obs, cards)
            value = dict(zip(cand, np.array(totals) / 100))
            best = cand[best_index(totals)]
            a_card = anchor.choose(obs, cards)
            h_card = h.act(obs, legal) - PLAY_BASE
            out.append((value[best] - value[a_card], value[best] - value[h_card], len(cards)))
            bola.step(play_action(best))
        else:
            bola.step(h.act(obs, legal))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--workers", type=int, default=5)
    a = p.parse_args()
    with ProcessPoolExecutor(a.workers) as pool:
        rows = [r for part in pool.map(game, range(a.games)) for r in part]
    r = np.array(rows)
    print(f"{len(r)} decisions from {a.games} games")
    print(f"PIMC-30 (anchor) regret:  {r[:, 0].mean():.3f} pts/decision")
    print(f"HeuristicBot regret:      {r[:, 1].mean():.3f} pts/decision")


if __name__ == "__main__":
    main()
