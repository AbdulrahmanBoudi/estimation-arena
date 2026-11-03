"""Export one match game, net-pimc vs PIMC, as JSON for the training board's replay.

    PYTHONPATH=. uv run python experiments/trace_game.py --seed 20000123 --rounds 3 --out trace.json

Seats 0 and 2 play net-pimc, seats 1 and 3 PIMC (the duplicate lineup). Every card decision by
seat 0 in the first ``--rounds`` rounds records the whole table (each seat's own view, a
referee's picture), the net's card probabilities, the 3 cards the search kept, their search
values, and PIMC-100's value of every legal card for reference. Round scores cover the game.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401
from estimation.bots import PIMCBot, PIMCConfig
from estimation.bots.pimc import best_index
from estimation.engine import Bola, Phase, RulesConfig, cards_of
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode

from arena.bots import load_manifest, make_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20_000_123)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rules = RulesConfig()
    me = make_submission(load_manifest("net-pimc"), a.seed, rules)
    mate = make_submission(load_manifest("net-pimc"), a.seed + 2, rules)
    rivals = {1: PIMCBot(a.seed + 1), 3: PIMCBot(a.seed + 3)}
    bots = {0: me, 2: mate, **rivals}
    teacher = PIMCBot(a.seed + 9, PIMCConfig(samples=100))
    bola = Bola(a.seed)
    steps, rounds = [], []
    while not bola.is_over:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        if seat == 0 and obs.phase is Phase.PLAY and len(legal) > 1 and bola.round_no <= a.rounds:
            cards = [x - PLAY_BASE for x in legal]
            x = encode(obs, rules).astype(np.float32)
            h = x
            for w, b in me.layers[:-1]:
                h = np.maximum(h @ w + b, 0.0)
            w, b = me.layers[-1]
            z = h @ w + b
            z = np.where(np.isin(np.arange(52), cards), z, -np.inf)
            prob = np.exp(z - z.max())
            prob /= prob.sum()
            top = sorted(cards, key=lambda c: -prob[c])[: me.top]
            search = me.evaluate(obs, top, me.sample_deals(obs))
            chosen = top[best_index(search)]
            cand, totals = teacher.search(obs, cards)
            ref = {int(c): t / 100 for c, t in zip(cand, totals)}
            views = [bola.observation(s) for s in range(4)]
            steps.append({
                "round": bola.round_no, "trump": obs.trump, "fixed": obs.fixed_trump is not None,
                "estimates": list(obs.estimates), "won": list(obs.tricks_won),
                "hands": [cards_of(v.hand) for v in views],
                "trick": [[s, c] for s, c in obs.trick],
                "played": [c for t in obs.tricks for c in t.cards],
                "legal": cards, "prob": [float(prob[c]) for c in cards],
                "top": top, "search": [t / me.config.samples for t in search],
                "teacher": [ref.get(c, 0.0) for c in cards], "chosen": chosen,
            })
            bola.step(PLAY_BASE + chosen)
        else:
            before = bola.round_no
            bola.step(bots[seat].act(obs, legal))
            if bola.round_no != before or bola.is_over:
                rounds.append(list(bola.scores))
    a.out.write_text(json.dumps({"seed": a.seed, "seats": ["net-pimc", "PIMC", "net-pimc", "PIMC"],
                                 "steps": steps, "scores": rounds}))
    print(f"wrote {a.out}: {len(steps)} decisions, {len(rounds)} rounds")


if __name__ == "__main__":
    main()
