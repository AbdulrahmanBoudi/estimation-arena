"""Replay full games in the engine and on the GPU and require the same RL v2 at every step.

    .venv-jax/bin/python -m jaxenv.check_rlv2 --games 200 --net rlv2.npz

Every decision must agree on all 456 inputs and the chosen move, and the values must be close.
Half the games are RL v2 against itself; the rest use random and rule-based moves, which reach
rare positions. Games past int32's range (see jaxenv.game) stop being compared when flagged.
"""

import argparse
import random
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import estimation.bots  # noqa: F401
from estimation.engine import Bola, Phase
from estimation.rl.encoder import OFFSETS, encode as engine_encode
from estimation.rl.networks import load_policy

from . import game as G
from . import rlv2 as V
from .check import _chooser, bits

_new = jax.jit(jax.vmap(G.new_game))
_step = jax.jit(jax.vmap(G.game_step))
ATOL_INPUT, ATOL_VALUE = 1e-4, 2e-3


def _block(i: int) -> str:
    names = sorted(OFFSETS, key=OFFSETS.get)
    name = [n for n in names if OFFSETS[n] <= i][-1]
    return f"{name}+{i - OFFSETS[name]}"


def run(n: int, seed: int, policy, nets) -> dict:
    rng = random.Random(seed)
    encode = jax.jit(jax.vmap(V.encode))
    values = jax.jit(jax.vmap(lambda g, x: V.action_values(nets, g, x)))
    moves = jax.jit(jax.vmap(lambda g: V.act(nets, g)))
    bolas = [Bola(rng.randrange(2**31)) for _ in range(n)]
    choose = [None if i % 2 == 0 else _chooser(rng) for i in range(n)]
    g = _new(jnp.array(np.stack([bits(b.round.hands) for b in bolas])))
    done = np.zeros(n, bool)
    stats = {"decisions": 0, "multi": 0, "max_input": 0.0, "max_value": 0.0, "moves_differ": 0, "phases": {}}
    while True:
        xs, picks, flagged = (np.asarray(a) for a in (encode(g), moves(g), g.overflow))
        vals = np.asarray(values(g, jnp.asarray(xs)))
        live = [i for i, b in enumerate(bolas) if not b.is_over and not done[i]]
        if not live:
            return stats
        obs = {i: bolas[i].observation(bolas[i].current_player) for i in live}
        for i in live:
            if flagged[i]:
                done[i] = True
        live = [i for i in live if not done[i]]
        want_x = {i: engine_encode(obs[i]) for i in live}
        card = [i for i in live if obs[i].phase is Phase.PLAY]
        bid = [i for i in live if obs[i].phase is not Phase.PLAY]
        want_v = {}
        if card:
            out = policy.card_values(np.stack([want_x[i] for i in card]))
            want_v.update({i: np.concatenate([np.zeros(V.BID_ACTIONS), o]) for i, o in zip(card, out)})
        if bid:
            out = policy.bid_values(np.stack([want_x[i] for i in bid]))
            want_v.update({i: np.concatenate([o, np.zeros(V.CARDS)]) for i, o in zip(bid, out)})
        actions = np.zeros(n, np.int32)
        nxt = np.zeros((n, 4, 52), bool)
        for i in live:
            b = bolas[i]
            where = f"game {i} round {b.round_no} phase {obs[i].phase.name}"
            gap = np.abs(xs[i] - want_x[i])
            if gap.max() > ATOL_INPUT:
                bad = np.flatnonzero(gap > ATOL_INPUT)[:8].tolist()
                raise AssertionError(
                    f"{where}: inputs differ at {[_block(k) for k in bad]}: {xs[i][bad]} vs {want_x[i][bad]}")
            legal = b.legal_actions()
            if len(legal) > 1:
                gap_v = max(abs(float(vals[i][a]) - float(want_v[i][a])) for a in legal)
                if gap_v > ATOL_VALUE:
                    raise AssertionError(f"{where}: values differ by {gap_v}")
                stats["max_value"] = max(stats["max_value"], gap_v)
                stats["multi"] += 1
            stats["max_input"] = max(stats["max_input"], float(gap.max()))
            want = max(legal, key=lambda a: want_v[i][a])
            if int(picks[i]) != want:
                stats["moves_differ"] += 1
                raise AssertionError(f"{where}: move {int(picks[i])} vs {want}")
            phase = obs[i].phase.name
            stats["phases"][phase] = stats["phases"].get(phase, 0) + 1
            stats["decisions"] += 1
            a = want if choose[i] is None else choose[i](obs[i], legal)
            before = b.round_no
            b.step(a)
            actions[i] = a
            if not b.is_over and b.round_no != before:
                nxt[i] = bits(b.round.hands)
        g = _step(g, jnp.array(actions), jnp.array(nxt))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.check_rlv2", description=__doc__.split("\n")[0])
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path, default=Path("runs/m4-rules-v2/checkpoints/final.pt"))
    p.add_argument("--net", type=Path, default=Path("rlv2.npz"))
    a = p.parse_args(argv)
    t = time.perf_counter()
    s = run(a.games, a.seed, load_policy(a.checkpoint), V.load(a.net))
    print(f"PASS {a.games} games, {s['decisions']} decisions identical ({s['multi']} with a real choice), "
          f"{time.perf_counter() - t:.0f} s")
    print(f"  largest gap: inputs {s['max_input']:.2e}, values {s['max_value']:.2e}; by phase {s['phases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
