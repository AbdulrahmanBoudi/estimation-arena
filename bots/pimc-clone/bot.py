"""Rung 2: card play by a net that imitates a deep PIMC search (train/imitate.py).

Bidding, Dash Calls and estimates are HeuristicBot's, the same as PIMC's, so against PIMC any
difference comes from card play. A card decision is one forward pass of a small MLP in NumPy
(no PyTorch needed): encode the Observation, score the 52 cards, play the best legal one.

Uses net-pimc's net by default. Options: ``sub:pimc-clone:net=other.npz`` (relative to this folder).
"""

from functools import cache
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from arena.nets import fit_inputs
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig
from estimation.engine import Observation, Phase, RulesConfig, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.rl.encoder import encode

HERE = Path(__file__).resolve().parent


@cache
def load_net(path: str) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    data = np.load(HERE / path)
    layers = []
    k = 0
    while f"w{k}" in data:
        layers.append((data[f"w{k}"].astype(np.float32), data[f"b{k}"].astype(np.float32)))
        k += 1
    layers[0] = (fit_inputs(layers[0][0]), layers[0][1])  # nets from encoder v2 still load
    return layers, str(data["meta"])


def card_scores(layers: list[tuple[np.ndarray, np.ndarray]], x: np.ndarray) -> np.ndarray:
    for w, b in layers[:-1]:
        x = np.maximum(x @ w + b, 0.0)
    w, b = layers[-1]
    return x @ w + b


class CloneBot(HeuristicBot):
    __slots__ = ("layers", "rules")

    def __init__(self, config: HeuristicConfig, rules: RulesConfig, net: str) -> None:
        super().__init__(config)
        self.rules = rules
        self.layers, _ = load_net(net)

    def act(self, obs: Observation, legal: list[int]) -> int:
        if obs.phase is not Phase.PLAY:
            return super().act(obs, legal)
        if len(legal) == 1:
            return legal[0]
        scores = card_scores(self.layers, encode(obs, self.rules).astype(np.float32))
        cards = [a - PLAY_BASE for a in legal]
        return play_action(max(cards, key=lambda c: scores[c]))


    def explain(self, obs: Observation, cards: list[int]) -> tuple[int, dict[str, dict[int, float]]]:
        z = card_scores(self.layers, encode(obs, self.rules).astype(np.float32))
        zl = np.array([z[c] for c in cards])
        prob = np.exp(zl - zl.max())
        prob /= prob.sum()
        return cards[int(np.argmax(zl))], {"net": dict(zip(cards, prob.tolist()))}


def make(seed: int, rules: RulesConfig, net: str = "../net-pimc/net.npz") -> CloneBot:
    return CloneBot(HeuristicConfig(super_call_min=rules.super_call_min), rules, net)
