"""A Python-tier submission: `make(seed, rules)` returns any object with `act(obs, legal)`.

The engine is importable here (`uv sync` installs it), so a submission can reuse its bots,
its hand evaluation or its PIMC search as building blocks.
"""
from estimation.bots.heuristic import HeuristicBot, HeuristicConfig
from estimation.engine import RulesConfig


def make(seed: int, rules: RulesConfig):
    config = HeuristicConfig(super_call_min=rules.super_call_min, risk_aversion=0.5)
    return HeuristicBot(config)
