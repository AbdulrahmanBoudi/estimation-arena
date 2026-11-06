"""Hybrid v4, Yosif's strongest bot (estimation-engine, Hybrid iteration plan step 3).

One checkpoint holds three nets, trained by self-play and then by copying a deeper search
(expert iteration): BiddingNet picks the bid, CardPlayNet keeps its top cards, and GuessNet
guesses who holds each unseen card. For each card decision the bot imagines hidden hands placed
by GuessNet's guesses, keeps those that best explain everyone's bids, and plays each kept card
out with CardPlayNet for all four seats.

Options: ``checkpoint`` (path from the repo root) and any HybridConfig field, e.g.
``sub:hybrid-v4:samples=16``.
"""

from pathlib import Path

from estimation.bots.hybrid.bot import HybridBot, HybridConfig
from estimation.bots.registry import configure
from estimation.engine import RulesConfig

ROOT = Path(__file__).resolve().parents[2]
V4 = {"deals": "both", "top_k": "2", "samples": "32"}  # Hybrid v4's search on v3's nets


def make(
    seed: int, rules: RulesConfig, checkpoint: str = "runs/expert/round-1/hybrid.pt", **options: str
) -> HybridBot:
    path = Path(checkpoint)
    config = HybridConfig(rules=rules, checkpoint=str(path if path.is_absolute() else ROOT / path))
    return HybridBot(seed, configure(config, {**V4, **options}, "hybrid-v4"))
