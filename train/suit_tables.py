"""Export the engine's suit-symmetry index tables for train.imitate --augment.

    uv run python -m train.suit_tables --out train/suit_tables.npz

Card play is symmetric under relabelling the four suits (estimation.rl.symmetry), so every
recorded card decision stands for 24. The tables let the GPU environment apply that without
the engine: ``features[:, feature_source[p]]``, ``cards[:, card_source[p]]`` and
``card_destination[p][choice]`` for permutation ``p``.
"""

import argparse
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order)
from estimation.rl.encoder import ENCODER_VERSION
from estimation.rl.symmetry import SUIT_PERMUTATIONS, card_destination, card_source, feature_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("train/suit_tables.npz"))
    args = parser.parse_args()
    np.savez(
        args.out,
        feature_source=np.stack([feature_source(p) for p in SUIT_PERMUTATIONS]).astype(np.int64),
        card_source=np.stack([card_source(p) for p in SUIT_PERMUTATIONS]).astype(np.int64),
        card_destination=np.stack([card_destination(p) for p in SUIT_PERMUTATIONS]).astype(np.int64),
        encoder_version=ENCODER_VERSION,
    )
    print(f"wrote {args.out}: {len(SUIT_PERMUTATIONS)} permutations, encoder v{ENCODER_VERSION}")


if __name__ == "__main__":
    main()
