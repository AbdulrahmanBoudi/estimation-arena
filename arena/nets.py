"""Nets trained on an older version of the engine's RL encoder, which grows by appended blocks.

Encoder v3 appended 20 inputs to v2's 456 (each player's last call in each trump, this round).
``fit_inputs`` gives a net's first layer zero weights for inputs appended since it was trained,
so it computes exactly what it computed on the encoder it was trained on.
"""

import numpy as np


def fit_inputs(w0: np.ndarray) -> np.ndarray:
    """``w0`` (inputs, hidden) with zero rows for the encoder inputs it was trained without."""
    from estimation.rl.encoder import NUM_FEATURES  # import order: the encoder imports the bots

    missing = NUM_FEATURES - w0.shape[0]
    if missing < 0:
        raise ValueError(f"a net with {w0.shape[0]} inputs; the encoder has only {NUM_FEATURES}")
    if missing == 0:
        return w0
    return np.concatenate([w0, np.zeros((missing, w0.shape[1]), w0.dtype)])
