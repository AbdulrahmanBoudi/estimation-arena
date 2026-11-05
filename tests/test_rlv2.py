"""RL v2 on the GPU must match the engine's at every step (a quick slice of jaxenv.check_rlv2)."""

from pathlib import Path

import pytest

pytest.importorskip("jax")

from arena.bots import ROOT  # noqa: E402
from jaxenv import rlv2 as V  # noqa: E402
from jaxenv.check_rlv2 import run  # noqa: E402


def test_inputs_values_and_moves_match_the_engine():
    checkpoint, net = ROOT / "runs/m4-rules-v2/checkpoints/final.pt", ROOT / "runs/m4-rules-v2/rlv2.npz"
    if not (checkpoint.exists() and net.exists()):
        pytest.skip("needs the models: python -m arena.models fetch")
    from estimation.rl.networks import load_policy

    stats = run(2, 21, load_policy(Path(checkpoint)), V.load(net))
    assert stats["multi"] > 1_000
