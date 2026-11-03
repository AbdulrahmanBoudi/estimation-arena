"""record -> imitate -> play, end to end on a tiny teacher (a few seconds)."""

import numpy as np

from arena.bots import load_manifest, make_submission
from estimation.engine import Bola, RulesConfig
from train.imitate import main as imitate
from train.record import record_chunk


def test_record_train_play(tmp_path):
    games, rows = record_chunk(range(2), "pimc:samples=2", None, 1.0, tmp_path)
    assert games == 2 and rows > 500
    shard = np.load(next(tmp_path.glob("*.npz")))
    assert shard["features"].shape == (rows, shard["features"].shape[1])
    assert shard["legal"][np.arange(rows), shard["choice"]].all()  # the teacher plays legal cards

    net = tmp_path / "net.npz"
    imitate(["--data", str(tmp_path), "--out", str(net), "--epochs", "1", "--hidden", "32", "--val", "0.5"])

    bot = make_submission(load_manifest("pimc-clone"), 0, RulesConfig(), {"net": str(net)})
    bola = Bola(7)
    while not bola.is_over:
        seat = bola.current_player
        bola.step(bot.act(bola.observation(seat), bola.legal_actions()))  # raises if illegal
