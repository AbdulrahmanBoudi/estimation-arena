import importlib.util
import json
import sys
from pathlib import Path

from estimation.arena.duplicate import play_set

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_eval_set", ROOT / "scripts" / "build-eval-set.py")
builder = importlib.util.module_from_spec(spec)
sys.modules["build_eval_set"] = builder
spec.loader.exec_module(builder)


def test_a_game_replays_and_holds_every_deal() -> None:
    result, _ = play_set(("heuristic",) * 4, builder.PRACTICE_SEED, 0, keep_games=True)
    row = builder.row(0, result.games[0])
    assert json.loads(json.dumps(row)) == row
    assert row["seats"] == ["Heuristic"] * 4
    assert row["deals"][0]["round"] == 1
    assert all(len({card for hand in deal["hands"] for card in hand}) == 52 for deal in row["deals"])


def test_the_practice_seed_is_not_a_published_ladder_seed() -> None:
    seeds = {json.loads(path.read_text())["seed"] for path in (ROOT / "results").glob("*.json")}
    assert builder.PRACTICE_SEED not in seeds
