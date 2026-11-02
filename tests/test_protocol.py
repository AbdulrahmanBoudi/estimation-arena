"""One game through the wire protocol and one through the Python tier, each against random."""
from estimation.arena.match import play_bola
from estimation.engine.config import DEFAULT_CONFIG

from arena.bots import load_manifest, make_submission, maker
from arena.protocol import WireBot, observation_to_dict
from estimation.engine import Bola


def test_observation_is_json_friendly():
    import json

    bola = Bola(1, DEFAULT_CONFIG)
    data = observation_to_dict(bola.observation(bola.current_player))
    json.dumps(data)
    assert data["phase_name"] in {"DASH", "AUCTION", "ESTIMATION", "PLAY"}
    assert len(data["hand_cards"]) == 13


def test_wire_bot_plays_a_full_game():
    bots = [make_submission(load_manifest("lowcard-wire"), s, DEFAULT_CONFIG) for s in range(2)]
    bots += [maker("random", 2, DEFAULT_CONFIG), maker("random", 3, DEFAULT_CONFIG)]
    assert isinstance(bots[0], WireBot)
    record = play_bola(7, bots, DEFAULT_CONFIG)
    assert record.rounds >= 18
    for bot in bots[:2]:
        bot.close()


def test_python_bot_plays_a_full_game():
    bots = [maker("sub:cautious-heuristic", s, DEFAULT_CONFIG) for s in range(2)]
    bots += [maker("random", 2, DEFAULT_CONFIG), maker("random", 3, DEFAULT_CONFIG)]
    record = play_bola(7, bots, DEFAULT_CONFIG)
    assert record.rounds >= 18
