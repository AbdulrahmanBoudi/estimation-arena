import importlib.util
from pathlib import Path

from arena.bots import list_submissions, load_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llm_play", ROOT / "experiments" / "bots" / "llm_play.py")
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)

OBS = {"seat": 1, "round_no": 3, "phase_name": "PLAY", "scores": [0, 0, 0, 0], "carry": 1, "hand_cards": [0, 51],
       "fixed_trump": None, "trump": 3, "void_announced": [], "dash_callers": [], "log": [[0, 7]], "top_bid": 7,
       "top_bidder": 0, "caller": 0, "with_players": [], "estimates": [5, 2, None, None], "over": None, "total": None,
       "risks": None, "tricks": [], "tricks_won": [0, 0, 0, 0], "trick": [[0, 12]]}  # fmt: skip


def test_llm_bots_are_never_rated():
    assert not {"llm-claude", "llm-codex"} & set(list_submissions())
    assert load_manifest("llm-claude").timeout_s > llm.TRIES * llm.CALL_TIMEOUT_S - 1


def test_names_match_the_action_ids():
    assert llm.action_name(2) == "bid 4 clubs"
    assert llm.action_name(51) == "bid 13 no trump"
    assert llm.action_name(54) == "estimate 0"
    assert llm.action_name(68) == "play 2♣"
    assert llm.action_name(119) == "play A♠"


def test_the_prompt_lists_only_this_seats_view_and_the_legal_moves():
    text = llm.prompt_for(OBS, [68, 119])
    assert "Your hand: 2♣ A♠." in text
    assert "68 = play 2♣" in text and "119 = play A♠" in text
    assert "seat 0: A♣" in text


def test_a_reply_must_end_with_one_legal_number():
    assert llm.parse_reply("70", [70, 76]) == 70
    assert llm.parse_reply(" 70\n", [70, 76]) == 70
    assert llm.parse_reply("I hold 3 spades, so:\n**76**", [70, 76]) == 76
    assert llm.parse_reply("71", [70, 76]) is None
    assert llm.parse_reply("70 or 76", [70, 76]) is None
    assert llm.parse_reply("I will play 76 because it is safe", [70, 76]) is None
    assert llm.parse_reply("", [70, 76]) is None


def test_a_forced_move_costs_no_call():
    logged = []
    assert llm.decide(OBS, [68], lambda _: 1 / 0, logged.append) == 68
    assert logged[0]["status"] == "forced"


def test_a_bad_call_is_retried():
    replies = iter([RuntimeError("down"), ("nonsense", {}), ("119", {})])

    def ask(_):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    logged = []
    assert llm.decide(OBS, [68, 119], ask, logged.append) == 119
    assert logged[0]["status"] == "ok" and logged[0]["tries"] == 3 and len(logged[0]["errors"]) == 2


def test_it_falls_back_and_says_so():
    logged = []
    assert llm.decide(OBS, [68, 119], lambda _: ("no", {}), logged.append) == 68
    assert logged[0]["status"] == "fallback"
