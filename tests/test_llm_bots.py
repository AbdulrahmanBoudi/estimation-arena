import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from arena.bots import list_submissions, load_manifest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llm_play", ROOT / "experiments" / "bots" / "llm_play.py")
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
report_spec = importlib.util.spec_from_file_location("llm_report", ROOT / "experiments" / "llm_report.py")
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)

OBS = {"seat": 1, "round_no": 3, "phase_name": "PLAY", "scores": [0, 0, 0, 0], "carry": 1, "hand_cards": [0, 51],
       "fixed_trump": None, "trump": 3, "void_announced": [], "dash_callers": [], "log": [[0, 7]], "top_bid": 7,
       "top_bidder": 0, "caller": 0, "with_players": [], "estimates": [5, 2, None, None], "over": None, "total": None,
       "risks": None, "tricks": [], "tricks_won": [0, 0, 0, 0], "trick": [[0, 12]]}  # fmt: skip


NAMES = ("llm-claude", "llm-codex", "llm-claude-api", "llm-codex-sdk")


def test_llm_bots_are_never_rated():
    assert not set(NAMES) & set(list_submissions())
    for name in NAMES:
        assert load_manifest(name).timeout_s > llm.TRIES * llm.CALL_TIMEOUT_S


def _fake_claude(stop_reason="end_turn", text="thinking it over\n70", category=None):
    usage = SimpleNamespace(input_tokens=90, output_tokens=40, cache_read_input_tokens=600, cache_creation_input_tokens=0)
    response = SimpleNamespace(stop_reason=stop_reason, stop_details=SimpleNamespace(category=category), model="claude-opus-5-5",
                               content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)], usage=usage)  # fmt: skip
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))), calls


def test_the_api_player_caches_the_rules_and_sends_only_this_seats_view():
    client, calls = _fake_claude()
    reply, extra = llm.ask_claude_api("the seat's view", client)
    sent = calls[0]
    assert reply == "thinking it over\n70" and extra["model"] == "claude-opus-5-5"
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"][0]["text"] == llm.RULES and sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"] == [{"role": "user", "content": "the seat's view"}]
    assert sent["thinking"] == {"type": "adaptive"} and sent["output_config"]["effort"] == "medium"
    assert "tools" not in sent


def test_the_api_player_treats_a_refusal_as_a_failed_call():
    client, _ = _fake_claude(stop_reason="refusal", category="cyber")
    with pytest.raises(RuntimeError, match="refused: cyber"):
        llm.ask_claude_api("view", client)


def test_a_codex_reply_that_used_a_tool_is_refused():
    item = lambda kind: SimpleNamespace(root=SimpleNamespace(type=kind))  # noqa: E731
    last = SimpleNamespace(input_tokens=9, cached_input_tokens=1, output_tokens=2, reasoning_output_tokens=1)
    good = SimpleNamespace(items=[item("userMessage"), item("reasoning"), item("agentMessage")], final_response="70",
                           usage=SimpleNamespace(last=last))  # fmt: skip
    assert llm.codex_reply(good)[0] == "70"
    bad = SimpleNamespace(items=[item("userMessage"), item("commandExecution")], final_response="70", usage=None)
    with pytest.raises(RuntimeError, match="commandExecution"):
        llm.codex_reply(bad)


def test_the_report_prices_opus_and_counts_problems():
    rows = [
        {"bot": "claude-api", "status": "ok", "tries": 2, "seconds": 3.0, "model": "claude-opus-5-5", "errors": ["down"],
         "usage": {"input_tokens": 1000, "output_tokens": 1000, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 1000}},
        {"bot": "claude-api", "status": "forced"},
    ]  # fmt: skip
    summary = report.summarize(rows)["claude-api"]
    assert summary["cost_usd"] == round((4 + 20 + 0.2 + 5) * 1000 / 1e6, 3)
    assert summary["retried_moves"] == 1 and summary["statuses"] == {"ok": 1, "forced": 1}


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
