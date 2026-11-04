"""Positions: King to Sub-Kooz, ties shared, and the set records built from them."""

import pytest
from estimation.arena.duplicate import SetResult

from arena.positions import position_reward, position_shares, summarize


@pytest.mark.parametrize(
    "scores, seat, shares",
    [
        ((50, 10, 0, -5), 0, [1, 0, 0, 0]),
        ((50, 10, 0, -5), 3, [0, 0, 0, 1]),
        ((50, 50, 0, -5), 1, [0.5, 0.5, 0, 0]),
        ((50, 7, 7, -5), 2, [0, 0.5, 0.5, 0]),
        ((7, 7, 7, 7), 0, [0.25, 0.25, 0.25, 0.25]),
    ],
)
def test_shares(scores, seat, shares):
    assert position_shares(scores, seat) == pytest.approx(shares)


@pytest.mark.parametrize("scores", [(50, 10, 0, -5), (50, 50, 0, -5), (7, 7, 7, -5), (7, 7, 7, 7)])
def test_rewards_sum_to_zero(scores):
    assert sum(position_reward(scores, s) for s in range(4)) == pytest.approx(0)


def test_summary_tracks_each_bot_through_the_rotations():
    # Lineup A B A B. Seat 0 is highest in every rotation, seat 3 lowest; rotations swap who sits
    # there, so across a set both bots hold every position equally and the set is drawn.
    scores = ((40, 30, 20, 10),) * 4
    result = SetResult(0, 0, scores, (18,) * 4)
    summary, pairs = summarize([result], ("A", "B", "A", "B"))
    assert summary["A"]["reward"]["mean"] == pytest.approx(0)
    assert sum(summary["A"]["shares"]) == pytest.approx(1)
    assert (pairs[0]["a_wins"], pairs[0]["draws"], pairs[0]["b_wins"]) == (0, 1, 0)


def test_the_better_placed_bot_wins_the_set():
    # A's seats finish first and third in rotation 0 (seats 0, 2), and so on: A always leads.
    scores = ((40, 10, 30, 0), (0, 40, 10, 30), (30, 0, 40, 10), (10, 30, 0, 40))
    summary, pairs = summarize([SetResult(0, 0, scores, (18,) * 4)], ("A", "B", "A", "B"))
    assert summary["A"]["reward"]["mean"] > 0 > summary["B"]["reward"]["mean"]
    assert (pairs[0]["a"], pairs[0]["a_wins"], pairs[0]["b_wins"]) == ("A", 1, 0)
