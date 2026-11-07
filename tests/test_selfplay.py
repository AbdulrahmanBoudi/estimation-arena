"""The position reward and the per-seat TD(lambda) credit, checked by hand."""

import pytest

pytest.importorskip("optax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxenv import rules as R  # noqa: E402
from jaxenv.selfplay import bidding_only, init_nets, load_policy, make_rollout, position_rewards, save_npz, td_lambda  # noqa: E402
from train.copy_search import card_targets, regret, split_games  # noqa: E402


@pytest.mark.parametrize(
    "scores, want",
    [
        ((50, 10, 0, -5), (1, 0.2, -0.2, -1)),
        ((-5, 0, 10, 50), (-1, -0.2, 0.2, 1)),
        ((50, 50, 0, -5), (0.6, 0.6, -0.2, -1)),  # share King and Sub-King
        ((50, 7, 7, -5), (1, 0, 0, -1)),
        ((7, 7, 7, -5), (1 / 3, 1 / 3, 1 / 3, -1)),
        ((7, 7, 7, 7), (0, 0, 0, 0)),
    ],
)
def test_position_rewards(scores, want):
    got = position_rewards(jnp.array(scores))
    assert jnp.allclose(got, jnp.array(want), atol=1e-6)
    assert abs(float(got.sum())) < 1e-6


def run(values, seats, finished=None, overflow=None, reward=None, lam=0.5):
    t = len(values)
    return td_lambda(
        jnp.array(values, jnp.float32)[:, None],
        jnp.array(seats)[:, None],
        jnp.array(finished or [False] * t)[:, None],
        jnp.array(overflow or [False] * t)[:, None],
        jnp.zeros((t, 1, 4)) if reward is None else jnp.array(reward, jnp.float32)[:, None],
        lam,
    )


def test_each_seat_chains_on_its_own_next_decision():
    # seat 0 at t=0,2; seat 1 at t=1,3. Each seat's last decision has no successor.
    adv, has = run([0.1, 0.2, 0.4, 0.8], [0, 1, 0, 1])
    assert has[:, 0].tolist() == [True, True, False, False]
    assert jnp.allclose(adv[:, 0], jnp.array([0.3, 0.6, 0.0, 0.0]), atol=1e-6)


def test_game_end_gives_each_seat_its_reward_and_lambda_carries_back():
    # seat 0 moves at t=0 and t=1, then the game ends with seat 0 winning.
    r = [[0, 0, 0, 0], [1, -1 / 3, -1 / 3, -1 / 3]]
    adv, has = run([0.2, 0.5], [0, 0], finished=[False, True], reward=r)
    assert has[:, 0].tolist() == [True, True]
    assert abs(float(adv[1, 0]) - 0.5) < 1e-6  # 1 - 0.5
    assert abs(float(adv[0, 0]) - (0.3 + 0.5 * 0.5)) < 1e-6  # (0.5 - 0.2) + lambda * 0.5


def test_a_new_game_cuts_the_chain():
    # t=0 is seat 0's last move of a game that finishes there; t=1 is seat 0 in the next game.
    r = [[-1 / 3, 1, -1 / 3, -1 / 3], [0, 0, 0, 0]]
    adv, has = run([0.1, 0.9], [0, 0], finished=[True, False], reward=r)
    assert has[:, 0].tolist() == [True, False]
    assert abs(float(adv[0, 0]) - (-1 / 3 - 0.1)) < 1e-6  # no value or credit from the next game


def test_an_overflowed_game_gives_no_reward_or_credit():
    r = [[0, 0, 0, 0], [1, 0, 0, 0]]
    adv, has = run([0.2, 0.5], [0, 0], overflow=[False, True], reward=r)
    assert has[1, 0].item() is False  # its final decision has no target at all
    assert has[0, 0].item() is True  # the earlier one bootstraps from it, with no reward
    assert abs(float(adv[0, 0]) - 0.3) < 1e-6


def test_a_batch_gets_the_same_rewards_as_one_game():
    batch = jnp.array([(50, 10, 0, -5), (7, 7, 7, -5)])
    assert jnp.allclose(position_rewards(batch)[1], position_rewards(batch[1]), atol=1e-6)
    assert jnp.allclose(position_rewards(batch)[0], position_rewards(batch[0]), atol=1e-6)


def test_objectives_are_zero_sum_and_pay_what_they_say():
    from jaxenv.selfplay import objective_rewards

    scores = jnp.array([[300, 10, 10, -320], [50, 50, 0, -5]])
    for name in ("weighted", "king", "delta"):
        assert jnp.allclose(objective_rewards(scores, name).sum(1), 0, atol=1e-5)
    king = objective_rewards(scores, "king")
    assert jnp.allclose(king[0], jnp.array([0.75, -0.25, -0.25, -0.25]))
    assert jnp.allclose(king[1], jnp.array([0.25, 0.25, -0.25, -0.25]))
    delta = objective_rewards(scores, "delta")
    assert delta[0, 0] > delta[0, 1] > delta[0, 3]


@pytest.mark.parametrize("objective", ["king", "delta"])
def test_training_runs_with_each_objective(tmp_path, objective):
    from jaxenv.selfplay import main

    args = ["--objective", objective, "--out", str(tmp_path), "--iterations", "1", "--batch", "16", "--steps", "8",
            "--minibatch", "64", "--hidden", "16", "--depth", "1"]  # fmt: skip
    assert main(args) in (0, None)
    assert (tmp_path / "log.jsonl").read_text().strip()


def test_a_refit_can_only_change_bidding_outputs():
    policy = init_nets(jax.random.key(0), 16, 2)["policy"]
    grads = jax.tree.map(jnp.ones_like, policy)
    kept = bidding_only(grads)
    first_card = R.NUM_ACTIONS - 52
    assert all(float(jnp.abs(layer["w"]).sum()) == 0 for layer in kept["body"])
    assert float(jnp.abs(kept["policy"]["w"][:, first_card:]).sum()) == 0
    assert float(jnp.abs(kept["policy"]["b"][first_card:]).sum()) == 0
    assert float(jnp.abs(kept["policy"]["w"][:, :first_card]).sum()) > 0


def test_a_saved_net_loads_back(tmp_path):
    policy = init_nets(jax.random.key(0), 16, 2)["policy"]
    save_npz(policy, tmp_path / "net.npz", {})
    loaded = load_policy(tmp_path / "net.npz")
    assert jnp.allclose(loaded["body"][0]["w"], policy["body"][0]["w"])
    assert jnp.allclose(loaded["policy"]["w"], policy["policy"]["w"])


def test_a_refit_rollout_never_trains_card_play():
    from jaxenv import game as G

    nets = init_nets(jax.random.key(0), 16, 2)
    games = jax.vmap(G.new_game)(G.deal_pool(jax.random.key(1), 8))
    _, flat, _ = make_rollout(8, 64, 0.95, refit=True)(nets, nets["policy"], games, jax.random.key(2))
    legal = flat["x"][:, -R.NUM_ACTIONS :].astype(jnp.float32) > 0.5
    only_cards = ~legal[:, : R.NUM_ACTIONS - 52].any(1)
    assert not flat["valid"][only_cards].any()


def test_targets_favour_the_better_card_and_ignore_illegal_ones():
    values = np.zeros((1, 52), np.float32)
    values[0, [3, 7, 9]] = [20.0, 10.0, 99.0]
    legal = np.zeros((1, 52), bool)
    legal[0, [3, 7]] = True
    target = card_targets(values, legal)
    assert target[0, 3] > target[0, 7] > 0
    assert target[0, 9] == 0
    assert abs(target.sum() - 1) < 1e-5


def test_regret_counts_what_the_top_cards_lose():
    values = np.zeros((1, 52), np.float32)
    values[0, [3, 7, 9]] = [20.0, 10.0, 5.0]
    legal = np.zeros((1, 52), bool)
    legal[0, [3, 7, 9]] = True
    logits = np.zeros((1, 52), np.float32)
    logits[0, [7, 9, 3]] = [3.0, 2.0, 1.0]
    assert regret(logits, values, legal) == {"top1": 10.0, "top2": 10.0, "top3": 0.0}


def test_held_out_games_stay_whole():
    games = np.array([0, 0, 1, 1, 20, 21])
    assert split_games(games).tolist() == [True, True, False, False, True, False]
