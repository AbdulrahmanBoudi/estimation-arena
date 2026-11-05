"""RL v2 on the GPU: its 456 inputs and both of its nets as pure JAX functions.

Plays and scores as the ``rl`` bot does, so it can sit in a batch of self-play games as an
opponent. ``jaxenv.check_rlv2`` requires the inputs, the values and the chosen move to match at
every step. Default rules only, like ``jaxenv.rules``.
"""

import numpy as np

import jax
import jax.numpy as jnp

from .game import Game
from .rules import (
    AUCTION as IN_AUCTION,
    CALLS_TO_DOUBLE,
    CALLER_WITH_BONUS,
    DASH_OVER,
    DASH_UNDER,
    DASH_CALL,
    ESTIMATE_BASE,
    ESTIMATION,
    FIRST_BID,
    LAST_BID,
    NO_DASH_CALL,
    OPEN_ROUNDS,
    PASS,
    PLAY,
    PLAY_BASE,
    RISK_POINTS,
    SEATS,
    SUPER_CALL_MIN,
    WIN_BASE,
    WITH,
    legal_mask,
    rank_of,
    trump_of,
)

HAND, PLAYED, TABLE, UNSEEN, VOIDS, TRICK = 0, 52, 208, 260, 312, 324
CONTEXT, AUCTION, PLAYERS, SCORES, SHAPE, ANNOUNCED = 338, 361, 374, 430, 443, 452
NUM_FEATURES = 456
BID_ACTIONS, CARDS = PLAY_BASE, 52
FIXED_TRUMPS = 5
CLIP, STAKES_SCALE, SCORE_SCALE, ROUND_SD = 5.0, 100.0, 200.0, 20.0

_nodes, _weights = np.polynomial.hermite_e.hermegauss(24)
_NODES = jnp.asarray(_nodes, jnp.float32)
_NODE_WEIGHTS = jnp.asarray(_weights / _weights.sum(), jnp.float32)


def _clip(x):
    return jnp.clip(x, -CLIP, CLIP)


def _hand_values(cards) -> jax.Array:
    """Suit lengths / 13 and the expected tricks under each trump / 13, from a 52-card mask."""
    suits = cards.reshape(4, 13)
    length = suits.sum(1)
    ace, king, queen, jack = suits[:, 12], suits[:, 11], suits[:, 10], suits[:, 9]
    side = ace * 1.0 + king * jnp.where(length >= 2, 0.75, 0.25) + queen * jnp.where(length >= 3, 0.4, 0.1)
    sans = side + ((length > 4) & (ace | king)) * (length - 4) * 0.5
    trump = (ace * 1.0 + (king & (length >= 2)) * 0.9 + (queen & (length >= 3)) * 0.6
             + (jack & (length >= 4)) * 0.3 + jnp.maximum(length - 3, 0) * 1.0)  # fmt: skip
    ruff = jnp.where(length == 0, 0.8, jnp.where(length == 1, 0.4, 0.0))
    held = jnp.where(length > 0, jnp.minimum(length, trump + ruff.sum() - ruff), 0.0)
    tricks = side.sum() - side + held
    return jnp.concatenate([length / 13.0, tricks / 13.0, sans.sum()[None] / 13.0])


def _win_chance(scores, rounds_left, me):
    """The chance ``me`` finishes first if every score moves by an independent normal step."""
    spread = ROUND_SD * jnp.sqrt(rounds_left.astype(jnp.float32))
    gaps = (scores[:, None] - scores[None, :]) / spread  # [seat, other]
    phi = 0.5 * (1.0 + jax.scipy.special.erf((gaps[:, :, None] + _NODES) / jnp.sqrt(2.0)))
    phi = jnp.where(jnp.eye(4, dtype=bool)[:, :, None], 1.0, phi)
    chance = (phi.prod(1) * _NODE_WEIGHTS).sum(1)
    return chance[me] / chance.sum()


def encode(g: Game) -> jax.Array:
    """The ``NUM_FEATURES`` inputs of the seat to act in ``g``."""
    r = g.round
    me = r.cur
    rel = (me + SEATS) % 4
    in_auction = r.phase == IN_AUCTION
    in_play = r.phase == PLAY
    known = r.est >= 0
    total = (r.est * known).sum()

    # Who is With and who has passed, as in jaxenv.encode.
    top_trump = trump_of(r.top)
    calls_top = r.calls[:, jnp.clip(top_trump, 0, 4)]
    with_a = (r.top_seat >= 0) & (SEATS != r.top_seat) & r.out & ~r.dash & (calls_top >= 0)
    is_with = jnp.where(in_auction, with_a, r.with_w >= 0)
    level = jnp.where(in_auction, jnp.where(with_a, rank_of(calls_top), 0), jnp.maximum(r.with_w, 0))
    passed = r.out & ~r.dash & ~is_with

    # Cards.
    on_table = jnp.arange(4) < r.tlen
    table = jnp.zeros(52, bool).at[jnp.maximum(r.trick, 0)].max(on_table)
    unseen = ~r.played.any(0) & ~r.hands[me]

    # The trick in progress.
    has_trick = r.tlen > 0
    best, offset = r.trick[0], 0
    for i in range(1, 4):
        card = r.trick[i]
        suit = card // 13
        takes = (i < r.tlen) & (((suit == best // 13) & (card > best)) | ((suit != best // 13) & (suit == r.trump)))
        best, offset = jnp.where(takes, card, best), jnp.where(takes, i, offset)
    trick = jnp.concatenate([
        jax.nn.one_hot(r.trick[0] // 13, 4) * has_trick,
        jax.nn.one_hot(r.tlen, 4) * in_play,
        jax.nn.one_hot((r.leader + offset - me) % 4, 4) * has_trick,
        jnp.stack([has_trick & (best // 13 == r.trump), has_trick * (best % 13) / 12.0]),
    ])  # fmt: skip

    # Round context.
    has_top = r.top >= FIRST_BID
    caller_est = jnp.where(r.caller >= 0, r.est[jnp.maximum(r.caller, 0)], -1)
    contract = jnp.where(caller_est >= 0, caller_est, jnp.where(has_top, rank_of(r.top), r.est.max().clip(0)))
    settled = in_play  # the seats' estimates are complete only once play starts
    risks = jnp.where(settled & (r.last >= 0), jnp.abs(total - 13) // 2, 0)
    calls = (r.caller >= 0) + (r.with_w >= 0).sum()
    doubled = (r.phase == ESTIMATION) | in_play
    multiplier = r.carry * jnp.where(doubled & (calls >= CALLS_TO_DOUBLE), 2, 1)
    context = jnp.concatenate([
        jax.nn.one_hot(r.phase, 4),
        (r.fixed_trump >= 0)[None],
        jax.nn.one_hot(r.fixed_trump, 5),
        r.super_call[None],
        (contract >= SUPER_CALL_MIN)[None],
        jax.nn.one_hot(r.trump, 5),
        jnp.stack([
            (total - 13) / 13.0,
            settled,
            _clip(jnp.log2(r.carry.astype(jnp.float32)) / 3.0),
            _clip(jnp.log2(multiplier.astype(jnp.float32)) / 3.0),
            _clip(risks / 3.0),
            r.ntricks / 13.0,
        ]),
    ])  # fmt: skip

    # The auction.
    has_bidder = r.top_seat >= 0
    auction = jnp.concatenate([
        has_top[None],
        (has_top * (rank_of(r.top) - 4) / 9.0)[None],
        jax.nn.one_hot(top_trump, 5) * has_top,
        jax.nn.one_hot((r.top_seat - me) % 4, 4) * has_bidder,
        (in_auction & (r.top_seat == me))[None],
        (in_auction * (4 - r.dash.sum() - passed.sum()) / 4.0)[None],
    ])  # fmt: skip

    # Per player: what it scores if it makes its estimate, and if it misses by one.
    est, won = r.est, r.won
    left = 13 - r.ntricks
    alive = (won <= est) & (est <= won + left)
    is_caller = SEATS == r.caller
    called = is_caller | is_with
    is_super = called & (est >= SUPER_CALL_MIN)
    dash_points = jnp.where(total > 13, DASH_OVER, DASH_UNDER)
    bonus = jnp.where(called, CALLER_WITH_BONUS, 0)
    win = jnp.where(r.dash, dash_points, jnp.where(is_super, est * est, WIN_BASE + est + bonus))
    lose = jnp.where(r.dash, -dash_points, jnp.where(is_super, -((est * est) // 2), -1 - bonus))
    last_seat = jnp.where((r.phase == ESTIMATION) | ((r.caller >= 0) & in_play), r.last, -1)
    is_last = SEATS == last_seat
    win = (win + is_last * RISK_POINTS * risks) * multiplier
    lose = (lose - is_last * RISK_POINTS * risks) * multiplier
    players = jnp.stack([
        known * est / 13.0,
        known,
        won / 13.0,
        known * (est - won) / 13.0,
        known * _clip(win / STAKES_SCALE),
        known * _clip(lose / STAKES_SCALE),
        known & alive,
        known & ~alive,
        is_caller,
        is_with,
        r.dash,
        is_last,
        passed,
        is_with * level / 13.0,
    ], axis=1)[rel].ravel()  # fmt: skip

    # The scoreboard.
    scores = r.scores.astype(jnp.float32)
    own = scores[me]
    is_open = r.fixed_trump < 0
    fixed_left = FIXED_TRUMPS - g.fixed_index
    rounds_left = jnp.where(is_open, jnp.maximum(0, OPEN_ROUNDS - r.round_no + 1), 0) + fixed_left
    board = jnp.concatenate([
        _clip(scores[rel] / SCORE_SCALE),
        _clip((scores.max() - own) / STAKES_SCALE)[None],
        jnp.where(scores[rel[1:]] < own, 0.5, 1.0),
        jnp.stack([
            (scores > own).sum() / 3.0,
            is_open * jnp.maximum(0, 14 - r.round_no) / 13.0,
            fixed_left / 5.0,
            g.replays / 5.0,
            _win_chance(scores, rounds_left, me),
        ]),
    ])  # fmt: skip

    return jnp.concatenate([
        r.hands[me],
        r.played[rel[1:]].ravel(),
        table,
        unseen,
        r.shown_out[rel[1:]].ravel(),
        trick,
        context,
        auction,
        players,
        board,
        _hand_values(r.hands[me] | r.played[me]),
        r.announced[rel],
    ], dtype=jnp.float32)  # fmt: skip


# Facts about each action, as the nets derive them from the inputs.
_cards = jnp.arange(CARDS)
_rank = (_cards % 13).astype(jnp.float32)
_suit = jax.nn.one_hot(_cards // 13, 4)
_higher = (_suit @ _suit.T) * (_rank[:, None] > _rank[None, :])
_lower = _higher.T
_kind = np.zeros((BID_ACTIONS, 6), np.float32)
_tricks = np.zeros(BID_ACTIONS, np.float32)
_trump = np.zeros((BID_ACTIONS, 5), np.float32)
_estimate = np.zeros(BID_ACTIONS, np.float32)
_kind[NO_DASH_CALL, 0] = _kind[DASH_CALL, 1] = _kind[WITH, 3] = _kind[PASS, 4] = 1
for _a in range(FIRST_BID, LAST_BID + 1):
    _n, _t = divmod(_a - FIRST_BID, 5)
    _kind[_a, 2], _tricks[_a], _trump[_a, _t] = 1, (_n + 4) / 13, 1
for _k in range(14):
    _kind[ESTIMATE_BASE + _k, 5], _tricks[ESTIMATE_BASE + _k], _estimate[ESTIMATE_BASE + _k] = 1, _k / 13, 1
_with = np.zeros(BID_ACTIONS, np.float32)
_with[WITH] = 1
_kind, _tricks, _trump, _estimate, _with = map(jnp.asarray, (_kind, _tricks, _trump, _estimate, _with))


def card_features(x) -> jax.Array:
    hand, unseen = x[HAND : HAND + CARDS], x[UNSEEN : UNSEEN + CARDS]
    is_trump = x[CONTEXT + 12 : CONTEXT + 16] @ _suit.T
    is_led = x[TRICK : TRICK + 4] @ _suit.T
    winner_trump, winner_rank = x[TRICK + 12], x[TRICK + 13] * 12
    above = _rank > winner_rank + 0.5
    beats = is_trump * (1 - winner_trump + winner_trump * above) + is_led * (1 - winner_trump) * above
    higher_unseen = unseen @ _higher
    voids = x[VOIDS : VOIDS + 12].reshape(3, 4).sum(0) / 3
    return jnp.concatenate([
        jnp.stack([
            is_trump,
            is_led,
            _rank / 12,
            (1 - x[TRICK + 4]) * jnp.minimum(beats, 1.0),
            higher_unseen < 0.5,
            higher_unseen / 13,
            (unseen @ _suit) @ _suit.T / 13,
            (hand @ _suit) @ _suit.T / 13,
            hand @ _higher / 13,
            hand @ _lower / 13,
            voids @ _suit.T,
        ], axis=1),
        _suit,
    ], axis=1)  # fmt: skip


def bid_features(x) -> jax.Array:
    top_exists = x[AUCTION]
    top_tricks = (x[AUCTION + 1] * 9 + 4) / 13 * top_exists
    n = _tricks + _with * top_tricks
    action_trump = (_trump + _with[:, None] * x[AUCTION + 2 : AUCTION + 7]
                    + _estimate[:, None] * x[CONTEXT + 12 : CONTEXT + 17])  # fmt: skip
    expected = action_trump @ x[SHAPE + 4 : SHAPE + 9]
    return jnp.concatenate([
        _kind,
        action_trump,
        jnp.stack([
            n,
            expected,
            expected - n,
            n * 13 > 7.5,
            _estimate * (x[CONTEXT + 17] + _tricks),
            _estimate * top_exists * (jnp.abs(_tricks - top_tricks) < 0.01),
        ], axis=1),
    ], axis=1)  # fmt: skip


def _linear(net, name, x):
    return x @ net[f"{name}.weight"].T + net[f"{name}.bias"]


def _head(net, h, phi):
    joint = jax.nn.relu(_linear(net, "head.state", h) + phi @ net["head.action.weight"].T)
    return _linear(net, "head.per_action", h) + _linear(net, "head.out", joint)[:, 0]


def _trunk(net, x):
    for layer in (0, 2, 4):
        x = jax.nn.relu(_linear(net, f"trunk.{layer}", x))
    return x


def card_values(nets, x) -> jax.Array:
    return _head(nets["card"], _trunk(nets["card"], x), card_features(x))


def bid_values(nets, x) -> jax.Array:
    return _head(nets["bidding"], _trunk(nets["bidding"], x), bid_features(x))


def action_values(nets, g: Game, x=None) -> jax.Array:
    """A value per action id, the legal ones only meaningful: cards in play, else bids."""
    x = encode(g) if x is None else x
    return jnp.where(
        g.round.phase == PLAY,
        jnp.concatenate([jnp.zeros(BID_ACTIONS), card_values(nets, x)]),
        jnp.concatenate([bid_values(nets, x), jnp.zeros(CARDS)]),
    )


def act(nets, g: Game) -> jax.Array:
    """The move the bot plays: the legal action of highest value, the lowest id on a tie."""
    return jnp.argmax(jnp.where(legal_mask(g.round), action_values(nets, g), -jnp.inf))


def load(path) -> dict:
    """The nets from a weights file (``train/export_rlv2.py`` writes it from a checkpoint)."""
    nets: dict = {"card": {}, "bidding": {}}
    with np.load(path) as data:
        for key in data.files:
            net, name = key.split("/", 1)
            nets[net][name] = jnp.asarray(data[key])
    return nets
