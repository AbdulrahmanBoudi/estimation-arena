"""Bitter-lesson self-play: one net for every decision, from raw state, PPO on the GPU.

    .venv-jax/bin/python -m jaxenv.selfplay --out ~/workspace/estimation-data/bitter/run1 \\
        --logdir ~/workspace/estimation-data/tb

- One MLP over jaxenv.encode: Dash Calls, bids, With, estimates and cards. No hand evaluation,
  no imitation, no Heuristic; it starts from nothing.
- Rollouts run inside ``lax.scan``: rules, encoder and net all on the GPU, nothing on the host.
- Half the games are pure self-play. In the other half two seats play a past version from a
  pool, so the policy cannot just cycle.
- Credit: every decision of a seat gets that seat's round result relative to the other three,
  over 50, clipped to +-5, against a learned value baseline. A round cut off by the end of a
  rollout is dropped. Decisions with one legal move are not trained on.
- Evaluation: greedy against a uniform random player and against the oldest pool version, in
  points per round. Real strength is measured by the arena (bots/bitter-net).
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")  # before JAX touches the GPU

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from . import encode as E  # noqa: E402
from . import game as G  # noqa: E402
from . import rules as R  # noqa: E402

POOL_DEALS = 8  # deals per game per rollout


def init_net(key, hidden: int, depth: int):
    sizes = [E.SIZE] + [hidden] * depth
    keys = jax.random.split(key, depth + 2)
    glorot = jax.nn.initializers.orthogonal(jnp.sqrt(2.0))
    body = [{"w": glorot(k, (a, b)), "b": jnp.zeros(b)} for k, a, b in zip(keys, sizes[:-1], sizes[1:])]
    small = jax.nn.initializers.orthogonal(0.01)
    return {
        "body": body,
        "policy": {"w": small(keys[-2], (hidden, R.NUM_ACTIONS)), "b": jnp.zeros(R.NUM_ACTIONS)},
        "value": {"w": jax.nn.initializers.orthogonal(1.0)(keys[-1], (hidden, 1)), "b": jnp.zeros(1)},
    }


def apply_net(net, x):
    for layer in net["body"]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    return x @ net["policy"]["w"] + net["policy"]["b"], (x @ net["value"]["w"] + net["value"]["b"])[..., 0]


def _where_games(cond, a, b):
    return jax.tree.map(lambda x, y: jnp.where(cond.reshape(-1, *[1] * (x.ndim - 1)), x, y), a, b)


def round_rewards(r):
    """[B, 4]: each seat's round score minus the mean of the other three, over 50, clipped."""
    res = r.result.astype(jnp.float32)
    rel = res - (res.sum(1, keepdims=True) - res) / 3.0
    return jnp.clip(rel / 50.0, -5.0, 5.0)


def play_turn(net, opp, g, learner, key, greedy=False):
    """One decision in every game: (action, logp, value, x, legal count)."""
    seat = g.round.cur
    x = jax.vmap(E.encode)(g.round, seat)
    mask = jax.vmap(G.game_legal)(g)
    logits, value = apply_net(net, x)
    opp_logits, _ = apply_net(opp, x)
    mine = jnp.take_along_axis(learner, seat[:, None], 1)[:, 0]
    logits = jnp.where(mine[:, None], logits, opp_logits)
    logits = jnp.where(mask, logits, -1e9)
    action = jnp.argmax(logits, 1) if greedy else jax.random.categorical(key, logits)
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], 1)[:, 0]
    return action, logp, value, x, mine, mask.sum(1)


def _advance_games(g, used, base, pool, action):
    r = jax.vmap(R.step)(g.round, action)
    ended = (r.phase == R.DONE) & ~g.over
    before = g.round_no
    g = jax.vmap(G.game_step)(g, action, pool[base + used % POOL_DEALS])
    used = used + (g.round_no != before)
    finished = g.over | g.overflow
    fresh = jax.vmap(G.new_game)(pool[base + (used + 1) % POOL_DEALS])
    return _where_games(finished, fresh, g), used + finished, r, ended


def make_rollout(batch: int, steps: int):
    base = jnp.arange(batch) * POOL_DEALS

    def rollout(net, opp, g, key):
        k_pool, k_seats, k_mix, k_steps = jax.random.split(key, 4)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        # half pure self-play; the rest: two opposite seats against the past version
        pairs = jax.random.bernoulli(k_seats, 0.5, (batch,))
        two = jnp.stack([pairs, ~pairs, pairs, ~pairs], 1)
        selfplay = jax.random.bernoulli(k_mix, 0.5, (batch,))[:, None]
        learner = selfplay | two

        def one(carry, k):
            g, used = carry
            action, logp, value, x, mine, n_legal = play_turn(net, opp, g, learner, k)
            seat = g.round.cur
            g, used, r, ended = _advance_games(g, used, base, pool, action)
            rew = round_rewards(r)
            return (g, used), dict(x=x.astype(jnp.float16), action=action, logp=logp, value=value, seat=seat,
                                   train=mine & (n_legal > 1), ended=ended, rew=rew)  # fmt: skip

        (g, _), tr = jax.lax.scan(one, (g, jnp.zeros(batch, jnp.int32)), jax.random.split(k_steps, steps))

        def back(carry, t):
            ret, has = carry
            ret = jnp.where(t["ended"][:, None], t["rew"], ret)
            has = has | t["ended"]
            target = jnp.take_along_axis(ret, t["seat"][:, None], 1)[:, 0]
            return (ret, has), (target, has & t["train"])

        zeros = (jnp.zeros((batch, 4)), jnp.zeros(batch, bool))
        _, (target, valid) = jax.lax.scan(back, zeros, {k: tr[k] for k in ("ended", "rew", "seat", "train")}, reverse=True)
        data = dict(x=tr["x"], action=tr["action"], logp=tr["logp"], value=tr["value"], target=target, valid=valid)
        flat = jax.tree.map(lambda x: x.reshape(batch * steps, *x.shape[2:]), data)  # one copy only
        adv = flat["target"] - flat["value"]
        w = flat["valid"]
        n = jnp.maximum(w.sum(), 1)
        mean = (adv * w).sum() / n
        std = jnp.sqrt((((adv - mean) ** 2) * w).sum() / n)
        flat["adv"] = (adv - mean) / (std + 1e-6)
        return g, flat

    return jax.jit(rollout)


def make_update(tx, clip: float, vf: float):
    def loss_fn(net, mb, ent):
        x = mb["x"].astype(jnp.float32)
        logits, value = apply_net(net, x)
        legal = x[:, -R.NUM_ACTIONS:] > 0.5
        logits = jnp.where(legal, logits, -1e9)
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, mb["action"][:, None], 1)[:, 0]
        ratio = jnp.exp(logp - mb["logp"])
        w = mb["valid"].astype(jnp.float32)
        n = jnp.maximum(w.sum(), 1.0)
        pg = (w * -jnp.minimum(ratio * mb["adv"], jnp.clip(ratio, 1 - clip, 1 + clip) * mb["adv"])).sum() / n
        v = (w * (value - mb["target"]) ** 2).sum() / n
        entropy = (w * -(jnp.exp(logp_all) * jnp.where(legal, logp_all, 0.0)).sum(1)).sum() / n
        kl = (w * (mb["logp"] - logp)).sum() / n
        return pg + vf * v - ent * entropy, dict(pg=pg, value_loss=v, entropy=entropy, kl=kl)

    @jax.jit
    def step(net, opt, mb, ent):
        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(net, mb, ent)
        updates, opt = tx.update(grads, opt, net)
        return optax.apply_updates(net, updates), opt, info

    return step


def make_eval(batch: int, steps: int):
    """Greedy ``net`` in two opposite seats against ``opp``: its edge in points per round."""
    base = jnp.arange(batch) * POOL_DEALS

    @jax.jit
    def run(net, opp, key):
        k_pool, k_steps = jax.random.split(key)
        pool = G.deal_pool(k_pool, batch * POOL_DEALS)
        g = jax.vmap(G.new_game)(pool[base])
        learner = jnp.broadcast_to(jnp.array([True, False, True, False]), (batch, 4))
        learner = learner ^ (jnp.arange(batch) % 2 == 0)[:, None]

        def one(carry, k):
            g, used, total, rounds = carry
            action, *_ = play_turn(net, opp, g, learner, k, greedy=True)
            g, used, r, ended = _advance_games(g, used, base, pool, action)
            res = r.result.astype(jnp.float32) / r.carry[:, None]  # per-round points, carry removed
            edge = (res * learner).sum(1) / 2 - (res * ~learner).sum(1) / 2
            return (g, used, total + jnp.where(ended, edge, 0.0).sum(), rounds + ended.sum()), None

        (_, _, total, rounds), _ = jax.lax.scan(one, (g, jnp.zeros(batch, jnp.int32), 0.0, 0), jax.random.split(k_steps, steps))
        return total / jnp.maximum(rounds, 1), rounds

    return run


def save_npz(net, path: Path, meta: dict) -> None:
    arrays = {}
    for k, layer in enumerate(net["body"]):
        arrays[f"w{k}"], arrays[f"b{k}"] = np.asarray(layer["w"]), np.asarray(layer["b"])
    arrays["pw"], arrays["pb"] = np.asarray(net["policy"]["w"]), np.asarray(net["policy"]["b"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v.astype(np.float32) for k, v in arrays.items()}, meta=json.dumps(meta))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m jaxenv.selfplay", description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--batch", type=int, default=4096, help="games at once")
    p.add_argument("--steps", type=int, default=128, help="decisions per game per iteration")
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--entropy", type=float, default=0.01)
    p.add_argument("--entropy-end", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=32768)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--pool", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", type=Path)
    p.add_argument("--resume", type=Path, help="state.pkl to continue from")
    a = p.parse_args(argv)
    a.out = a.out.expanduser()
    a.out.mkdir(parents=True, exist_ok=True)
    print("device:", jax.devices()[0], "| input size:", E.SIZE, flush=True)
    key = jax.random.key(a.seed)
    key, k_net, k_random = jax.random.split(key, 3)
    net = init_net(k_net, a.hidden, a.depth)
    random_net = jax.tree.map(jnp.zeros_like, net)  # all-zero logits: uniform over legal moves
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    opt = tx.init(net)
    pool = [jax.device_get(net)]
    start = 1
    if a.resume:
        state = pickle.loads(a.resume.expanduser().read_bytes())
        net, opt, pool, start = state["net"], state["opt"], state["pool"], state["iteration"] + 1
    rollout = make_rollout(a.batch, a.steps)
    update = make_update(tx, a.clip, a.vf)
    evaluate = make_eval(2048, 2500)
    key, k = jax.random.split(key)
    g = jax.vmap(G.new_game)(G.deal_pool(k, a.batch))
    writer = None
    if a.logdir:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(str(a.logdir.expanduser() / f"bitter-{a.out.name}"))
    log = (a.out / "log.jsonl").open("a")
    params = sum(x.size for x in jax.tree.leaves(net))
    print(f"net: {params:,} parameters; {a.batch * a.steps:,} decisions per iteration", flush=True)
    t0, decisions = time.perf_counter(), 0
    for it in range(start, a.iterations + 1):
        key, k_opp, k_roll, k_perm, k_eval = jax.random.split(key, 5)
        opp = pool[int(jax.random.randint(k_opp, (), 0, len(pool)))]
        g, flat = rollout(net, opp, g, k_roll)
        n = a.batch * a.steps
        w = flat["valid"]
        frac = (it - 1) / max(a.iterations - 1, 1)
        ent = a.entropy + (a.entropy_end - a.entropy) * frac
        for epoch in range(a.epochs):
            order = jnp.argsort(jax.random.uniform(jax.random.fold_in(k_perm, epoch), (n,)))
            for s in range(0, n - a.minibatch + 1, a.minibatch):
                idx = order[s : s + a.minibatch]
                mb = jax.tree.map(lambda x: x[idx], flat)
                net, opt, info = update(net, opt, mb, ent)
        decisions += n
        if it % a.snapshot_every == 0:
            pool = (pool + [jax.device_get(net)])[-a.pool :]
        row = {"iteration": it, "valid": float(w.mean()), **{k: float(v) for k, v in info.items()},
               "decisions": decisions, "per_s": decisions / (time.perf_counter() - t0)}  # fmt: skip
        if it % a.eval_every == 0 or it == 1:
            vs_random, rounds = evaluate(net, random_net, k_eval)
            vs_old, _ = evaluate(net, pool[0], jax.random.fold_in(k_eval, 1))
            row |= {"vs_random": float(vs_random), "vs_oldest": float(vs_old), "eval_rounds": int(rounds)}
            print(f"iter {it:>5}: vs random {row['vs_random']:+8.1f}  vs oldest {row['vs_oldest']:+7.1f} points/round | "
                  f"entropy {row['entropy']:.2f} value loss {row['value_loss']:.3f} kl {row['kl']:.4f} | {row['per_s']:,.0f} decisions/s", flush=True)
            save_npz(net, a.out / "net.npz", {"iteration": it, "args": {k: str(v) for k, v in vars(a).items()}})
            (a.out / "state.pkl").write_bytes(pickle.dumps(
                {"net": jax.device_get(net), "opt": jax.device_get(opt), "pool": pool, "iteration": it}))
        if it % 200 == 0:
            save_npz(net, a.out / f"net-{it:05d}.npz", {"iteration": it})
        log.write(json.dumps(row) + "\n")
        log.flush()
        if writer:
            for k, v in row.items():
                writer.add_scalar(k, v, it)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
