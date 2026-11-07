"""Teach Self-taught's net from a deeper search: Yosif's expert iteration, on our one-net bot.

    python -m train.copy_search record --out DIR --games 500 --workers 5   # engine environment
    python -m train.copy_search copy --out DIR --net bots/bitter-search/net.npz  # JAX environment

record plays Self-taught + look-ahead in all four seats. At a share of its card decisions, a deeper
search values every legal card and plays its best. copy trains the net towards those values, and
measures on held-out games how much its top cards lose by the teacher's values. jaxenv.selfplay
--refit then fits the bidding to the new card play, leaving the card outputs exactly as copied.
Games use their own practice seeds, never the ladder's.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE_SEED = 30_000_000
PART_GAMES = 25
TEMPERATURE = 5.0  # points of search value that make one card e times likelier in the target
TOPS = (1, 2, 3)


def record_game(game: int, play_spec: str, teacher_spec: str, label_share: float) -> dict[str, np.ndarray]:
    from estimation.bots.hybrid.bot import equivalent_runs
    from estimation.bots.knowledge import read_knowledge
    from estimation.bots.pimc import best_index
    from estimation.engine import Bola, Phase, play_action
    from estimation.engine.actions import PLAY_BASE
    from estimation.engine.config import DEFAULT_CONFIG

    from arena.bots import maker
    from jaxenv.encode_np import encode

    seed = BASE_SEED + game
    players = [maker(play_spec, seed + 1 + s, DEFAULT_CONFIG) for s in range(4)]
    teachers = [maker(teacher_spec, seed + 5 + s, DEFAULT_CONFIG) for s in range(4)]
    labels = random.Random(seed)
    bola = Bola(seed, DEFAULT_CONFIG)
    xs, masks, values = [], [], []
    while not bola.is_over:
        seat = bola.current_player
        obs, legal = bola.observation(seat), bola.legal_actions()
        action = -1
        if obs.phase is Phase.PLAY and len(legal) > 1 and labels.random() < label_share:
            cards = [a - PLAY_BASE for a in legal]
            x = encode(obs, legal)
            logits = teachers[seat]._net(x)
            search = teachers[seat].search(obs, cards, [float(logits[PLAY_BASE + c]) for c in cards])
            if len(search.cards) > 1:
                value_of = dict(zip(search.cards, search.means, strict=True))
                row = np.zeros(52, np.float32)
                for run in equivalent_runs(cards, read_knowledge(obs).unseen):  # touching cards share a value
                    row[run] = next(value_of[c] for c in run if c in value_of)
                mask = np.zeros(52, bool)
                mask[cards] = True
                xs.append(x.astype(np.float16))
                masks.append(mask)
                values.append(row)
                action = play_action(search.cards[best_index(search.means)])
        if action < 0:
            action = players[seat].act(obs, legal)
        bola.step(action)
    if not xs:
        return {"x": np.zeros((0, 0), np.float16), "legal": np.zeros((0, 52), bool), "values": np.zeros((0, 52), np.float32)}
    return {"x": np.stack(xs), "legal": np.stack(masks), "values": np.stack(values)}


def record_part(path: str, first: int, count: int, play_spec: str, teacher_spec: str, label_share: float) -> str:
    rows = [(first + k, record_game(first + k, play_spec, teacher_spec, label_share)) for k in range(count)]
    rows = [(g, r) for g, r in rows if len(r["legal"])]
    part = Path(path)
    temporary = part.with_name(part.name + ".part.npz")
    np.savez_compressed(
        temporary,
        x=np.concatenate([r["x"] for _, r in rows]),
        legal=np.concatenate([r["legal"] for _, r in rows]),
        values=np.concatenate([r["values"] for _, r in rows]),
        game=np.concatenate([np.full(len(r["legal"]), g, np.int32) for g, r in rows]),
    )
    temporary.replace(part)
    return part.name


def _one_thread() -> None:
    import torch

    torch.set_num_threads(1)


def record(args: argparse.Namespace) -> None:
    """Saved part by part, so an interrupted recording resumes where it stopped."""
    parts = args.out / "records"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(parts / f"part-{first:06d}.npz", first, min(args.part_games, args.games - first))
            for first in range(0, args.games, args.part_games)]  # fmt: skip
    missing = [job for job in jobs if not job[0].exists()]
    print(f"{len(missing)} of {len(jobs)} parts to record", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_one_thread) as pool:
        futures = [pool.submit(record_part, str(path), first, count, args.play, args.teacher, args.label_share)
                   for path, first, count in missing]  # fmt: skip
        for done, future in enumerate(as_completed(futures), 1):
            print(f"{done}/{len(missing)} {future.result()}", flush=True)


def load_records(folder: Path) -> dict[str, np.ndarray]:
    parts = [np.load(path) for path in sorted((folder / "records").glob("part-*.npz")) if not path.name.endswith(".part.npz")]
    return {key: np.concatenate([part[key] for part in parts]) for key in ("x", "legal", "values", "game")}


def split_games(game: np.ndarray, share: float = 0.05) -> np.ndarray:
    """True for the rows of held-out games: whole games, so no position leaks across."""
    return game % round(1 / share) == 0


def card_targets(values: np.ndarray, legal: np.ndarray, temperature: float = TEMPERATURE) -> np.ndarray:
    """A distribution over the legal cards from the search's value of each."""
    best = np.where(legal, values, -np.inf).max(1, keepdims=True)
    raw = np.where(legal, np.exp(np.clip((values - best) / temperature, -30.0, 0.0)), 0.0)
    return (raw / raw.sum(1, keepdims=True)).astype(np.float32)


def regret(card_logits: np.ndarray, values: np.ndarray, legal: np.ndarray) -> dict[str, float]:
    """Mean points lost, by the teacher's values, if the net's top k cards were searched perfectly."""
    best = np.where(legal, values, -np.inf).max(1)
    order = np.argsort(-np.where(legal, card_logits, -np.inf), axis=1)
    out = {}
    for k in TOPS:
        kept = np.zeros_like(legal)
        np.put_along_axis(kept, order[:, :k], True, 1)
        kept &= legal
        out[f"top{k}"] = float((best - np.where(kept, values, -np.inf).max(1)).mean())
    return out


def copy(args: argparse.Namespace) -> None:
    import jax
    import jax.numpy as jnp
    import optax

    from jaxenv import rules as R
    from jaxenv.selfplay import apply_policy, load_policy, save_npz

    data = load_records(args.out)
    hold = split_games(data["game"])
    print(f"{len(hold):,} decisions, {int(hold.sum()):,} held out", flush=True)
    policy = load_policy(args.net)
    targets = card_targets(data["values"], data["legal"])
    train_rows = np.flatnonzero(~hold)
    hold_rows = np.flatnonzero(hold)
    first_card = R.NUM_ACTIONS - 52

    @jax.jit
    def cards_of(net, x):
        return apply_policy(net, x)[0][:, first_card:]

    def measure(net) -> dict[str, float]:
        logits = np.concatenate([
            np.asarray(cards_of(net, jnp.asarray(data["x"][hold_rows[s : s + 4096]], jnp.float32)))
            for s in range(0, len(hold_rows), 4096)
        ])  # fmt: skip
        return regret(logits, data["values"][hold_rows], data["legal"][hold_rows])

    before = measure(policy)
    print("before", before, flush=True)
    steps = args.epochs * (len(train_rows) // args.batch)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(args.lr, steps)))
    state = tx.init(policy)

    @jax.jit
    def step(net, state, x, legal, target):
        def loss_fn(net):
            logits = jnp.where(legal, apply_policy(net, x)[0][:, first_card:], -1e9)
            return -(target * jax.nn.log_softmax(logits)).sum(1).mean()

        loss, grads = jax.value_and_grad(loss_fn)(net)
        updates, state = tx.update(grads, state, net)
        return optax.apply_updates(net, updates), state, loss

    rng = np.random.default_rng(0)
    for epoch in range(args.epochs):
        order = rng.permutation(train_rows)
        total = 0.0
        batches = len(order) // args.batch
        for b in range(batches):
            rows = np.sort(order[b * args.batch : (b + 1) * args.batch])
            policy, state, loss = step(policy, state, jnp.asarray(data["x"][rows], jnp.float32),
                                       jnp.asarray(data["legal"][rows]), jnp.asarray(targets[rows]))  # fmt: skip
            total += float(loss)
        print(f"epoch {epoch + 1}: loss {total / max(batches, 1):.4f} held out {measure(policy)}", flush=True)
    after = measure(policy)
    save_npz(policy, args.out / "copied.npz", {"before": before, "after": after, "decisions": len(hold)})
    (args.out / "copy.json").write_text(json.dumps({"before": before, "after": after, "decisions": len(hold)}, indent=1))
    print("after", after, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    rec = commands.add_parser("record")
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--games", type=int, default=500)
    rec.add_argument("--part-games", type=int, default=PART_GAMES)
    rec.add_argument("--workers", type=int, default=5)
    rec.add_argument("--label-share", type=float, default=0.5)
    rec.add_argument("--play", default="sub:bitter-search")
    rec.add_argument("--teacher", default="sub:bitter-search:samples=64:top_k=0")
    cop = commands.add_parser("copy")
    cop.add_argument("--out", type=Path, required=True)
    cop.add_argument("--net", type=Path, required=True)
    cop.add_argument("--epochs", type=int, default=4)
    cop.add_argument("--batch", type=int, default=1024)
    cop.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args(argv)
    {"record": record, "copy": copy}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
