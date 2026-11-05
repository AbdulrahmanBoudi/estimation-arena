"""Record teacher-labelled card decisions for imitation learning (rung 2).

    uv run python -m train.record --teacher pimc:samples=100 --games 800 --workers 4 \\
        --out runs/rung2/pimc100

At every card decision with more than one legal card, the teacher (a PIMC bot) searches and
scores every legal card: the mean, over its sampled deals, of the seat's own round score minus
the mean of the other three. One row per decision:

    features  (456,) float16  the acting seat's encoded Observation (estimation.rl.encoder)
    legal     (52,)  bool     legal cards
    values    (52,)  float32  teacher's value per legal card, 0 elsewhere
    choice    ()     int16    teacher's card
    hint      ()     int16    HeuristicBot's card, the teacher's rollout rule (a baseline)

Who plays the card is a separate choice (``--student``, ``--beta``): by default the teacher plays
(behaviour cloning). With a student, each decision the teacher's card is played with probability
beta and the student's otherwise, so the data covers the positions the student actually reaches
(DAgger). Bidding and estimates are HeuristicBot's, as in PIMC.

Game seeds start at 10,000,000 so training deals never overlap the arena's rating deals.
Data stays outside git: one .npz shard per chunk of games.
"""

from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import estimation.bots  # noqa: F401  (import order: the encoder imports the bots package)
from estimation.bots import HeuristicBot, make_bot
from estimation.bots.pimc import best_index
from estimation.engine import Bola, Phase, play_action
from estimation.engine.actions import PLAY_BASE
from estimation.engine.config import DEFAULT_CONFIG
from estimation.rl.encoder import encode

from arena.bots import maker

SEED_BASE = 10_000_000


def _one_thread() -> None:
    import torch

    torch.set_num_threads(1)


def record_chunk(
    games: range, teacher_spec: str, student_spec: str | None, beta: float, out: Path
) -> tuple[int, int]:
    """Play ``games`` (game indices), write one shard, return (games, rows)."""
    rows: dict[str, list] = {"features": [], "legal": [], "values": [], "choice": [], "hint": [], "game": []}
    for g in games:
        seed = SEED_BASE + g
        bola = Bola(seed)
        teacher = make_bot(teacher_spec, seed)
        if not hasattr(teacher, "search"):
            raise ValueError(f"teacher {teacher_spec!r} has no search(); use a pimc spec")
        samples = teacher.config.samples
        others = HeuristicBot()
        students = [maker(student_spec, seed + s, DEFAULT_CONFIG) for s in range(4)] if student_spec else None
        coin = random.Random(seed)
        while not bola.is_over:
            seat = bola.current_player
            obs = bola.observation(seat)
            legal = bola.legal_actions()
            if obs.phase is Phase.PLAY and len(legal) > 1:
                cards = [a - PLAY_BASE for a in legal]
                candidates, totals = teacher.search(obs, cards)
                choice = candidates[best_index(totals)]
                mask = np.zeros(52, np.bool_)
                mask[cards] = True
                values = np.zeros(52, np.float32)
                for card, total in zip(candidates, totals, strict=True):
                    values[card] = total / samples
                rows["features"].append(encode(obs).astype(np.float16))
                rows["legal"].append(mask)
                rows["values"].append(values)
                rows["choice"].append(choice)
                rows["hint"].append(candidates[0])  # search lists its card rule's own card first
                rows["game"].append(g)
                if students is None or coin.random() < beta:
                    action = play_action(choice)
                else:
                    action = students[seat].act(obs, legal)
            elif students is not None:
                action = students[seat].act(obs, legal)
            else:
                action = others.act(obs, legal)
            bola.step(action)
    path = out / f"games-{games.start:07d}-{games.stop:07d}.npz"
    np.savez_compressed(
        path,
        features=np.stack(rows["features"]),
        legal=np.stack(rows["legal"]),
        values=np.stack(rows["values"]),
        choice=np.array(rows["choice"], np.int16),
        hint=np.array(rows["hint"], np.int16),
        game=np.array(rows["game"], np.int32),
        teacher=teacher_spec,
        student=student_spec or "",
        beta=beta,
    )
    return len(games), len(rows["choice"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m train.record", description=__doc__.split("\n")[0])
    parser.add_argument("--teacher", default="pimc:samples=100")
    parser.add_argument("--student", help="lineup spec of the bot that plays (DAgger), e.g. sub:pimc-clone")
    parser.add_argument("--beta", type=float, default=1.0, help="chance the teacher's card is played (1)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--first", type=int, default=0, help="first game index (0)")
    parser.add_argument("--chunk", type=int, default=10, help="games per shard (10)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out = args.out.expanduser()
    args.out.mkdir(parents=True, exist_ok=True)
    chunks = [
        range(start, min(start + args.chunk, args.first + args.games))
        for start in range(args.first, args.first + args.games, args.chunk)
    ]
    task = partial(record_chunk, teacher_spec=args.teacher, student_spec=args.student, beta=args.beta, out=args.out)
    start = time.perf_counter()
    done_games = done_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_one_thread) as pool:
        for games, rows in pool.map(task, chunks):
            done_games += games
            done_rows += rows
            elapsed = time.perf_counter() - start
            print(f"{done_games}/{args.games} games, {done_rows} rows, {elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
