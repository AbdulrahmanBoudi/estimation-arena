#!/usr/bin/env python3
"""Build the public eval set: fixed practice deals, bot-only games on them and reference scores.

    uv run python scripts/build-eval-set.py --out eval-set --sets 100 --workers 5
    HF_TOKEN=... uv run --with huggingface_hub python scripts/build-eval-set.py --out eval-set --publish NAME

Every pairing plays the same deals in duplicate sets, so scores compare directly. The practice
seed is public on purpose and must never be the ladder's. Every game is replayed before it is
written, and only bots play, so the set holds no human games.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from estimation.arena.duplicate import play_set  # noqa: E402
from estimation.engine import Bola, cards_of  # noqa: E402

from arena.bots import maker  # noqa: E402
from arena.models import prefer_ipv4  # noqa: E402
from arena.positions import summarize  # noqa: E402
from arena.run import _one_thread, engine_version  # noqa: E402

PRACTICE_SEED = 20261105
REPO = "estimation-eval"
RL_V2 = "rl:checkpoint=runs/m4-rules-v2/checkpoints/final.pt"
NAMES = {"sub:bitter-net": "Self-taught", "sub:net-pimc": "Copycat", RL_V2: "RL v2", "heuristic": "Heuristic"}
PAIRINGS = (("sub:bitter-net", RL_V2), ("sub:bitter-net", "heuristic"), ("sub:net-pimc", RL_V2))


def replay(game) -> list[dict]:
    """Step the engine through ``game``; return the hands of each deal and check the final scores."""
    bola = Bola(game.deal_seed)
    deals: list[dict] = []
    seen = None
    for action in game.actions:
        first = bola.observation(0)
        if (first.round_no, first.replays) != seen:
            seen = (first.round_no, first.replays)
            hands = [cards_of(bola.observation(seat).hand) for seat in range(4)]
            deals.append({"round": first.round_no, "replay": first.replays, "hands": hands})
        bola.step(action)
    if tuple(bola.scores) != tuple(game.scores):
        raise RuntimeError(f"set {game.set_index} rotation {game.rotation} did not replay")
    return deals


def row(pairing: int, game) -> dict:
    return {
        "id": f"{pairing}-{game.set_index}-{game.rotation}",
        "pairing": pairing,
        "set": game.set_index,
        "rotation": game.rotation,
        "deal_seed": game.deal_seed,
        "seats": [NAMES[name] for name in game.seats],
        "scores": list(game.scores),
        "rounds": game.rounds,
        "deals": replay(game),
        "actions": list(game.actions),
    }


def _task(lineup: tuple[str, ...], index: int):
    result, _ = play_set(lineup, PRACTICE_SEED, index, make=maker, keep_games=True)
    return result


def build(out: Path, sets: int, workers: int, pairings=PAIRINGS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    reference = []
    with (out / "games.jsonl").open("w") as games, ProcessPoolExecutor(workers, initializer=_one_thread) as pool:
        for number, (a, b) in enumerate(pairings):
            lineup = (a, b, a, b)
            results = list(pool.map(partial(_task, lineup), range(sets)))
            for result in results:
                for game in result.games:
                    games.write(json.dumps(row(number, game), separators=(",", ":")) + "\n")
            places, _ = summarize(results, lineup)
            sides = {
                NAMES[name]: {
                    "positions": places[name]["shares"],
                    "weighted_rank": places[name]["reward"],
                    "king": places[name]["king"],
                    "score_delta": places[name]["delta"],
                }
                for name in (a, b)
            }
            reference.append({"pairing": number, "sets": sets, "games": sets * 4, "sides": sides})
            print(f"pairing {number}: {' vs '.join(sides)} done", flush=True)
    data = {
        "practice_seed": PRACTICE_SEED,
        "engine": engine_version(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "reference": reference,
    }
    (out / "reference.json").write_text(json.dumps(data, indent=1) + "\n")
    return data


def card(data: dict) -> str:
    lines = ["| Pairing | Bot | King | Weighted rank | Score difference |", "|---|---|---:|---:|---:|"]
    for entry in data["reference"]:
        for name, side in entry["sides"].items():
            lines.append(
                f"| {entry['pairing']} | {name} | {side['king']['mean']:.1%} | {side['weighted_rank']['mean']:+.3f} | {side['score_delta']['mean']:+.1f} |"
            )
    template = (ROOT / "models" / REPO / "README.md").read_text()
    return template.replace("{{sets}}", str(data["reference"][0]["sets"])).replace("{{table}}", "\n".join(lines))


def publish(out: Path, owner: str) -> None:
    prefer_ipv4()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # its uploader picks its own addresses
    from huggingface_hub import HfApi

    (out / "README.md").write_text(card(json.loads((out / "reference.json").read_text())))
    api = HfApi(token=os.environ["HF_TOKEN"])
    name = f"{owner}/{REPO}"
    api.create_repo(name, repo_type="dataset", exist_ok=True)
    commit = api.upload_folder(repo_id=name, repo_type="dataset", folder_path=out, commit_message="Publish the eval set")
    print(commit.commit_url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sets", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--publish", metavar="OWNER", help="upload an already built set")
    args = parser.parse_args()
    if args.publish:
        publish(args.out, args.publish)
    else:
        build(args.out, args.sets, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
