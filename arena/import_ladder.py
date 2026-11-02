"""Import the pairwise results of an engine ladder run, so the anchors are rated on the
leaderboard exactly as on Yosif's ladder.

    uv run python -m arena.import_ladder runs/ladder-v2

Reads every ``pairs/*.json`` report under the ladder folder and writes ``ladder/pairs.json``:
only the win/draw/loss counts per pair, a few KB, which goes in git. The ladder's own games
and reports stay outside git.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

from estimation.arena.report import load_pairs

from .bots import ROOT

LADDER_FILE = ROOT / "ladder" / "pairs.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.import_ladder")
    parser.add_argument("ladder", type=Path, help="a ladder run folder with pairs/*.json")
    args = parser.parse_args(argv)
    files = sorted((args.ladder / "pairs").glob("*.json"))
    if not files:
        parser.error(f"no pairs/*.json under {args.ladder}")
    pairs = [dataclasses.asdict(p) for f in files for p in load_pairs(f)]
    meta = {}
    ladder_json = args.ladder / "ladder.json"
    if ladder_json.is_file():
        data = json.loads(ladder_json.read_text())
        meta = {"sets": data.get("sets"), "seed": data.get("seed"), "ratings": data.get("ratings")}
    LADDER_FILE.parent.mkdir(parents=True, exist_ok=True)
    LADDER_FILE.write_text(
        json.dumps(
            {
                "source": args.ladder.name,
                "imported_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "pair_files": len(files),
                **meta,
                "pairs": pairs,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(pairs)} pair records from {len(files)} files -> {LADDER_FILE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
