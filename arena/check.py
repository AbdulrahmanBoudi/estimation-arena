"""Check a bot before opening a pull request. It runs what smoke runs on that bot, plus the
things smoke does not look at (see docs/submitting.md).

    uv run python -m arena.check <bot>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from .bots import ROOT, Manifest, load_manifest
from .models import load_lock, sha256
from .run import main as run_main

MAX_FILE_BYTES = 5_000_000  # anything bigger is a model: Hugging Face, not Git
TARGET_S = 0.25  # AGENTS.md: card decisions should stay under this; the hard limit is 5 s


def problems(manifest: Manifest, lock: dict[str, dict[str, str]], root: Path = ROOT) -> list[str]:
    """What would stop this bot from being rated, in words that say how to fix it."""
    found = []
    if manifest.folder.parent != root / "bots":
        found.append(f"{manifest.folder} is not under bots/, so it would never be rated")
    for field in ("author", "description"):
        if not getattr(manifest, field):
            found.append(f"manifest.json has no {field}")
    if not manifest.method:
        found.append("manifest.json has no method: the leaderboard draws its steps from it")

    in_lock = {
        path.relative_to(root).as_posix(): path
        for path in manifest.folder.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() in lock
    }
    for path in manifest.folder.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        if name not in in_lock and path.stat().st_size > MAX_FILE_BYTES:
            found.append(f"{name} is too big for Git: upload it to Hugging Face and add it to models.lock.json")

    needed = {manifest.needs: root / manifest.needs} if manifest.needs else {}
    for name, path in {**in_lock, **needed}.items():
        entry = lock.get(name)
        if entry is None:
            found.append(f"{name} is not in models.lock.json, so CI cannot download it")
        elif not re.fullmatch(r"[0-9a-f]{40}", entry.get("revision", "")):
            found.append(f"{name}: models.lock.json must pin a full Hugging Face commit")
        elif not path.is_file():
            found.append(f"{name} is missing: run python -m arena.models fetch")
        elif sha256(path) != entry.get("sha256"):
            found.append(f"{name} does not match the sha256 in models.lock.json")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.check", description=__doc__.split("\n")[0])
    parser.add_argument("bot", help="folder name under bots/")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.bot)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"problem: {error}")
        return 1
    found = problems(manifest, load_lock())
    for problem in found:
        print(f"problem: {problem}")
    if found:
        return 1

    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "result.json"
        code = run_main(
            ["--bot", args.bot, "--sets", "1", "--seed", "1000", "--workers", "2",
             "--anchors", "Random,Heuristic", "--out", str(out)]
        )  # fmt: skip
        if code:
            print(f"problem: the games ended with exit code {code}")
            return 1
        result = json.loads(out.read_text())

    slow = [v["anchor"] for v in result["versus"] if v["timing"]["p95_s"] > TARGET_S]
    if slow:
        print(f"warning: over {TARGET_S} s per decision against {', '.join(slow)}; the hard limit is 5 s")
    print(f"{args.bot} is ready for a pull request. Scores from one set mean nothing; the ladder rates it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
