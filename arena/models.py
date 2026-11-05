"""Model files live on Hugging Face, pinned in ``models.lock.json``.

Each entry maps a path in this repo to a Hugging Face repo, an exact commit and a checksum:

    {"bots/bitter-net/net.npz": {"repo": "owner/name", "revision": "<commit>", "file": "net.npz",
                                 "sha256": "<hex>"}}

``python -m arena.models fetch`` downloads what is missing and refuses a file whose checksum
differs. The lock sits outside the bot folders, so it never changes a bot's hash: the hash
covers the model's bytes, wherever they came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / "models.lock.json"
HUB = "https://huggingface.co"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(lock_file: Path = LOCK_FILE) -> dict[str, dict[str, str]]:
    return json.loads(lock_file.read_text()) if lock_file.is_file() else {}


def url_for(entry: dict[str, str]) -> str:
    return f"{HUB}/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}"


def _fetch_one(path: Path, entry: dict[str, str]) -> None:
    """Download to a temporary file and move it into place only once the checksum matches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url_for(entry), timeout=60) as reply, part.open("wb") as f:
            while chunk := reply.read(1 << 20):
                f.write(chunk)
        got = sha256(part)
        if got != entry["sha256"]:
            raise ValueError(f"{path}: checksum {got} does not match the lock {entry['sha256']}")
        os.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


def fetch(root: Path = ROOT, lock_file: Path = LOCK_FILE) -> list[str]:
    """Make every locked file present and correct. Returns the paths it downloaded."""
    fetched = []
    for rel, entry in load_lock(lock_file).items():
        path = root / rel
        if path.exists():
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"{rel}: the file here differs from the lock; move it aside to re-fetch")
            continue
        _fetch_one(path, entry)
        fetched.append(rel)
    return fetched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.models", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["fetch"])
    parser.parse_args(argv)
    fetched = fetch()
    print(f"{len(load_lock())} models ready, {len(fetched)} downloaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
