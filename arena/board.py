"""The live board's data: standings and matches, rebuilt only when a file changed."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .leaderboard import build
from .live import LIVE, public, read_matches
from .bots import ROOT

# The deploy worktree's results are the last merged ones; the newest are where matches run.
RESULTS_ROOT = Path(os.environ.get("ESTIMATION_RESULTS_ROOT", Path.home() / "workspace" / "estimation-arena"))
RECENT = 8
_cache: dict[str, Any] = {"signature": None, "standings": []}
ROW_FIELDS = ("name", "key", "kind", "anchor", "provisional", "elo", "king_elo", "score_elo", "positions", "games", "played_at")
MATCH_FIELDS = ("id", "kind", "key", "name", "opponent", "sets", "done", "state", "strip", "shares", "reward", "king", "delta", "started_at", "updated_at")


def _root() -> Path:
    return RESULTS_ROOT if (RESULTS_ROOT / "anchors.json").is_file() else ROOT


def _signature(root: Path) -> tuple:
    paths = [*root.glob("results/*.json"), root / "ladder" / "positions.json", *LIVE.glob("*.json")]
    stamps = []
    for path in paths:
        try:
            stamps.append((path.name, path.stat().st_mtime_ns))
        except OSError:
            pass
    return tuple(sorted(stamps))


def payload() -> dict[str, Any]:
    root = _root()
    matches = read_matches()
    signature = (_signature(root), tuple(m["state"] for m in matches))
    if signature != _cache["signature"]:
        board = build(root, matches)
        _cache["standings"] = [{k: row.get(k) for k in ROW_FIELDS} for row in board["rows"]]
        _cache["signature"] = signature
    names = {row["key"]: row["name"] for row in _cache["standings"]}
    view = [{**{k: m.get(k) for k in MATCH_FIELDS}, "opponent": names.get(m["opponent"], public(m["opponent"])[1])} for m in matches]
    return {
        "standings": _cache["standings"],
        "playing": [m for m in view if m["state"] == "playing"],
        "recent": [m for m in view if m["state"] != "playing"][:RECENT],
    }


def api(_: Request) -> Response:
    return JSONResponse(payload())
