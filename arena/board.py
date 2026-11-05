"""The live board's data: standings and matches, rebuilt only when a file changed."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from .guide import GUIDE, STYLE as GUIDE_STYLE
from .leaderboard import build, method_svg
from .live import LIVE, public, read_matches
from .plain import about_bot, plain_name
from .training import training_of
from .bots import ROOT, SUB_PREFIX

# The deploy worktree's results are the last merged ones; the newest are where matches run.
RESULTS_ROOT = Path(os.environ.get("ESTIMATION_RESULTS_ROOT", Path.home() / "workspace" / "estimation-arena"))
RECENT = 8
_cache: dict[str, Any] = {"signature": None, "standings": []}
ROW_FIELDS = ("name", "code_name", "key", "kind", "anchor", "provisional", "elo", "king_elo", "score_elo", "positions", "games", "played_at")
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
    view = [{**{k: m.get(k) for k in MATCH_FIELDS}, "name": plain_name(m["key"], m["name"]), "opponent": names.get(m["opponent"], plain_name(m["opponent"], public(m["opponent"])[1]))} for m in matches]
    return {
        "standings": _cache["standings"],
        "playing": [m for m in view if m["state"] == "playing"],
        "recent": [m for m in view if m["state"] != "playing"][:RECENT],
    }


def _manifest(key: str) -> dict[str, Any]:
    """A submission's manifest, from the merged bots or the ones still being tested."""
    name = key[len(SUB_PREFIX) :].split(":")[0]
    for root in (ROOT, _root()):
        path = root / "bots" / name / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text())
    return {}


def explain(key: str) -> dict[str, Any] | None:
    """How one bot on the board plays: its steps and training, as HTML we generate."""
    if not any(row["key"] == key for row in _cache["standings"]):
        return None
    if key.startswith(SUB_PREFIX):
        about = _manifest(key)
    else:
        anchors = json.loads((_root() / "anchors.json").read_text())
        about = next((a for a in anchors if a["spec"] == key), {})
    plain = about_bot(key)
    trained = training_of(key) or {}
    return {
        "description": plain.get("about", about.get("description", "")),
        "code_name": next(row["code_name"] for row in _cache["standings"] if row["key"] == key),
        "method": method_svg(plain.get("method", about.get("method", []))),
        "training": {
            "summary": trained.get("summary", ""),
            "stages": [{"title": s["title"], "svg": method_svg(s["steps"], s["repeat"])} for s in trained.get("stages", [])],
            "curves": trained.get("curves", ""),
            "lab": trained.get("lab", False),
        },
    }


def api(_: Request) -> Response:
    return JSONResponse(payload())


def guide(_: Request) -> Response:
    return JSONResponse({"html": GUIDE, "css": GUIDE_STYLE})


def bot(request: Request) -> Response:
    if _cache["signature"] is None:
        payload()
    data = explain(request.query_params.get("key", ""))
    return JSONResponse(data) if data else JSONResponse({"error": "no such bot on the board"}, status_code=404)
