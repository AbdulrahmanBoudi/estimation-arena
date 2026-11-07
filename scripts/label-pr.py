#!/usr/bin/env python3
"""Label a pull request by what it changes. Reads only GitHub's file list, never runs its code.

    scripts/label-pr.py 30 --dry-run

The labels are defined here and created on first use. Other labels on a pull request are left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

LABELS = {
    "new-strat": ("1a7f37", "Adds a new bot to the leaderboard"),
    "strat-update": ("54aeff", "Changes an existing bot"),
    "engine-update": ("b60205", "Moves to a new engine commit, so every bot may play differently"),
    "model-update": ("8250df", "Adds or changes a trained model"),
    "anchors": ("d93f0b", "Changes the benchmark bots everyone is rated against"),
    "harness": ("6e7781", "Changes how games are run or rated"),
    "training": ("0e8a16", "Changes training or experiments"),
    "ci": ("24292f", "Changes GitHub workflows"),
    "docs": ("c5def5", "Changes documentation only"),
    "fork": ("fbca04", "From a fork: the owner starts smoke by hand"),
}
ENGINE_SOURCE = re.compile(r'source = \{ git = "[^"]*/estimation-engine[^"]*" \}')


def labels_for(files: list[dict[str, str]], engine_changed: bool = False, fork: bool = False) -> set[str]:
    """``files`` as GitHub lists them: each with ``filename`` and ``status``."""
    labels = {"engine-update"} if engine_changed else set()
    if fork:
        labels.add("fork")
    new_bots = {
        f["filename"].split("/")[1]
        for f in files
        if f["status"] == "added" and re.fullmatch(r"bots/[^/]+/manifest\.json", f["filename"])
    }
    if new_bots:
        labels.add("new-strat")
    for name in (f["filename"] for f in files):
        if name.startswith("bots/"):
            if name.split("/")[1] not in new_bots:
                labels.add("strat-update")
        elif name == "models.lock.json" or name.startswith("models/"):
            labels.add("model-update")
        elif name == "anchors.json":
            labels.add("anchors")
        elif name.startswith(".github/"):
            labels.add("ci")
        elif name.startswith(("train/", "jaxenv/", "experiments/")):
            labels.add("training")
        elif name.startswith(("arena/", "scripts/", "tests/")) or name in ("pyproject.toml", "uv.lock"):
            labels.add("harness")
    if files and all(f["filename"].endswith(".md") or f["filename"].startswith("docs/") for f in files):
        labels.add("docs")
    return labels


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def engine_source(repo: str, sha: str) -> str | None:
    lock = gh("api", f"repos/{repo}/contents/uv.lock?ref={sha}", "-H", "Accept: application/vnd.github.raw")
    match = ENGINE_SOURCE.search(lock)
    return match.group() if match else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pr", type=int)
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", "essamgouda97/estimation-arena"))
    parser.add_argument("--dry-run", action="store_true", help="print the labels, change nothing")
    args = parser.parse_args(argv)

    pr = json.loads(gh("api", f"repos/{args.repo}/pulls/{args.pr}"))
    lines = gh("api", "--paginate", f"repos/{args.repo}/pulls/{args.pr}/files", "--jq", ".[] | @json")
    files = [json.loads(line) for line in lines.splitlines()]
    head_repo = (pr["head"].get("repo") or {}).get("full_name", args.repo)
    engine_changed = any(f["filename"] == "uv.lock" for f in files) and (
        engine_source(args.repo, pr["base"]["sha"]) != engine_source(head_repo, pr["head"]["sha"])
    )
    wanted = labels_for(files, engine_changed, fork=head_repo != args.repo)
    print(" ".join(sorted(wanted)) or "no labels")
    if args.dry_run:
        return 0

    existing = {label["name"] for label in json.loads(gh("label", "list", "-R", args.repo, "--limit", "200", "--json", "name"))}
    for name in sorted(wanted - existing):
        color, description = LABELS[name]
        gh("label", "create", name, "-R", args.repo, "--color", color, "--description", description)
    current = {label["name"] for label in pr["labels"]}
    for name in sorted(current & set(LABELS) - wanted):
        gh("api", "-X", "DELETE", f"repos/{args.repo}/issues/{args.pr}/labels/{name}")
    add = sorted(wanted - current)
    if add:
        fields = [arg for name in add for arg in ("-f", f"labels[]={name}")]
        gh("api", "-X", "POST", f"repos/{args.repo}/issues/{args.pr}/labels", *fields)
    return 0


if __name__ == "__main__":
    sys.exit(main())
