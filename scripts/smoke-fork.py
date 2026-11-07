#!/usr/bin/env python3
"""Owner only: run smoke on a pull request from a fork, which GitHub cannot run by itself.

    scripts/smoke-fork.py 30            # what to review, and the command to start the run
    scripts/smoke-fork.py 30 <commit>   # start it for the commit you reviewed
"""

from __future__ import annotations

import json
import subprocess
import sys

REPO = "essamgouda97/estimation-arena"
INSTALL_FILES = ("pyproject.toml", "uv.lock")  # these run with the engine token


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main(argv: list[str]) -> int:
    if len(argv) not in (1, 2):
        print(__doc__)
        return 2
    pr = argv[0]
    if len(argv) == 2:
        gh("workflow", "run", "smoke.yml", "-R", REPO, "--ref", "main", "-f", f"pr={pr}", "-f", f"sha={argv[1]}")
        print(f"Started. Runs: https://github.com/{REPO}/actions/workflows/smoke.yml")
        return 0

    info = json.loads(gh("pr", "view", pr, "-R", REPO, "--json", "author,title,headRefOid,isCrossRepository,files"))
    if not info["isCrossRepository"]:
        print("This pull request is not from a fork, so smoke runs by itself.")
        return 0
    print(f"{info['title']} by {info['author']['login']}\n")
    for file in info["files"]:
        flag = "  <- runs with the engine token" if file["path"] in INSTALL_FILES else ""
        print(f"  +{file['additions']} -{file['deletions']}  {file['path']}{flag}")
    print(f"\nRead it: https://github.com/{REPO}/pull/{pr}/files")
    print(f"Then:    scripts/smoke-fork.py {pr} {info['headRefOid']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
