#!/usr/bin/env python3
"""The smallest possible wire bot: reads one JSON line per decision, answers one action id.

It never dashes, passes whenever it may, estimates the lowest number it may and plays the
lowest legal card. Replace `choose` with your own logic in any language that can read stdin.
"""
import json
import sys

PASS = 53
NO_DASH = 0


def choose(obs: dict, legal: list[int]) -> int:
    cards = [a for a in legal if a >= 68]
    if cards:
        return min(cards)  # card id = action - 68; lowest id is the lowest club first
    if PASS in legal:
        return PASS
    if NO_DASH in legal:
        return NO_DASH
    return min(legal)  # the lowest legal estimate


def main() -> None:
    for line in sys.stdin:
        msg = json.loads(line)
        if msg["type"] == "act":
            sys.stdout.write(f"{choose(msg['obs'], msg['legal'])}\n")
            sys.stdout.flush()
        elif msg["type"] == "bye":
            return


if __name__ == "__main__":
    main()
