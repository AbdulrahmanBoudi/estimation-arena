"""The wire protocol: a bot is a process. One JSON line per decision in, one integer line out.

The harness writes one JSON object per line to the bot's stdin:

    {"type": "hello", "protocol": 1, "seat_seed": 123, "rules": {...}, "budget_s": 0.25}
    {"type": "act", "obs": {...}, "legal": [68, 70, 81]}      the bot answers "70\\n"
    {"type": "bye"}

``obs`` is ``estimation.engine.Observation`` as JSON with its field names unchanged (see the
engine's ``observation.py`` for what each means), plus ``phase_name`` and ``hand_cards`` (the
hand as a list of card ids, 0-51) for convenience. ``legal`` lists the action ids the bot may
choose from, in the engine's fixed 120-id action space. A reply that is not one of them, or
that takes longer than the hard timeout, ends the game as a loss for that bot.

One process plays one game: the harness starts the command when a game starts and ends it
when the game ends, so a bot cannot carry anything between games (duplicate sets replay the
same deal four times).
"""

from __future__ import annotations

import dataclasses
import json
import os
import selectors
import subprocess
import time
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from estimation.engine import Observation, RulesConfig

PROTOCOL = 1
DEFAULT_TIMEOUT_S = 5.0  # hard limit per decision; the 0.25 s card budget is measured, not enforced
DEFAULT_BUDGET_S = 0.25


class ProtocolError(RuntimeError):
    """The bot process broke the protocol: bad reply, illegal id, timeout or exit."""


def observation_to_dict(obs: Observation) -> dict[str, Any]:
    data = dataclasses.asdict(obs)
    data["phase"] = int(obs.phase)
    data["phase_name"] = obs.phase.name
    data["hand_cards"] = [card for card in range(52) if obs.hand >> card & 1]
    data["with_players"] = [list(w) for w in obs.with_players]
    data["log"] = [list(entry) for entry in obs.log]
    return data


def rules_to_dict(rules: RulesConfig) -> dict[str, Any]:
    return dataclasses.asdict(rules)


def _kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=2)
    except (OSError, subprocess.TimeoutExpired):
        pass


class WireBot:
    """An ``estimation.bots.Bot`` that forwards every decision to a child process."""

    def __init__(
        self,
        command: Sequence[str],
        cwd: Path,
        seat_seed: int,
        rules: RulesConfig,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        budget_s: float = DEFAULT_BUDGET_S,
    ) -> None:
        self.command = list(command)
        self.timeout_s = timeout_s
        stderr = None if os.environ.get("ARENA_BOT_STDERR") else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            self.command,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
        )
        self._finalizer = weakref.finalize(self, _kill, self.proc)
        self._buffer = b""
        self._selector = selectors.DefaultSelector()
        assert self.proc.stdout is not None
        self._selector.register(self.proc.stdout, selectors.EVENT_READ)
        self._send(
            {
                "type": "hello",
                "protocol": PROTOCOL,
                "seat_seed": seat_seed,
                "rules": rules_to_dict(rules),
                "budget_s": budget_s,
            }
        )

    def __repr__(self) -> str:
        return f"WireBot({' '.join(self.command)})"

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise ProtocolError(f"{self!r} exited before the game ended") from error

    def _readline(self, timeout: float) -> bytes:
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ProtocolError(f"{self!r} took more than {timeout:g} s to answer")
            if not self._selector.select(left):
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise ProtocolError(f"{self!r} exited without answering")
            self._buffer += chunk
        line, _, self._buffer = self._buffer.partition(b"\n")
        return line

    def act(self, obs: Observation, legal: list[int]) -> int:
        self._send({"type": "act", "obs": observation_to_dict(obs), "legal": legal})
        line = self._readline(self.timeout_s).strip()
        try:
            action = int(line)
        except ValueError as error:
            raise ProtocolError(f"{self!r} answered {line[:40]!r}, not an action id") from error
        if action not in legal:
            raise ProtocolError(f"{self!r} chose {action}, which is not legal now: {legal}")
        return action

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"type": "bye"})
                assert self.proc.stdin is not None
                self.proc.stdin.close()
            except (ProtocolError, OSError):
                pass
        self._finalizer()
