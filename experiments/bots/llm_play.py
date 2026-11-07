"""LLM players: a wire bot that asks Claude or Codex for every decision, by CLI or by SDK.

Each decision is a fresh call that sees only the seat's own view. Tools are switched off, the
call runs in an empty folder, and a reply that used a tool is thrown away. A failed call is
retried; if every try fails the bot plays its lowest legal move and logs a fallback, so one
network hiccup doesn't end a long game. Set ESTIMATION_LLM_LOG to keep a JSON line per decision.

The SDK players need their packages on the python3 the manifest runs, so put a virtualenv with
anthropic and openai-codex first on PATH. The Claude key comes from KEY_FILE and is never logged.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

TRIES = 3
CALL_TIMEOUT_S = 120
KEY_FILE = Path.home() / ".config/home-server/secrets/anthropic-api-key"
CLAUDE_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
SUITS = ("clubs", "diamonds", "hearts", "spades", "no trump")
SYMBOLS = "♣♦♥♠"
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")

RULES = """You are playing Estimation, a 4-player trick-taking card game. Each player plays alone,
13 cards each. Seats are 0-3 and play goes 0, 1, 2, 3, 0. Rank high to low: A K Q J 10 9 8 7 6 5 4 3 2.
Trumps, strongest first: no trump, spades, hearts, diamonds, clubs. A game is 13 open rounds, then
5 fixed-trump rounds (no trump, spades, hearts, diamonds, clubs). Highest total score wins.

Open rounds:
- Dash Call: each player may promise to take zero tricks. It is public and takes them out of the auction.
- Auction: a bid is 4-13 tricks plus a trump; it must beat the top bid (more tricks, or the same with
  a stronger trump). With joins the top bid at its number and ends your own bidding; if that bidder
  raises, you answer. Pass leaves the auction. The top bidder gets a final turn to keep or raise. The
  winner is the Caller. Anyone who called in the winning trump is With, scored like the Caller.
- If nobody bids, the round is skipped and the next round counts double.
Estimation: the others say how many tricks they will take, at most the Caller's number. Estimates can
never total 13, so someone always fails. A total of 14+ is Over and 12 or less is Under. The last
estimator takes one Risk for every 2 the total is away from 13.
Fixed-trump rounds: no Dash Call or auction. Everyone estimates in turn, and the highest estimate is
Caller. Bidding 8+ in another trump is a super call, which opens an auction.
Play: the Caller leads. You must follow the led suit if you can. The highest trump wins the trick,
otherwise the highest card of the led suit. The winner leads next.
Scoring: take exactly your estimate for +10 plus your estimate; otherwise lose the difference. Being
Caller or With, each Risk, and being the only winner or only loser each add or subtract 10. A Dash Call
is worth 25 in an Over round or 33 in an Under round. A call of 8+ is worth its square, or half that
lost on a miss. Three or more Callers and With players double the round. If everyone fails, the round
scores 0 and the next one counts double. Doubles stack.

More detail:
- The first seat to act moves one place each round. Right after the deal, every player missing a whole
  suit is announced, without saying which suit.
- A With player keeps their number as their estimate when it equals the Caller's. Otherwise they choose
  an estimate from their number up to the Caller's. Anyone else who estimates exactly the Caller's number
  also becomes With.
- The last estimator may not choose a number that makes the total 13.
- In a fixed-trump round, on a tied highest estimate the first to say it is Caller and the others are With.
- A super call that changes the trump is scored as usual, and that fixed trump is played again next round.
"""


def card_name(card: int) -> str:
    return RANKS[card % 13] + SYMBOLS[card // 13]


def action_name(action: int) -> str:
    if action == 0:
        return "no Dash Call"
    if action == 1:
        return "Dash Call"
    if action <= 51:
        tricks, trump = divmod(action - 2, 5)
        return f"bid {tricks + 4} {SUITS[trump]}"
    if action == 52:
        return "With"
    if action == 53:
        return "Pass"
    if action <= 67:
        return f"estimate {action - 54}"
    return f"play {card_name(action - 68)}"


def _history(log: list) -> str:
    calls = [f"seat {seat}: {action_name(action)}" for seat, action in log if action < 68]
    return "; ".join(calls) or "none"


def prompt_for(obs: dict, legal: list[int]) -> str:
    """Everything this seat may know, in words, without the rules, which every player adds in its
    own way. ``obs`` holds no other seat's unplayed cards."""
    hand = sorted(obs["hand_cards"], key=lambda c: (c // 13, -(c % 13)))
    lines = [
        f"You are seat {obs['seat']}. Round {obs['round_no']}. Phase: {obs['phase_name']}.",
        f"Game scores by seat: {list(obs['scores'])}. This round's multiplier: x{obs['carry']}.",
        f"Your hand: {' '.join(card_name(c) for c in hand)}.",
    ]
    if obs.get("fixed_trump") is not None:
        lines.append(f"Fixed trump this round: {SUITS[obs['fixed_trump']]}.")
    if obs.get("trump") is not None:
        lines.append(f"Trump: {SUITS[obs['trump']]}.")
    if obs.get("void_announced"):
        lines.append(f"Seats missing at least one suit: {list(obs['void_announced'])}.")
    if obs.get("dash_callers"):
        lines.append(f"Dash callers: {list(obs['dash_callers'])}.")
    lines.append(f"Calls this round: {_history(obs['log'])}.")
    if obs.get("top_bid") is not None:
        lines.append(f"Top bid: {action_name(obs['top_bid'])}, by seat {obs['top_bidder']}.")
    if obs.get("caller") is not None:
        lines.append(f"Caller: seat {obs['caller']}. With players: {[w[0] for w in obs['with_players']]}.")
    if any(e is not None for e in obs["estimates"]):
        lines.append(f"Estimates by seat: {list(obs['estimates'])}.")
    if obs.get("over") is not None:
        lines.append(f"Total {obs['total']}, {'Over' if obs['over'] else 'Under'}. Risks: {obs['risks']}.")
    for k, (leader, cards, winner) in enumerate(obs["tricks"], 1):
        lines.append(f"Trick {k}: seat {leader} led {' '.join(card_name(c) for c in cards)}; seat {winner} won.")
    if obs["phase_name"] == "PLAY":
        lines.append(f"Tricks won by seat: {list(obs['tricks_won'])}.")
        table = " ".join(f"seat {seat}: {card_name(card)}" for seat, card in obs["trick"]) or "nothing, you lead"
        lines.append(f"On the table: {table}.")
    lines.append("Legal moves:")
    lines += [f"{action} = {action_name(action)}" for action in legal]
    lines.append("Think as you need, then end with a final line holding only the number of your chosen move.")
    return "\n".join(lines)


def parse_reply(reply: str, legal: list[int]) -> int | None:
    """The move in a reply's last line, if that line is just one legal number."""
    lines = (reply or "").strip().splitlines()
    match = re.fullmatch(r"\W*(\d+)\W*", lines[-1]) if lines else None
    return int(match.group(1)) if match and int(match.group(1)) in legal else None


def ask_claude(prompt: str, folder: str) -> tuple[str, dict]:
    command = ["claude", "-p", f"{RULES}\n{prompt}", "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
               "--setting-sources", "", "--no-session-persistence", "--output-format", "json"]  # fmt: skip
    if os.environ.get("ESTIMATION_LLM_CLAUDE_MODEL"):
        command += ["--model", os.environ["ESTIMATION_LLM_CLAUDE_MODEL"]]
    done = subprocess.run(command, cwd=folder, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=CALL_TIMEOUT_S)
    data = json.loads(done.stdout)
    if data.get("is_error"):
        raise RuntimeError(str(data.get("result"))[:200])
    if data.get("num_turns", 1) > 1:
        raise RuntimeError("the reply took more than one turn, so a tool may have run")
    return data.get("result", ""), {"usage": data.get("usage"), "model": list(data.get("modelUsage") or {})}


def ask_codex(prompt: str, folder: str) -> tuple[str, dict]:
    command = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
               "--sandbox", "read-only", "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
               "-c", "web_search=disabled", "--json", f"{RULES}\n{prompt}"]  # fmt: skip
    if os.environ.get("ESTIMATION_LLM_CODEX_MODEL"):
        command += ["-m", os.environ["ESTIMATION_LLM_CODEX_MODEL"]]
    done = subprocess.run(command, cwd=folder, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=CALL_TIMEOUT_S)
    reply, usage = "", None
    for line in done.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message":
            reply = item.get("text", "")
        elif item.get("type") not in (None, "reasoning"):
            raise RuntimeError(f"the reply used a tool: {item.get('type')}")
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    return reply, {"usage": usage}


def ask_claude_api(prompt: str, client) -> tuple[str, dict]:
    """One decision through the Anthropic SDK. The rules are a cached system block.

    Fallbacks are on: a refused request reruns on another model, so the log records which model
    answered. A move from any other model counts as a swap in experiments/llm_report.py.
    """
    response = client.beta.messages.create(
        model=os.environ.get("ESTIMATION_LLM_CLAUDE_MODEL", CLAUDE_MODEL),
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": os.environ.get("ESTIMATION_LLM_EFFORT", "medium")},
        system=[{"type": "text", "text": RULES, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise RuntimeError(f"refused: {category}")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("ran out of tokens before answering")
    reply = "".join(block.text for block in response.content if block.type == "text")
    usage = response.usage
    return reply, {
        "model": response.model,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        },
    }


def claude_client():
    import anthropic

    return anthropic.Anthropic(api_key=KEY_FILE.read_text().strip(), timeout=CALL_TIMEOUT_S, max_retries=2)


ALLOWED_ITEMS = {"userMessage", "reasoning", "agentMessage"}


def _item_type(item) -> str:
    return getattr(getattr(item, "root", item), "type", type(item).__name__)


class CodexPlayer:
    """One Codex app-server per game and a fresh thread per decision, so no move remembers another.

    A call can't be timed out from the SDK, so an alarm does it, and the server is restarted after
    any failure because a stuck turn may still hold it.
    """

    OVERRIDES = ("features.shell_tool=false", "features.unified_exec=false", 'web_search="disabled"', "mcp_servers={}")

    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.codex = None

    def _start(self):
        import openai_codex

        self.codex = openai_codex.Codex(openai_codex.CodexConfig(config_overrides=self.OVERRIDES, cwd=self.folder))
        self.codex.__enter__()
        return openai_codex

    def close(self) -> None:
        if self.codex is not None:
            codex, self.codex = self.codex, None
            try:
                codex.__exit__(None, None, None)
            except Exception:  # noqa: BLE001, S110 - it's already being replaced
                pass

    def __call__(self, prompt: str) -> tuple[str, dict]:
        def expired(*_):
            raise TimeoutError("no answer in time")

        previous = signal.signal(signal.SIGALRM, expired)
        signal.alarm(CALL_TIMEOUT_S)
        try:
            sdk = self._start() if self.codex is None else __import__("openai_codex")
            thread = self.codex.thread_start(
                base_instructions=RULES, ephemeral=True, cwd=self.folder,
                approval_mode=sdk.ApprovalMode.deny_all, sandbox=sdk.Sandbox.read_only,
            )  # fmt: skip
            result = thread.run(prompt)
            return codex_reply(result)
        except Exception as error:
            self.close()
            raise RuntimeError(f"{type(error).__name__}: {str(error)[:160]}") from error
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def codex_reply(result) -> tuple[str, dict]:
    """The reply of a finished turn, refusing one that used anything but thinking and text."""
    used = {_item_type(item) for item in result.items} - ALLOWED_ITEMS
    if used:
        raise RuntimeError(f"the reply used a tool: {sorted(used)}")
    last = result.usage.last if result.usage else None
    usage = {
        "input_tokens": last.input_tokens,
        "cached_input_tokens": last.cached_input_tokens,
        "output_tokens": last.output_tokens,
        "reasoning_output_tokens": last.reasoning_output_tokens,
    } if last else None  # fmt: skip
    return result.final_response or "", {"usage": usage}


ASK = {"claude": ask_claude, "codex": ask_codex}


def decide(obs: dict, legal: list[int], ask: Callable[[str], tuple[str, dict]], log: Callable[[dict], None]) -> int:
    """A forced move costs no call. Otherwise ask, retry on failure, and log what happened."""
    entry = {"seat": obs["seat"], "round": obs["round_no"], "phase": obs["phase_name"], "legal": len(legal)}
    if len(legal) == 1:
        log({**entry, "status": "forced", "action": legal[0]})
        return legal[0]
    prompt = prompt_for(obs, legal)
    errors = []
    for attempt in range(1, TRIES + 1):
        start = time.perf_counter()
        try:
            reply, extra = ask(prompt)
        except Exception as error:  # noqa: BLE001 - any failed call is retried, then logged
            errors.append(f"{type(error).__name__}: {str(error)[:160]}")
            continue
        action = parse_reply(reply, legal)
        seconds = round(time.perf_counter() - start, 2)
        if action is not None:
            log({**entry, "status": "ok", "action": action, "tries": attempt, "seconds": seconds, **extra, "errors": errors})
            return action
        errors.append(f"bad reply: {(reply or '')[:80]!r}")
    log({**entry, "status": "fallback", "action": min(legal), "errors": errors})
    return min(legal)


def main(provider: str) -> None:
    path = os.environ.get("ESTIMATION_LLM_LOG")

    def log(entry: dict) -> None:
        if path:
            with open(path, "a") as out:
                out.write(json.dumps({"bot": provider, **entry}) + "\n")

    with tempfile.TemporaryDirectory() as folder:
        codex = CodexPlayer(folder) if provider == "codex-sdk" else None
        if provider == "claude-api":
            client = claude_client()
            ask = lambda prompt: ask_claude_api(prompt, client)  # noqa: E731
        else:
            ask = codex or (lambda prompt: ASK[provider](prompt, folder))
        try:
            for line in sys.stdin:
                message = json.loads(line)
                if message["type"] == "act":
                    print(decide(message["obs"], message["legal"], ask, log), flush=True)
                elif message["type"] == "bye":
                    return
        finally:
            if codex:
                codex.close()
