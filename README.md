# Estimation Arena

A competition harness and leaderboard for bots that play **Estimation**, built on
[estimation-engine](https://github.com/YosifElessawi/estimation-engine) (private: the engine,
its rules spec and its bots). This repo holds none of the engine's code or data, only the
things a competition adds:

| Piece | What it is |
|---|---|
| `arena/protocol.py` | The wire protocol: a bot is a process, one JSON line in, one action id out |
| `arena/bots.py` | Submissions under `bots/<name>/` with a `manifest.json`, Python tier or wire tier |
| `arena/run.py` | Plays a submission against the anchors in duplicate sets, writes `results/<name>.json` |
| `arena/leaderboard.py` | One Elo fit over every result and the imported ladder, renders `site/index.html` |
| `anchors.json` | The frozen engine bots everyone is measured against |
| `ladder/pairs.json` | Yosif's ladder-v2 pairwise results, so anchors keep the Elo they have there |
| `.github/workflows` | `smoke` on every bot PR, `ladder` nightly with GitHub Pages publishing |

## How a game is played

```
harness (arena/run.py)                          your bot (one process per game)
  seeded Bola, same deal x 4 rotations   ---->   {"type":"act","obs":{...},"legal":[68,70,81]}
  legal action list, hard timeout        <----   70
  illegal id or timeout = forfeit
```

The observation is the engine's `Observation` as JSON: your hand, every public call, every
played card, scores. Action ids are the engine's fixed 120-id space (0-1 dash, 2-51 bids,
52 with, 53 pass, 54-67 estimates, 68-119 play card id - 68). You only ever pick from `legal`.
A fresh process plays each game, so nothing carries over between the four rotations of a deal.

## Run it locally

Needs [uv](https://docs.astral.sh/uv/) and read access to the engine repo (uv clones it over
HTTPS with your normal git credentials).

```bash
uv sync
uv run pytest -q
uv run python -m arena.run --bot lowcard-wire --sets 4 --workers 4 --progress
uv run python -m arena.leaderboard && open site/index.html
```

Without the engine's `runs/` folder only the four CPU anchors exist (Random, Heuristic, PIMC,
Search bot). To rate against **RL v2** and the **Hybrid** too, get the `runs` zip from Yosif's
Drive and unzip it in this folder so `runs/m4-rules-v2/checkpoints/final.pt` exists;
`anchors.json` picks them up. `runs/` is git-ignored. The Hybrid is slow (16 rollouts per
card), so budget about a minute per set for it.

## Submit a bot

1. Make `bots/<your-name>/manifest.json`:
   - **Wire tier**, any language: `"kind": "wire"`, `"command": ["python3", "bot.py"]`. See
     `bots/lowcard-wire` (40 lines, stdlib only).
   - **Python tier**: `"kind": "python"`, `"entry": "bot.py:make"` where `make(seed, rules)`
     returns an object with `act(obs, legal) -> int`. The engine is importable, so you can
     build on its heuristic, hand evaluation or PIMC search. See `bots/cautious-heuristic`.
2. Run it locally with the command above. An illegal action or a decision over the hard
   timeout (5 s default, `timeout_s` in the manifest) forfeits the game. Card decisions are
   budgeted at 0.25 s; slower ones are counted and shown, not forfeited.
3. Open a pull request. The `smoke` workflow plays 2 sets against Random and Heuristic and
   comments the result. Once merged, the nightly `ladder` run rates it and republishes the page.

Rules are the engine's defaults at the pinned commit in `uv.lock`; a season bumps that pin
deliberately and re-runs everyone.

## Hosting and sharing the leaderboard

The cheapest working setup, which the workflows here assume:

- **Compute**: GitHub Actions for the smoke runs and the nightly ladder against the four CPU
  anchors. Free for a public repo, 2,000 minutes a month for a private one. A nightly run of
  two bots at 20 sets takes well under an hour.
- **Secrets**: `ENGINE_TOKEN`, a fine-grained personal access token with read access to the
  engine repo, so uv can install it in CI. `LADDER_SEED`, any integer, keeps the leaderboard
  deals private so nobody tunes to them.
- **Leaderboard page**: `site/index.html` is committed by the ladder run and published with
  GitHub Pages (Settings, Pages, Source: GitHub Actions). On a **public** repo Pages is free.
  On a **private** repo Pages needs GitHub Pro, so the free alternatives are: make this repo
  public (it contains no engine code or private data, only bot names and Elo), or serve
  `site/` from the home server behind its existing gateway, or just send friends the
  committed `site/index.html`, which is a single self-contained file.
- **RL and Hybrid anchors**: they need `runs/`, which stays off GitHub. Register the home
  server as a self-hosted runner with `runs/` unzipped and change `runs-on` in `ladder.yml`
  to `[self-hosted, estimation]`; everything else stays the same.

## Seeds, fairness and what is measured

- Every result is reproducible: `arena.run --seed S` plays exactly the deals of seed S, and
  the pairwise counts in `results/*.json` are what the Elo fit uses.
- Duplicate sets replay one deal four times with the seats rotated, so luck cancels. Because
  of that, a bot is a new process per game: it never sees the seed or the set index.
- **RSA** is the mean score advantage per game over an anchor. **Elo** is one Bradley-Terry
  fit (Random fixed at 400) over all pairwise results, including the imported engine ladder,
  which is why the anchors on the page match Yosif's ladder.
