# AGENTS.md

Instructions for coding agents working in this repo. Humans: see README.md.

## What this is

A competition for bots that play Estimation. The rules engine is
[estimation-engine](https://github.com/YosifElessawi/estimation-engine), a **private** repo
installed by uv from git. This repo holds the harness, the bots, their results, GPU self-play
and the leaderboard page. Trained models live on Hugging Face, pinned in `models.lock.json`.

## Setup

```bash
uv sync --group rl                       # needs read access to the engine repo
uv run python -m arena.models fetch      # the trained models, checked against the lock
uv run pytest -q                         # must pass before any commit
uv run python -m arena.leaderboard       # the current standings
```

For GPU training, a second environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv-jax uv sync --group jax
```

No access to the engine repo? Stop and ask the owner; do not vendor the engine.
`runs/` and model files are git-ignored; keep them that way. Run the fetch again after a
checkout that changes `models.lock.json`.

## Train a bot

```bash
.venv-jax/bin/python -m jaxenv.selfplay --out runs/my-run --benchmark runs/m4-rules-v2/rlv2.npz
```

It trains from nothing on self-play, past versions and RL v2, and writes `runs/my-run/net.npz`.
`--objective` picks the reward: `weighted`, `king` or `delta`, the three scores on the
leaderboard (`docs/metrics.md`). The log reports the King rate against RL v2; 25% is an even
share. To enter a net, copy `bots/bitter-net` to `bots/<name>/` with your `net.npz` beside it.

## Map

| Path | Role |
|---|---|
| `arena/protocol.py` | Wire protocol: one JSON line in, one action id out, one process per game |
| `arena/bots.py` | Loads `bots/<name>/manifest.json`; `sub:<name>:key=value` passes options |
| `arena/run.py` | Rates one bot vs every anchor, writes `results/<name>.json` |
| `arena/sweep.py` | Tunes a python bot's options vs one opponent |
| `arena/env.py` | PettingZoo AEC environment for RL training |
| `arena/leaderboard.py` | Elo fit, renders `site/` |
| `arena/models.py` | Downloads the models in `models.lock.json`, refuses a wrong checksum |
| `anchors.json` | Frozen engine bots everyone is rated against |
| `jaxenv/` | The rules, RL v2 and self-play training on the GPU; `docs/jax-rules.md` |
| `docs/training-design.md` | How self-play is made GPU-bound |
| `train/` | Imitation (`record`, `imitate`) and self-play RL (`rl_cards`, `rl_bid`); see its README |
| `experiments/` | Write-ups per rung, the TensorBoard board, diagnostic bots that are never rated |

## Adding or changing a bot

1. `bots/<name>/manifest.json` plus code. Python tier: `make(seed, rules, **options)`.
   Wire tier: any language, see `bots/lowcard-wire`.
2. Tune with `arena.sweep` on a **tuning seed** (default 1000). Confirm the winner on a fresh
   seed with at least 1,000 sets before believing it. Trust a difference only when the 95%
   intervals do not overlap.
3. Rate with `arena.run`. Never tune against the leaderboard seed (`LADDER_SEED` secret).
4. Model files don't go in Git. Upload yours to a public Hugging Face repo and add its repo,
   commit, file and sha256 to `models.lock.json`.
5. Open a pull request. The `smoke` workflow plays it and comments; after merging, the ladder
   rates it against every benchmark.

A bot that loads a model should name it under `needs` in its manifest.

## Rules of the game you must not break

- Default `RulesConfig` only. A rules change is a new season, decided by the owner.
- A bot sees only its own `Observation`. No reading other hands, no seeds, no shared state
  between games. Wire bots get a fresh process per game.
- Card decisions should stay under 0.25 s; the hard limit is 5 s.
- Raw scores can explode: an all-lose round doubles the next and doubles stack. Compare bots by
  set wins, first place and Elo, not raw score.

## Compute

- Sweeps over 100 sets and ratings against RL v2 or Hybrid take a while on a laptop; use a
  bigger machine for them if you have one. Training wants a GPU.
- JAX works on old cards. On a GTX 1070, use `jax[cuda12]` as pinned here; PyTorch 2.9 and later
  no longer support it.

## Hygiene

- Never print or commit tokens, `.env` files or secrets. CI secrets: `ENGINE_TOKEN`, `LADDER_SEED`.
- Commit `results/*.json` and `site/` only from real runs; never hand-edit them.
- Keep experiments reproducible: record the command, seed and sets next to the result.
