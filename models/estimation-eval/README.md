---
license: mit
pretty_name: Estimation Arena eval set
tags:
  - reinforcement-learning
  - card-games
  - estimation
configs:
  - config_name: default
    data_files: games.jsonl
---

# Estimation Arena eval set

Fixed practice deals for the card game Estimation, with full games played on them by bots and reference scores to compare against. Only bots play, so there are no human games here. MIT licence.

Code and leaderboard: [estimation-arena](https://github.com/essamgouda97/estimation-arena).

## Reference scores

{{sets}} duplicate sets per pairing: each deal is played four times with the seats rotated, so card luck cancels. All pairings use the same deals. An even match is 25% King, 0 weighted rank and 0 score difference. `reference.json` has 95% intervals.

{{table}}

## Files

- `games.jsonl`: one game per line.
- `reference.json`: the table above with intervals, the practice seed and the code version.

## A game

| Field | Meaning |
|---|---|
| `seats` | The bot in each of the four seats |
| `scores` | Final score of each seat |
| `deal_seed` | The seed all four games of a set share |
| `deals` | Each deal in order: `round`, `replay`, and `hands`, the 13 cards of each seat |
| `actions` | Every action in order, from the first decision of the game |

A card is `suit * 13 + rank`, with suits clubs, diamonds, hearts, spades (0 to 3) and rank 0 for a two up to 12 for an ace.

| Action | Meaning |
|---|---|
| 0, 1 | No dash call, dash call |
| 2 to 51 | Bid: `2 + (tricks - 4) * 5 + trump`, trump 0 to 4 for clubs, diamonds, hearts, spades, no trump |
| 52, 53 | With, pass |
| 54 to 67 | Estimate: `action - 54` tricks |
| 68 to 119 | Play the card `action - 68` |

## Limits

These are for comparing and training, not official ratings. Ratings come from the leaderboard, which uses deals kept separate from this set. Playing a new bot on these deals with the same scoring needs the game engine, which is private; the hands are all here, so the public GPU environment in the repo can use them too.
