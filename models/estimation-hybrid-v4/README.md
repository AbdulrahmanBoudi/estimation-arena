---
license: mit
tags:
  - reinforcement-learning
  - card-games
  - estimation
---

# Hybrid v4 (Yosif's bot)

By Yosif Elessawi, under the MIT licence. `hybrid.pt` is a PyTorch checkpoint holding only weights and numbers: a bid net, a card net and a guess net that estimates where each hidden card is. The nets were trained by self-play and then by copying a deeper search. No human games were used.

At play time the bid net picks the bid. For each card, the bot keeps the card net's top 2 choices, imagines 32 deals of the hidden hands from the guess net that fit the bids so far, plays each deal to the end with the card net in all four seats, and plays the choice with the best average score.

Rated on the [Estimation Arena leaderboard](https://essamgouda97.github.io/estimation-arena/). Code and how to run it: [estimation-arena](https://github.com/essamgouda97/estimation-arena) (`bots/hybrid-v4`). Running it currently needs the rules engine, which is private.
