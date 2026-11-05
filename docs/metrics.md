# Metrics

The same games are scored three ways. Each decides its own set winner, so each has its own Elo. A
training run picks one as its reward (`jaxenv.selfplay --objective`); every check shows all three.

| Metric | Per game | Even match | Elo field | Good at showing |
| --- | --- | --- | --- | --- |
| Weighted rank | King +1, Sub-King +0.2, Sub-Kooz -0.2, Kooz -1 | 0 | `elo` | Overall finishing order, the default ranking |
| King only | 1 for King, else 0, ties shared | 25% | `king_elo` | Whether a bot actually wins the Bola |
| Score delta | Our seats' mean score minus theirs | 0 | `score_elo` | Margin; shows progress after set wins saturate |

Score delta is 1.5 x the engine's RSA in a two-against-two lineup. Its mean can swing on one exploding
game, while its Elo counts only who won each set.

Set-win Elo saturates once a bot wins nearly every set, so read the per-match values with their
intervals as well. Elo also assumes strength is transitive; a cross-table of bots is the check for
that. Result files keep every game's raw scores (`versus[].scores`), so a new metric needs no replay.

## Trying a different objective

```bash
.venv-jax/bin/python -m jaxenv.selfplay --objective king --out runs/run3
```
