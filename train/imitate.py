"""Train a card-play net to imitate a teacher's recorded decisions (rung 2).

    python -m train.imitate --data ~/workspace/estimation-data/rung2/pimc100 --out bots/pimc-clone/net.npz

Needs only NumPy and PyTorch, so it runs in its own environment (see train/README.md): on the
GTX 1070 that is PyTorch 2.8 with CUDA 12.6, which the engine's environment cannot install.

The net is an MLP from the 456 encoder features to 52 card logits; illegal cards are masked out.
The target mixes the teacher's card (one-hot) with a softmax over the teacher's value of every
legal card, so the student also learns *how much* worse each alternative is:

    target = (1 - soft) * onehot(choice) + soft * softmax(values / temperature)

Held-out games (``--val``) measure two things: agreement (student plays the teacher's card) and
regret, the teacher's value of its own card minus its value of the student's card, in points per
decision. Regret is the number that matters; agreement over-punishes near-ties.

Weights are saved as plain NumPy arrays, so the bot needs no PyTorch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def load(data: Path) -> dict[str, np.ndarray]:
    shards = sorted(data.expanduser().glob("*.npz"))
    if not shards:
        raise SystemExit(f"no .npz shards in {data}")
    parts = [np.load(s) for s in shards]
    keys = ["features", "legal", "values", "choice", "game"]
    if all("hint" in p for p in parts):
        keys.append("hint")
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def baselines(values: np.ndarray, legal: np.ndarray, choice: np.ndarray, hint: np.ndarray | None) -> dict[str, float]:
    """Regret of simple policies under the teacher's values, in points per decision."""
    rows = np.arange(len(choice))
    best = values[rows, choice]
    out = {"random": float((((best[:, None] - values) * legal).sum(1) / legal.sum(1)).mean())}
    if hint is not None:
        out["heuristic"] = float((best - values[rows, hint]).mean())
    return out


class CardNet(nn.Module):
    def __init__(self, features: int, hidden: int, layers: int) -> None:
        super().__init__()
        dims = [features] + [hidden] * layers
        self.body = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))
        self.head = nn.Linear(hidden, 52)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.body:
            x = F.relu(layer(x))
        return self.head(x)


def targets(values: torch.Tensor, legal: torch.Tensor, choice: torch.Tensor, soft: float, temperature: float) -> torch.Tensor:
    hard = F.one_hot(choice.long(), 52).float()
    if soft <= 0:
        return hard
    soft_t = torch.softmax((values / temperature).masked_fill(~legal, float("-inf")), dim=1)
    return (1 - soft) * hard + soft * soft_t


@torch.no_grad()
def evaluate(net: CardNet, d: dict[str, torch.Tensor], batch: int) -> dict[str, float]:
    net.eval()
    agree = regret = 0.0
    n = len(d["choice"])
    for i in range(0, n, batch):
        x, legal = d["features"][i : i + batch], d["legal"][i : i + batch]
        values, choice = d["values"][i : i + batch], d["choice"][i : i + batch].long()
        pick = net(x).masked_fill(~legal, float("-inf")).argmax(1)
        agree += (pick == choice).sum().item()
        regret += (values.gather(1, choice[:, None]) - values.gather(1, pick[:, None])).sum().item()
    net.train()
    return {"agree": agree / n, "regret": regret / n}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m train.imitate", description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="one or more shard folders")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--soft", type=float, default=0.5, help="weight of the value softmax in the target")
    parser.add_argument("--temperature", type=float, default=2.0, help="points per e-fold in the value softmax")
    parser.add_argument("--val", type=float, default=0.05, help="share of games held out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--augment", type=Path, help="suit tables from train.suit_tables: relabel suits at random per row"
    )
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parts = [load(p) for p in args.data]
    common = set.intersection(*(set(p) for p in parts))
    raw = {k: np.concatenate([p[k] for p in parts]) for k in common}
    games = np.unique(raw["game"])
    rng = np.random.default_rng(args.seed)
    val_games = set(rng.choice(games, max(1, int(len(games) * args.val)), replace=False).tolist())
    is_val = np.isin(raw["game"], list(val_games))

    def tensors(mask: np.ndarray) -> dict[str, torch.Tensor]:
        return {
            "features": torch.as_tensor(raw["features"][mask].astype(np.float32), device=device),
            "legal": torch.as_tensor(raw["legal"][mask], device=device),
            "values": torch.as_tensor(raw["values"][mask], device=device),
            "choice": torch.as_tensor(raw["choice"][mask].astype(np.int64), device=device),
        }

    tr, va = tensors(~is_val), tensors(is_val)
    n = len(tr["choice"])
    print(f"{n} train rows, {len(va['choice'])} val rows from {len(games)} games, device {device}", flush=True)
    base = baselines(
        raw["values"][is_val], raw["legal"][is_val], raw["choice"][is_val].astype(np.int64),
        raw["hint"][is_val].astype(np.int64) if "hint" in raw else None,
    )
    print("val regret baselines: " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()) + " pts/decision", flush=True)

    tables = None
    if args.augment:
        t = np.load(args.augment)
        tables = {k: torch.as_tensor(t[k], device=device) for k in ("feature_source", "card_source", "card_destination")}
        if tables["feature_source"].shape[1] != tr["features"].shape[1]:
            raise SystemExit("suit tables do not match the recorded feature count")

    net = CardNet(tr["features"].shape[1], args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((n + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    start = time.perf_counter()
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = order[i : i + args.batch]
            x, legal, values, choice = tr["features"][idx], tr["legal"][idx], tr["values"][idx], tr["choice"][idx]
            if tables is not None:
                p = torch.randint(0, 24, (len(idx),), device=device)
                x = x.gather(1, tables["feature_source"][p])
                legal = legal.gather(1, tables["card_source"][p])
                values = values.gather(1, tables["card_source"][p])
                choice = tables["card_destination"][p].gather(1, choice[:, None])[:, 0]
            logits = net(x).masked_fill(~legal, -1e9)
            target = targets(values, legal, choice, args.soft, args.temperature)
            loss = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        m = evaluate(net, va, args.batch)
        history.append({"epoch": epoch + 1, "loss": total / n, **m})
        print(
            f"epoch {epoch + 1:>2}  loss {total / n:.4f}  val agree {m['agree']:.1%}  "
            f"regret {m['regret']:.3f} pts/decision  {time.perf_counter() - start:.0f} s",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, layer in enumerate(list(net.body) + [net.head]):
        arrays[f"w{k}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        arrays[f"b{k}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    meta["data"] = [str(p) for p in args.data]
    meta["rows"] = {"train": n, "val": int(len(va["choice"]))}
    meta["history"] = history
    meta["baselines"] = base
    np.savez(args.out, **arrays, meta=json.dumps(meta))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
