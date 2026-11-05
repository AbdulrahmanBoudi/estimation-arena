"""Write an RL checkpoint's weights to an .npz that needs neither PyTorch nor the engine to load.

    python -m train.export_rlv2 runs/m4-rules-v2/checkpoints/final.pt rlv2.npz
"""

import argparse

import numpy as np
import torch


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("out")
    a = p.parse_args(argv)
    payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    arrays = {f"{net}/{name}": t.numpy() for net in ("card", "bidding") for name, t in payload[net].items()}
    np.savez(a.out, **arrays)
    print(f"wrote {a.out}: {len(arrays)} arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
