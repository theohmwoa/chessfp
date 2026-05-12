#!/usr/bin/env python3
"""Plot training curves from a history.json produced by train.py.

Three panels:
  - CE loss (training, per log-step)
  - train accuracy + val top-1 (per eval-step)
  - within/between cosine + separation (per eval-step)

Usage:
    python scripts/plot_history.py
    python scripts/plot_history.py --history checkpoints/full_long/history.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", type=Path, default=ROOT / "checkpoints" / "full_long" / "history.json")
    p.add_argument("--out", type=Path, default=ROOT / "viz" / "training_curve.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = json.loads(args.history.read_text())
    train_rows = [r for r in hist if "loss" in r and "k5_top1" not in r]
    eval_rows = [r for r in hist if "k5_top1" in r]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # Panel 1: losses
    ax = axes[0]
    steps = [r["step"] for r in train_rows]
    ax.plot(steps, [r.get("loss", 0) for r in train_rows], label="total", linewidth=1.2)
    if any("ce" in r for r in train_rows):
        ax.plot(steps, [r.get("ce", 0) for r in train_rows], label="ce", linewidth=1.0, alpha=0.8)
    if any(r.get("sup", 0) > 0 for r in train_rows):
        ax.plot(steps, [r.get("sup", 0) for r in train_rows], label="supcon", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: accuracy
    ax = axes[1]
    ax.plot(steps, [r.get("acc", 0) for r in train_rows], label="train acc", linewidth=1.2, color="tab:orange")
    if eval_rows:
        e_steps = [r["step"] for r in eval_rows]
        ax.plot(e_steps, [r["k5_top1"] for r in eval_rows], "o-",
                label="val top1", color="tab:green")
        ax.plot(e_steps, [r["k5_top5"] for r in eval_rows], "x--",
                label="val top5", color="tab:green", alpha=0.5)
        # Random baselines
        if "n_eligible" in eval_rows[0]:
            n = eval_rows[0]["n_eligible"]
            ax.axhline(1.0 / n, color="gray", linestyle=":", label=f"random top1 (1/{n})")
            ax.axhline(5.0 / n, color="gray", linestyle="--", alpha=0.4, label=f"random top5 (5/{n})")
    ax.set_ylabel("accuracy")
    ax.set_title("train accuracy & val k-shot accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Panel 3: embedding separation
    ax = axes[2]
    if eval_rows:
        e_steps = [r["step"] for r in eval_rows]
        ax.plot(e_steps, [r.get("within_cos", 0) for r in eval_rows], "o-",
                label="within (same player)", color="tab:blue")
        ax.plot(e_steps, [r.get("between_cos", 0) for r in eval_rows], "o-",
                label="between (different)", color="tab:red")
        ax.plot(e_steps, [r.get("separation", 0) for r in eval_rows], "s-",
                label="separation", color="tab:purple", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("cosine similarity")
    ax.set_title("embedding cosine geometry")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
