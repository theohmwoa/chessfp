#!/usr/bin/env python3
"""Train the chess style model on processed parquet data.

Examples:
    python scripts/train.py --steps 500
    python scripts/train.py --steps 5000 --n-players-per-batch 12 --eval-every 250
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessfp.train import TrainConfig, train  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-players-per-batch", type=int, default=8)
    p.add_argument("--games-per-player", type=int, default=4)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--loss-mode", default="ce", choices=["ce", "supcon", "ce+supcon"],
                   help="ce: classification only; supcon: contrastive only; ce+supcon: both")
    p.add_argument("--supcon-weight", type=float, default=1.0)
    p.add_argument("--variance-weight", type=float, default=0.0,
                   help="VICReg variance reg; only useful when not using CE")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--min-games-per-player", type=int, default=8)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = TrainConfig(
        processed_dir=args.processed_dir,
        out_dir=args.out_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        steps=args.steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        n_players_per_batch=args.n_players_per_batch,
        games_per_player=args.games_per_player,
        max_len=args.max_len,
        loss_mode=args.loss_mode,
        supcon_weight=args.supcon_weight,
        variance_weight=args.variance_weight,
        temperature=args.temperature,
        val_frac=args.val_frac,
        min_games_per_player=args.min_games_per_player,
        device=args.device,
        seed=args.seed,
    )
    result = train(cfg)
    print(f"\nbest top1: {result['best_metric']:.3f} on {result['n_train_players']} players")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
