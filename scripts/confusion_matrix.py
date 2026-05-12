#!/usr/bin/env python3
"""Build a player-vs-player confusion matrix on the val split.

For each val game, embed it, predict its player via nearest centroid (centroids
computed from a held-out subset of *other* val games for that player). Write
an image + per-player accuracy table.

Usage:
    python scripts/confusion_matrix.py
    python scripts/confusion_matrix.py --checkpoint checkpoints/full_long/best.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from chessfp.model import StyleModel  # noqa: E402
from chessfp.train import embed_games, select_device, split_games_by_player  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[StyleModel, dict, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = StyleModel(
        d_model=cfg["d_model"], cnn_ch=cfg["cnn_ch"], cnn_blocks=cfg["cnn_blocks"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], max_len=cfg["max_len"],
        ffn_dim=cfg["ffn_dim"], dropout=cfg.get("dropout", 0.1),
        n_classes=ckpt.get("n_classes"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "full" / "best.pt")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--out-dir", type=Path, default=ROOT / "viz")
    p.add_argument("--centroid-games", type=int, default=30,
                   help="Val games per player used to form the centroid (held out from queries).")
    p.add_argument("--query-games", type=int, default=60,
                   help="Val games per player used as queries.")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = select_device(args.device)
    log.info("device: %s", device)
    model, cfg, ckpt = load_model(args.checkpoint, device)
    log.info("checkpoint step=%s metric=%s n_classes=%s",
             ckpt.get("step"), ckpt.get("metric"), ckpt.get("n_classes"))

    _, val_games = split_games_by_player(
        args.processed_dir, args.val_frac, min_games=args.centroid_games + 1, seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    centroids: dict[str, np.ndarray] = {}
    queries: dict[str, np.ndarray] = {}
    for pid in sorted(val_games):
        gs = val_games[pid]
        if len(gs) < args.centroid_games + 1:
            log.info("  skip %s — only %d val games", pid, len(gs))
            continue
        idx = rng.permutation(len(gs))
        ref_idx = idx[: args.centroid_games]
        qry_idx = idx[args.centroid_games : args.centroid_games + args.query_games]
        ref_games = [gs[i] for i in ref_idx]
        qry_games = [gs[i] for i in qry_idx]
        ref_emb = embed_games(model, ref_games, device, cfg["max_len"])
        qry_emb = embed_games(model, qry_games, device, cfg["max_len"])
        if ref_emb.size == 0 or qry_emb.size == 0:
            continue
        c = ref_emb.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroids[pid] = c
        queries[pid] = qry_emb
        log.info("  %-22s centroid=%d queries=%d", pid, len(ref_games), len(qry_games))

    pids = sorted(centroids.keys())
    C = np.stack([centroids[p] for p in pids], axis=0)  # (P, D)

    # Build confusion matrix and per-player accuracy.
    n = len(pids)
    cm = np.zeros((n, n), dtype=np.int64)
    for true_idx, pid in enumerate(pids):
        sims = queries[pid] @ C.T   # (n_q, P)
        preds = sims.argmax(axis=1)
        for pred_idx in preds:
            cm[true_idx, pred_idx] += 1

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    overall_top1 = cm.diagonal().sum() / cm.sum()
    log.info("overall top-1 (centroid-based): %.3f", overall_top1)
    log.info("per-class accuracy:")
    for pid, acc in sorted(zip(pids, per_class_acc), key=lambda x: -x[1]):
        log.info("  %-22s %.3f", pid, acc)

    # Plot: row-normalized for visual clarity
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pids, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(pids, fontsize=9)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(
        f"Confusion matrix (row-normalized)  "
        f"step {ckpt.get('step')}, overall top-1 = {overall_top1:.3f}"
    )
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=7)
    fig.tight_layout()
    cm_path = args.out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    log.info("wrote %s", cm_path)
    plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
