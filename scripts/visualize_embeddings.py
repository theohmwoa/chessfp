#!/usr/bin/env python3
"""Embed all val games through a trained checkpoint and project to 2D.

Produces two plots:
  - embeddings_pca.png   — quick PCA projection (linear, fast)
  - embeddings_umap.png  — UMAP projection (nonlinear, slower, usually nicer)

Each point is a game; colour = player. Run after training.

Usage:
    python scripts/visualize_embeddings.py
    python scripts/visualize_embeddings.py --checkpoint checkpoints/full_long/best.pt
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

from chessfp.dataset import read_games  # noqa: E402
from chessfp.model import StyleModel  # noqa: E402
from chessfp.train import embed_games, select_device, split_games_by_player  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[StyleModel, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = StyleModel(
        d_model=cfg["d_model"],
        cnn_ch=cfg["cnn_ch"],
        cnn_blocks=cfg["cnn_blocks"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
        ffn_dim=cfg["ffn_dim"],
        dropout=cfg.get("dropout", 0.1),
        n_classes=ckpt.get("n_classes"),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "full" / "best.pt")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--max-games-per-player", type=int, default=80,
                   help="Cap games per player to keep the plot legible and fast.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "viz")
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

    # Pull val games (same split rules as training so we get held-out).
    _, val_games = split_games_by_player(
        args.processed_dir, args.val_frac, min_games=8, seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    all_embs = []
    all_labels = []
    label_names = []
    for pid in sorted(val_games):
        games = val_games[pid]
        if len(games) == 0:
            continue
        if len(games) > args.max_games_per_player:
            idx = rng.choice(len(games), args.max_games_per_player, replace=False)
            games = [games[i] for i in idx]
        embs = embed_games(model, games, device, cfg["max_len"])
        if embs.size == 0:
            continue
        label_idx = len(label_names)
        label_names.append(pid)
        all_embs.append(embs)
        all_labels.append(np.full(len(embs), label_idx))
        log.info("  %-22s embedded %d games", pid, len(embs))

    X = np.concatenate(all_embs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    log.info("total embeddings: %d × %d, %d players", X.shape[0], X.shape[1], len(label_names))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cmap = plt.cm.get_cmap("tab20", len(label_names))

    # --- PCA ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=args.seed)
    Xp = pca.fit_transform(X)
    log.info("PCA explained var: %s", pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(11, 9))
    for i, name in enumerate(label_names):
        m = y == i
        ax.scatter(Xp[m, 0], Xp[m, 1], s=10, alpha=0.5, color=cmap(i), label=name)
    ax.set_title(
        f"PCA of game embeddings  (n={X.shape[0]} games, {len(label_names)} players, "
        f"ckpt step {ckpt.get('step')}, val top1={ckpt.get('metric'):.3f})"
    )
    ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
    fig.tight_layout()
    pca_path = args.out_dir / "embeddings_pca.png"
    fig.savefig(pca_path, dpi=120)
    log.info("wrote %s", pca_path)
    plt.close(fig)

    # --- UMAP ---
    try:
        import umap  # noqa: E402
        reducer = umap.UMAP(
            n_neighbors=20, min_dist=0.15, metric="cosine", random_state=args.seed
        )
        Xu = reducer.fit_transform(X)
        fig, ax = plt.subplots(figsize=(11, 9))
        for i, name in enumerate(label_names):
            m = y == i
            ax.scatter(Xu[m, 0], Xu[m, 1], s=10, alpha=0.5, color=cmap(i), label=name)
        ax.set_title(
            f"UMAP of game embeddings (cosine metric)  (n={X.shape[0]}, "
            f"ckpt step {ckpt.get('step')}, val top1={ckpt.get('metric'):.3f})"
        )
        ax.legend(loc="best", fontsize=8, ncol=2, framealpha=0.85)
        fig.tight_layout()
        umap_path = args.out_dir / "embeddings_umap.png"
        fig.savefig(umap_path, dpi=120)
        log.info("wrote %s", umap_path)
        plt.close(fig)
    except ImportError:
        log.warning("umap-learn not installed, skipping UMAP plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
