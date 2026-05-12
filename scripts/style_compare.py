#!/usr/bin/env python3
"""Compare the playing styles of two chess.com handles directly.

For each handle: fetch recent games, encode them, average to get a single
style centroid. Then report cosine similarity between the two centroids, AND
for each, the top-3 most stylistically similar pros from the training pool.

Examples:
    python scripts/style_compare.py Hikaru MagnusCarlsen
    python scripts/style_compare.py YourHandle FriendHandle --since 2024-01
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessfp.train import embed_games, select_device  # noqa: E402

# Reuse helpers from playlike.py via direct import.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("playlike", ROOT / "scripts" / "playlike.py")
_playlike = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_playlike)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("handle_a")
    p.add_argument("handle_b")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "full_long" / "best.pt")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--centroids-cache", type=Path,
                   default=ROOT / "checkpoints" / "full_long" / "centroids.npz")
    p.add_argument("--since", default="2025-01")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = select_device(args.device)
    model, cfg = _playlike.load_model(args.checkpoint, device)
    centroids = _playlike.compute_pro_centroids(
        model, args.processed_dir, device, cfg["max_len"], cache_path=args.centroids_cache,
    )

    centroid_for: dict[str, np.ndarray] = {}
    n_games_for: dict[str, int] = {}
    for handle in (args.handle_a, args.handle_b):
        log.info("fetching @%s ...", handle)
        games = _playlike.fetch_and_parse_user(handle, since=args.since)
        if not games:
            print(f"no eligible games for @{handle} since {args.since}", file=sys.stderr)
            return 1
        embs = embed_games(model, games, device, cfg["max_len"])
        c = embs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroid_for[handle] = c
        n_games_for[handle] = len(games)
        log.info("  embedded %d games", len(games))

    cos_ab = float(centroid_for[args.handle_a] @ centroid_for[args.handle_b])

    # Reference: average pairwise cosine among the pro centroids — gives a
    # sense of "what's a high similarity in this embedding space."
    pros = np.stack(list(centroids.values()), axis=0)
    pair_sims = pros @ pros.T
    np.fill_diagonal(pair_sims, np.nan)
    pro_mean = float(np.nanmean(pair_sims))
    pro_max = float(np.nanmax(pair_sims))
    pro_min = float(np.nanmin(pair_sims))

    print(f"\n=== style comparison: @{args.handle_a} vs @{args.handle_b} ===\n")
    print(f"  @{args.handle_a}: {n_games_for[args.handle_a]} games")
    print(f"  @{args.handle_b}: {n_games_for[args.handle_b]} games")
    print(f"\n  cosine similarity:  {cos_ab:+.4f}")
    print(f"  reference scale (pro↔pro centroids):")
    print(f"    min = {pro_min:+.4f}   mean = {pro_mean:+.4f}   max = {pro_max:+.4f}")
    if cos_ab > pro_mean + 0.5 * (pro_max - pro_mean):
        verdict = "very similar — these two players are stylistically closer than the average pro pair"
    elif cos_ab > pro_mean:
        verdict = "somewhat similar — closer than the average pro pair, but not unusually close"
    elif cos_ab > pro_mean - 0.5 * (pro_mean - pro_min):
        verdict = "moderate distance — typical for unrelated pros"
    else:
        verdict = "distant — less similar than the typical pro pair"
    print(f"\n  verdict: {verdict}\n")

    pro_ids = list(centroids.keys())
    P = np.stack([centroids[p] for p in pro_ids], axis=0)
    for handle in (args.handle_a, args.handle_b):
        sims = centroid_for[handle] @ P.T
        order = np.argsort(-sims)
        print(f"  top {args.top_k} pros for @{handle}:")
        for j in order[: args.top_k]:
            print(f"    {pro_ids[j]:<22} cos={sims[j]:+.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
