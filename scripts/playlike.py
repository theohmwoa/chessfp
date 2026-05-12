#!/usr/bin/env python3
"""Given a chess.com handle, return the most stylistically similar pros.

Pipeline:
  1. Load a trained checkpoint.
  2. Embed every game in data/processed/<player>.parquet for each pro,
     average to get a per-pro centroid (cached to disk).
  3. Fetch the input handle's recent games from chess.com.
  4. Embed those games, average to get the user's centroid.
  5. Cosine-rank against the pro centroids.

Examples:
    python scripts/playlike.py MagnusCarlsen
    python scripts/playlike.py YourHandle --since 2025-01 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessfp.dataset import read_games  # noqa: E402
from chessfp.encode import encode_game  # noqa: E402
from chessfp.dataset import GameRecord  # noqa: E402
from chessfp.fetch import ChessComClient, fetch_player  # noqa: E402
from chessfp.model import StyleModel  # noqa: E402
from chessfp.parse import iter_player_games  # noqa: E402
from chessfp.train import embed_games, select_device  # noqa: E402


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
    return model, cfg


def compute_pro_centroids(
    model: StyleModel,
    processed_dir: Path,
    device: torch.device,
    max_len: int,
    cache_path: Path | None = None,
    sample_per_player: int = 200,
) -> dict[str, np.ndarray]:
    """One unit-norm centroid per pro, computed from up to N games each."""
    if cache_path and cache_path.exists():
        log = logging.getLogger(__name__)
        log.info("loading cached centroids from %s", cache_path)
        data = np.load(cache_path)
        return {pid: data[pid] for pid in data.files}

    centroids: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)
    for path in sorted(processed_dir.glob("*.parquet")):
        if path.name.startswith("_"):
            continue
        pid = path.stem
        games = list(read_games(path))
        if not games:
            continue
        if len(games) > sample_per_player:
            idx = rng.choice(len(games), size=sample_per_player, replace=False)
            games = [games[i] for i in idx]
        embs = embed_games(model, games, device, max_len)
        if embs.size == 0:
            continue
        c = embs.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroids[pid] = c

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **centroids)
    return centroids


def fetch_and_parse_user(
    handle: str,
    since: str,
    rate_limit_s: float = 0.7,
) -> list[GameRecord]:
    """Pull a user's recent games and parse them into GameRecord-ish objects."""
    client = ChessComClient(rate_limit_s=rate_limit_s)
    with tempfile.TemporaryDirectory(prefix="chessfp_user_") as tmp:
        tmp_path = Path(tmp)
        fetch_player(client, {"id": "user", "handle": handle, "aliases": []}, tmp_path, since=since)
        user_dir = tmp_path / "user"
        if not user_dir.exists():
            return []
        # Reuse the same parsing+encoding the dataset uses, but in-memory.
        records: list[GameRecord] = []
        for parsed in iter_player_games(user_dir):
            boards, moves = encode_game(parsed)
            if boards.shape[0] == 0:
                continue
            records.append(
                GameRecord(
                    game_id=parsed.game_id,
                    end_time=parsed.end_time,
                    time_class=parsed.time_class,
                    time_control=parsed.time_control,
                    eco=parsed.eco or "",
                    focal_handle=parsed.focal_handle,
                    focal_color=parsed.focal_color,
                    focal_rating=parsed.focal_rating,
                    opp_handle=parsed.opp_handle,
                    opp_rating=parsed.opp_rating,
                    result=parsed.result,
                    boards=boards,
                    moves_uci=moves,
                )
            )
        return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("handle", help="chess.com username to fingerprint")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "full" / "best.pt")
    p.add_argument("--processed-dir", type=Path, default=ROOT / "data" / "processed")
    p.add_argument("--centroids-cache", type=Path, default=ROOT / "checkpoints" / "full" / "centroids.npz")
    p.add_argument("--since", default="2025-01", help="Only consider user's games since YYYY-MM")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--show-top-games", type=int, default=3,
                   help="Show the user's games most similar to the top-matched pro.")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2

    device = select_device(args.device)
    log.info("device: %s", device)

    model, cfg = load_model(args.checkpoint, device)
    log.info("model loaded from %s (step=%s)",
             args.checkpoint, torch.load(args.checkpoint, map_location='cpu', weights_only=False).get("step"))

    centroids = compute_pro_centroids(
        model, args.processed_dir, device, cfg["max_len"], cache_path=args.centroids_cache,
    )
    log.info("%d pro centroids ready", len(centroids))

    log.info("fetching @%s ...", args.handle)
    user_games = fetch_and_parse_user(args.handle, since=args.since)
    if not user_games:
        print(f"no eligible games for @{args.handle} since {args.since}", file=sys.stderr)
        return 1
    log.info("got %d eligible games for @%s", len(user_games), args.handle)

    user_embs = embed_games(model, user_games, device, cfg["max_len"])
    user_centroid = user_embs.mean(axis=0)
    user_centroid = user_centroid / (np.linalg.norm(user_centroid) + 1e-9)

    sims = sorted(
        ((pid, float(user_centroid @ proto)) for pid, proto in centroids.items()),
        key=lambda x: -x[1],
    )

    # Re-scale into "spread" — the raw cosines are usually tight (e.g. 0.95–1.00),
    # so present the rank-percentage relative to the spread for readability.
    raw = np.array([s for _, s in sims])
    if raw.max() - raw.min() > 1e-6:
        spread = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        spread = np.zeros_like(raw)

    print(f"\n=== style match for @{args.handle} ({len(user_games)} games) ===")
    print(f"{'rank':>4}  {'pro':<25} {'cos':>7}  {'spread':>7}  bar")
    for i, ((pid, sim), s) in enumerate(zip(sims[: args.top_k], spread[: args.top_k]), 1):
        bar = "#" * int(s * 40)
        print(f"  {i:2d}.  {pid:<25} {sim:+.4f}  {s:>6.2f}   {bar}")

    # Also: contribution per user game — which of their games are most "like the top pro"?
    if args.show_top_games > 0:
        top_pid, _ = sims[0]
        top_proto = centroids[top_pid]
        per_game = user_embs @ top_proto  # cosines per game
        order = np.argsort(-per_game)
        print(f"\n  Top {args.show_top_games} of @{args.handle}'s games most like {top_pid}:")
        for j in order[: args.show_top_games]:
            g = user_games[j]
            print(f"    cos={per_game[j]:+.3f}  https://www.chess.com/live/game/{g.game_id}  ({g.time_class}, {g.focal_color}, {g.result})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
