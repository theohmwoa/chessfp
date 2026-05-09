#!/usr/bin/env python3
"""Fetch chess.com game archives for all curated players into data/raw/.

Resumable: re-running skips months already on disk.

Examples:
    python scripts/fetch_games.py
    python scripts/fetch_games.py --only hikaru_nakamura magnus_carlsen
    python scripts/fetch_games.py --rate 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessfp.fetch import ChessComClient, fetch_player  # noqa: E402

DEFAULT_PLAYERS_FILE = ROOT / "players.json"
DEFAULT_OUT_DIR = ROOT / "data" / "raw"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--players", type=Path, default=DEFAULT_PLAYERS_FILE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--only", nargs="*", default=None, help="Only fetch these player ids")
    p.add_argument("--rate", type=float, default=1.0, help="Min seconds between requests")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = json.loads(args.players.read_text())
    players = cfg["players"]
    if args.only:
        wanted = set(args.only)
        players = [pl for pl in players if pl["id"] in wanted]
        if not players:
            print(f"No players matched --only {args.only}", file=sys.stderr)
            return 2

    client = ChessComClient(rate_limit_s=args.rate)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    totals = {"new_archives": 0, "skipped_archives": 0, "games": 0, "missing_handles": []}
    for player in players:
        logging.info("=== %s (%s) ===", player["name"], player["handle"])
        try:
            s = fetch_player(client, player, args.out_dir)
        except Exception as e:
            logging.exception("failed %s: %s", player["id"], e)
            continue
        totals["new_archives"] += s.new_archives
        totals["skipped_archives"] += s.skipped_archives
        totals["games"] += s.games
        totals["missing_handles"].extend(s.missing_handles)
        logging.info(
            "  -> +%d archives, %d skipped, +%d games",
            s.new_archives, s.skipped_archives, s.games,
        )

    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
