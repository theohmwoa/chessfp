#!/usr/bin/env python3
"""Walk data/raw/, parse + encode every game, write data/processed/{player_id}.parquet.

One row per game with:
  - metadata (handle, ratings, time class, result, ECO, ...)
  - moves_uci: list[str]            — focal player's moves only
  - boards_bytes: bytes             — (n_decisions, 18, 8, 8) uint8, .tobytes()
  - n_decisions: int32              — len(moves_uci); needed to reshape boards

Use chessfp.dataset.read_games(path) to load and decode a player file.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chessfp.encode import N_CHANNELS, encode_game  # noqa: E402
from chessfp.parse import ParseStats, iter_player_games  # noqa: E402

DEFAULT_RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "data" / "processed"
DEFAULT_PLAYERS_FILE = ROOT / "players.json"


GAME_SCHEMA = pa.schema([
    ("game_id", pa.string()),
    ("end_time", pa.int64()),
    ("time_class", pa.string()),
    ("time_control", pa.string()),
    ("eco", pa.string()),
    ("focal_handle", pa.string()),
    ("focal_color", pa.string()),
    ("focal_rating", pa.int32()),
    ("opp_handle", pa.string()),
    ("opp_rating", pa.int32()),
    ("result", pa.string()),
    ("n_decisions", pa.int32()),
    ("boards_bytes", pa.binary()),
    ("moves_uci", pa.list_(pa.string())),
])


def build_player(player_dir: Path, out_path: Path) -> dict:
    parse_stats = ParseStats()
    rows = []
    for game in iter_player_games(player_dir, stats=parse_stats):
        boards, moves = encode_game(game)
        if boards.shape[0] == 0:
            continue
        rows.append({
            "game_id": game.game_id,
            "end_time": game.end_time,
            "time_class": game.time_class,
            "time_control": game.time_control,
            "eco": game.eco or "",
            "focal_handle": game.focal_handle,
            "focal_color": game.focal_color,
            "focal_rating": game.focal_rating,
            "opp_handle": game.opp_handle,
            "opp_rating": game.opp_rating,
            "result": game.result,
            "n_decisions": int(boards.shape[0]),
            "boards_bytes": boards.tobytes(),
            "moves_uci": moves,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows, schema=GAME_SCHEMA)
        pq.write_table(table, out_path, compression="zstd")
    return {
        "games_written": len(rows),
        "parse_stats": parse_stats.__dict__,
        "out_path": str(out_path),
        "n_decisions_total": sum(r["n_decisions"] for r in rows),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--players", type=Path, default=DEFAULT_PLAYERS_FILE)
    p.add_argument("--only", nargs="*", default=None,
                   help="Only build these player ids")
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

    summary = {"channels": N_CHANNELS, "players": {}}
    for player in tqdm(players, desc="players", unit="p"):
        pid = player["id"]
        player_dir = args.raw_dir / pid
        if not player_dir.exists():
            logging.info("no raw data for %s, skipping", pid)
            continue
        out_path = args.out_dir / f"{pid}.parquet"
        result = build_player(player_dir, out_path)
        summary["players"][pid] = result
        logging.info(
            "%-22s %5d games, %7d decisions  ->  %s",
            pid, result["games_written"], result["n_decisions_total"], out_path.name,
        )

    summary_path = args.out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary written: {summary_path}")
    print(json.dumps({pid: r["games_written"] for pid, r in summary["players"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
