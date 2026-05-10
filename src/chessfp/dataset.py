"""Reader for the parquet datasets produced by scripts/build_dataset.py."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq

from .encode import N_CHANNELS


@dataclass
class GameRecord:
    game_id: str
    end_time: int
    time_class: str
    time_control: str
    eco: str
    focal_handle: str
    focal_color: str
    focal_rating: int
    opp_handle: str
    opp_rating: int
    result: str
    boards: np.ndarray  # (n_decisions, 18, 8, 8) uint8
    moves_uci: list[str]


def _row_to_record(row: dict) -> GameRecord:
    n = int(row["n_decisions"])
    boards = np.frombuffer(row["boards_bytes"], dtype=np.uint8).reshape(n, N_CHANNELS, 8, 8)
    return GameRecord(
        game_id=row["game_id"],
        end_time=int(row["end_time"]),
        time_class=row["time_class"],
        time_control=row["time_control"],
        eco=row["eco"],
        focal_handle=row["focal_handle"],
        focal_color=row["focal_color"],
        focal_rating=int(row["focal_rating"]),
        opp_handle=row["opp_handle"],
        opp_rating=int(row["opp_rating"]),
        result=row["result"],
        boards=boards,
        moves_uci=list(row["moves_uci"]),
    )


def read_games(path: Path) -> Iterator[GameRecord]:
    """Stream GameRecords from a player parquet."""
    table = pq.read_table(path)
    for row in table.to_pylist():
        yield _row_to_record(row)


def player_id_from_path(path: Path) -> str:
    return path.stem
