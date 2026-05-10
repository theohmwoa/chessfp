"""Parse chess.com archive JSON files into filtered, normalized game records."""
from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn

log = logging.getLogger(__name__)

# v1 filter defaults — favour signal-rich games:
# rapid + blitz only (bullet is too time-pressured to be stylistically clean;
# daily allows engine assistance), rated only, standard chess only,
# minimum length to skip flagrant pre-resigns / disconnects.
DEFAULT_TIME_CLASSES = frozenset({"rapid", "blitz"})
DEFAULT_MIN_PLIES = 20  # ~10 full moves


@dataclass
class ParsedGame:
    game_id: str
    url: str
    end_time: int
    time_class: str
    time_control: str
    rated: bool
    rules: str
    eco: str | None
    focal_handle: str
    focal_color: str  # "white" or "black"
    focal_rating: int
    opp_handle: str
    opp_rating: int
    result: str  # "win" | "loss" | "draw" from focal player's perspective
    moves_uci: list[str] = field(default_factory=list)

    @property
    def n_plies(self) -> int:
        return len(self.moves_uci)


@dataclass
class ParseStats:
    total: int = 0
    parsed: int = 0
    skipped_unrated: int = 0
    skipped_time_class: int = 0
    skipped_rules: int = 0
    skipped_short: int = 0
    skipped_no_focal: int = 0
    skipped_pgn_error: int = 0
    skipped_nonstandard_start: int = 0


def _focal_result(game_dict: dict, focal_color: str) -> str:
    other_color = "black" if focal_color == "white" else "white"
    if game_dict[focal_color]["result"] == "win":
        return "win"
    if game_dict[other_color]["result"] == "win":
        return "loss"
    return "draw"


def _extract_moves(pgn_text: str) -> list[str]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("read_game returned None")
    moves = []
    board = game.board()
    for move in game.mainline_moves():
        moves.append(move.uci())
        board.push(move)
    return moves


def parse_game(
    game_dict: dict,
    focal_handle: str,
    *,
    time_classes: frozenset[str] = DEFAULT_TIME_CLASSES,
    min_plies: int = DEFAULT_MIN_PLIES,
    rated_only: bool = True,
    rules: str = "chess",
    stats: ParseStats | None = None,
) -> ParsedGame | None:
    """Parse one game dict. Returns None if filtered out or malformed."""
    if stats is not None:
        stats.total += 1

    if rated_only and not game_dict.get("rated", False):
        if stats: stats.skipped_unrated += 1
        return None
    if game_dict.get("rules") != rules:
        if stats: stats.skipped_rules += 1
        return None
    tc = game_dict.get("time_class")
    if tc not in time_classes:
        if stats: stats.skipped_time_class += 1
        return None
    # Only standard starting positions — encoder replays from chess.Board()
    if game_dict.get("initial_setup", "") not in ("", chess.STARTING_FEN):
        if stats: stats.skipped_nonstandard_start += 1
        return None

    fh = focal_handle.lower()
    white_handle = game_dict["white"]["username"]
    black_handle = game_dict["black"]["username"]
    if white_handle.lower() == fh:
        focal_color = "white"
    elif black_handle.lower() == fh:
        focal_color = "black"
    else:
        if stats: stats.skipped_no_focal += 1
        return None

    other_color = "black" if focal_color == "white" else "white"
    try:
        moves = _extract_moves(game_dict["pgn"])
    except Exception as e:
        log.debug("pgn parse failed for %s: %s", game_dict.get("url"), e)
        if stats: stats.skipped_pgn_error += 1
        return None

    if len(moves) < min_plies:
        if stats: stats.skipped_short += 1
        return None

    parsed = ParsedGame(
        game_id=game_dict.get("uuid", game_dict.get("url", "")),
        url=game_dict.get("url", ""),
        end_time=game_dict.get("end_time", 0),
        time_class=tc,
        time_control=game_dict.get("time_control", ""),
        rated=bool(game_dict.get("rated", False)),
        rules=game_dict.get("rules", "chess"),
        eco=game_dict.get("eco"),
        focal_handle=focal_handle,
        focal_color=focal_color,
        focal_rating=int(game_dict[focal_color].get("rating", 0)),
        opp_handle=game_dict[other_color]["username"],
        opp_rating=int(game_dict[other_color].get("rating", 0)),
        result=_focal_result(game_dict, focal_color),
        moves_uci=moves,
    )
    if stats: stats.parsed += 1
    return parsed


def parse_archive_file(
    archive_path: Path,
    *,
    time_classes: frozenset[str] = DEFAULT_TIME_CLASSES,
    min_plies: int = DEFAULT_MIN_PLIES,
    stats: ParseStats | None = None,
) -> Iterator[ParsedGame]:
    """Iterate parsed games from a chess.com archive JSON file.

    The focal player's handle is inferred from the parent directory name —
    files live at data/raw/{player_id}/{handle}/{YYYY-MM}.json
    """
    focal_handle = archive_path.parent.name
    data = json.loads(archive_path.read_text())
    for game_dict in data.get("games", []):
        parsed = parse_game(
            game_dict,
            focal_handle,
            time_classes=time_classes,
            min_plies=min_plies,
            stats=stats,
        )
        if parsed is not None:
            yield parsed


def iter_player_games(
    player_dir: Path,
    *,
    time_classes: frozenset[str] = DEFAULT_TIME_CLASSES,
    min_plies: int = DEFAULT_MIN_PLIES,
    stats: ParseStats | None = None,
) -> Iterator[ParsedGame]:
    """Iterate every parsed game for a given player_id (across all alias dirs)."""
    for archive_path in sorted(player_dir.rglob("*.json")):
        yield from parse_archive_file(
            archive_path, time_classes=time_classes, min_plies=min_plies, stats=stats
        )
