"""Encode chess.Board states and moves to fixed-shape tensors for the model.

Board: 18 channels × 8 × 8 uint8
  0–5   : white pieces (P, N, B, R, Q, K)
  6–11  : black pieces (P, N, B, R, Q, K)
  12    : side to move (1 = white)
  13–16 : castling rights (WK, WQ, BK, BQ)
  17    : en passant target square

Move: 6 channels × 8 × 8 uint8
  0     : from-square (one-hot)
  1     : to-square   (one-hot)
  2–5   : promotion piece (N, B, R, Q) — all-ones plane if promoting

Total model input: 24 channels × 8 × 8.
"""
from __future__ import annotations

from typing import Iterator

import chess
import numpy as np

from .parse import ParsedGame

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_PROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
N_CHANNELS = 18
N_MOVE_CHANNELS = 6
N_INPUT_CHANNELS = N_CHANNELS + N_MOVE_CHANNELS  # 24


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into a (18, 8, 8) uint8 tensor."""
    arr = np.zeros((N_CHANNELS, 8, 8), dtype=np.uint8)
    for color_idx, color in enumerate((chess.WHITE, chess.BLACK)):
        for pt_idx, pt in enumerate(_PIECE_TYPES):
            channel = color_idx * 6 + pt_idx
            for sq in board.pieces(pt, color):
                arr[channel, chess.square_rank(sq), chess.square_file(sq)] = 1
    if board.turn == chess.WHITE:
        arr[12, :, :] = 1
    if board.has_kingside_castling_rights(chess.WHITE):
        arr[13, :, :] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        arr[14, :, :] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        arr[15, :, :] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        arr[16, :, :] = 1
    if board.ep_square is not None:
        arr[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    return arr


def iter_focal_positions(game: ParsedGame) -> Iterator[tuple[np.ndarray, str]]:
    """Replay the game; yield (board_tensor, move_uci) for each focal-player move."""
    focal_is_white = game.focal_color == "white"
    board = chess.Board()
    for move_uci in game.moves_uci:
        is_focal_turn = (board.turn == chess.WHITE) == focal_is_white
        move = chess.Move.from_uci(move_uci)
        if is_focal_turn:
            yield board_to_tensor(board), move_uci
        board.push(move)


def move_to_channels(move_uci: str) -> np.ndarray:
    """Encode a UCI move string into (6, 8, 8) uint8 channels."""
    arr = np.zeros((N_MOVE_CHANNELS, 8, 8), dtype=np.uint8)
    move = chess.Move.from_uci(move_uci)
    arr[0, chess.square_rank(move.from_square), chess.square_file(move.from_square)] = 1
    arr[1, chess.square_rank(move.to_square), chess.square_file(move.to_square)] = 1
    if move.promotion is not None:
        promo_idx = _PROMO_PIECES.index(move.promotion)  # 0..3
        arr[2 + promo_idx, :, :] = 1
    return arr


def moves_to_channels(moves_uci: list[str]) -> np.ndarray:
    """Vectorized: list of UCI strings -> (N, 6, 8, 8) uint8."""
    if not moves_uci:
        return np.empty((0, N_MOVE_CHANNELS, 8, 8), dtype=np.uint8)
    return np.stack([move_to_channels(m) for m in moves_uci], axis=0)


def encode_game(game: ParsedGame) -> tuple[np.ndarray, list[str]]:
    """Materialize all focal-player decisions in a game.

    Returns (boards, moves) where boards is shape (n_decisions, 18, 8, 8) uint8
    and moves is a list of UCI strings, length n_decisions.
    """
    boards = []
    moves = []
    for b, m in iter_focal_positions(game):
        boards.append(b)
        moves.append(m)
    if not boards:
        return np.empty((0, N_CHANNELS, 8, 8), dtype=np.uint8), []
    return np.stack(boards, axis=0), moves
