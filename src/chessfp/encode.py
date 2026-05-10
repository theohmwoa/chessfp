"""Encode chess.Board states to fixed-shape tensors for the model.

18 channels × 8 × 8 uint8:
  0–5   : white pieces (P, N, B, R, Q, K)
  6–11  : black pieces (P, N, B, R, Q, K)
  12    : side to move (1 = white)
  13–16 : castling rights (WK, WQ, BK, BQ)
  17    : en passant target square
"""
from __future__ import annotations

from typing import Iterator

import chess
import numpy as np

from .parse import ParsedGame

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
N_CHANNELS = 18


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
