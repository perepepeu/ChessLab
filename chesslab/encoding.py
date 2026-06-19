from __future__ import annotations

import numpy as np
import chess


PIECE_PLANES = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

INPUT_SIZE = 12 * 64 + 5
OUTPUT_SIZE = 64 * 64


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode pieces, side to move and castling rights into a flat float vector."""
    vector = np.zeros(INPUT_SIZE, dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_PLANES[(piece.color, piece.piece_type)]
        vector[plane * 64 + square] = 1.0
    vector[768] = 1.0 if board.turn == chess.WHITE else -1.0
    vector[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    vector[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    vector[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    vector[772] = float(board.has_queenside_castling_rights(chess.BLACK))
    return vector


def move_to_id(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square


def legal_move_groups(board: chess.Board) -> dict[int, list[chess.Move]]:
    groups: dict[int, list[chess.Move]] = {}
    for move in board.legal_moves:
        groups.setdefault(move_to_id(move), []).append(move)
    return groups


def material_score(board: chess.Board, color: chess.Color) -> float:
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.2, chess.ROOK: 5, chess.QUEEN: 9}
    score = 0.0
    for piece_type, value in values.items():
        score += len(board.pieces(piece_type, color)) * value
        score -= len(board.pieces(piece_type, not color)) * value
    return score

