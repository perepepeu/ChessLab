from __future__ import annotations

import chess

from .encoding import material_score
from .model import PolicyNetwork


MATE_SCORE = 10000.0


def evaluate(board: chess.Board, perspective: chess.Color) -> float:
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == perspective else MATE_SCORE
    if board.is_game_over(claim_draw=True):
        return 0.0
    score = material_score(board, perspective)
    mobility = board.legal_moves.count() * (0.015 if board.turn == perspective else -0.015)
    king_pressure = 0.2 if board.is_check() and board.turn != perspective else -0.2 if board.is_check() else 0.0
    return score + mobility + king_pressure


def ordered_moves(model: PolicyNetwork, board: chess.Board, limit: int = 14) -> list[chess.Move]:
    moves, probabilities = model.policy(board, temperature=0.35)
    ranked = sorted(zip(moves, probabilities), key=lambda pair: float(pair[1]), reverse=True)
    tactical = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
    result: list[chess.Move] = []
    for move in tactical + [move for move, _ in ranked]:
        if move not in result:
            result.append(move)
        if len(result) >= limit:
            break
    return result


def _minimax(model: PolicyNetwork, board: chess.Board, depth: int, alpha: float, beta: float,
             perspective: chess.Color) -> float:
    if depth == 0 or board.is_game_over(claim_draw=True):
        return evaluate(board, perspective)
    maximizing = board.turn == perspective
    value = -float("inf") if maximizing else float("inf")
    for move in ordered_moves(model, board, limit=10):
        board.push(move)
        child = _minimax(model, board, depth - 1, alpha, beta, perspective)
        board.pop()
        if maximizing:
            value = max(value, child)
            alpha = max(alpha, value)
        else:
            value = min(value, child)
            beta = min(beta, value)
        if beta <= alpha:
            break
    return value


def choose_with_search(model: PolicyNetwork, board: chess.Board, level: str = "policy",
                       temperature: float = 0.15, sample: bool = False) -> chess.Move | None:
    if level == "policy":
        return model.choose_move(board, temperature, sample)
    perspective = board.turn
    candidates = ordered_moves(model, board, limit=16 if level == "search2" else 24)
    if not candidates:
        return None
    depth = 1 if level == "tactical" else 2
    best_move, best_score = candidates[0], -float("inf")
    for move in candidates:
        board.push(move)
        score = _minimax(model, board, depth - 1, -float("inf"), float("inf"), perspective)
        board.pop()
        if score > best_score:
            best_move, best_score = move, score
    return best_move

