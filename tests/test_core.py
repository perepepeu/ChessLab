from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from chesslab.data import DatasetStore
from chesslab.encoding import INPUT_SIZE, OUTPUT_SIZE, encode_board, move_to_id
from chesslab.model import PolicyNetwork
from chesslab.replays import ReplayStore
from chesslab.competition import TournamentManager


PGN = b'''[Event "Tiny"]
[White "Ada"]
[Black "Turing"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
'''


class Upload:
    filename = "tiny.pgn"

    def __init__(self):
        self.stream = BytesIO(PGN)

    def read(self):
        return self.stream.read()


class EncodingTests(unittest.TestCase):
    def test_board_and_move_shapes(self):
        board = chess.Board()
        vector = encode_board(board)
        self.assertEqual(vector.shape, (INPUT_SIZE,))
        self.assertEqual(move_to_id(chess.Move.from_uci("e2e4")), chess.E2 * 64 + chess.E4)


class DatasetTests(unittest.TestCase):
    def test_import_and_position_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory))
            item = store.register_upload(Upload())
            self.assertEqual(item["games"], 1)
            self.assertEqual(item["positions"], 6)
            packed, _ = store.load_positions([item["id"]])
            self.assertEqual(sum(len(v[1]) for v in packed.values()), 6)

    def test_guided_examples_are_weighted_training_positions(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory))
            example = store.add_guided(chess.Board().fen(), "e2e4", "controle do centro", 4)
            self.assertEqual(example["san"], "e4")
            packed, selected = store.load_positions([], include_guided=True)
            self.assertEqual(len(packed["train"][1]), 4)
            self.assertEqual(selected[0]["id"], "guided")


class ModelTests(unittest.TestCase):
    def test_train_save_load_and_legal_move(self):
        model = PolicyNetwork([16], seed=7, name="Teste")
        board = chess.Board()
        x = np.stack([encode_board(board)] * 2)
        target = np.array([move_to_id(chess.Move.from_uci("e2e4"))] * 2, dtype=np.int32)
        loss = model.train_batch(x, target, 0.001)
        self.assertTrue(np.isfinite(loss))
        move = model.choose_move(board)
        self.assertIn(move, board.legal_moves)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path, {"id": "test"})
            loaded, metadata = PolicyNetwork.load(path)
            self.assertEqual(metadata["id"], "test")
            self.assertEqual(loaded.hidden_layers, [16])
            self.assertEqual(loaded.forward(x[:1]).shape, (1, OUTPUT_SIZE))


class ReplayTests(unittest.TestCase):
    def test_game_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReplayStore(Path(directory))
            game = chess.pgn.Game()
            game.headers.update({"White": "A", "Black": "B", "Result": "*"})
            node, board = game, game.board()
            for uci in ("e2e4", "e7e5"):
                move = chess.Move.from_uci(uci)
                node = node.add_variation(move); board.push(move)
            item = store.save_game(game, "selfplay")
            detail = store.detail(item["id"])
            self.assertEqual(detail["moves"], ["e2e4", "e7e5"])
            self.assertEqual(len(detail["fens"]), 3)

    def test_tournament_scoring(self):
        table = {"a": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0, "elo": 1200.0},
                 "b": {"played": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0.0, "elo": 1200.0}}
        TournamentManager._score(table, "a", "b", "1-0")
        self.assertEqual(table["a"]["points"], 1.0)
        self.assertGreater(table["a"]["elo"], table["b"]["elo"])


if __name__ == "__main__":
    unittest.main()
