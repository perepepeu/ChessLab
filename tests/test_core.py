from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import chess
import numpy as np

from chesslab.data import DatasetStore
from chesslab.encoding import INPUT_SIZE, OUTPUT_SIZE, encode_board, move_to_id
from chesslab.model import PolicyNetwork


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


if __name__ == "__main__":
    unittest.main()

