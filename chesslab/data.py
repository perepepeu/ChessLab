from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import numpy as np

from .encoding import encode_board, move_to_id


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.upload_dir = root / "uploads"
        self.registry_path = root / "datasets" / "registry.json"
        self.guided_path = root / "datasets" / "guided-examples.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")
        if not self.guided_path.exists():
            self.guided_path.write_text("[]", encoding="utf-8")
        self.bootstrap()

    def _registry(self) -> list[dict]:
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

    def _write_registry(self, data: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def bootstrap(self) -> None:
        known = {item["path"] for item in self._registry()}
        for path in self.upload_dir.glob("*.pgn"):
            if str(path.resolve()) not in known:
                self.register_path(path, copy=False)

    def register_upload(self, file_storage) -> dict:
        raw = file_storage.read()
        if not raw:
            raise ValueError("O arquivo PGN está vazio.")
        digest = hashlib.sha256(raw).hexdigest()
        safe_name = Path(file_storage.filename or "partidas.pgn").stem[:60]
        destination = self.upload_dir / f"{safe_name}-{digest[:8]}.pgn"
        destination.write_bytes(raw)
        return self.register_path(destination, copy=False)

    def register_path(self, path: Path, copy: bool = False) -> dict:
        if copy:
            destination = self.upload_dir / path.name
            shutil.copy2(path, destination)
            path = destination
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        registry = self._registry()
        existing = next((item for item in registry if item["sha256"] == digest), None)
        if existing:
            return existing
        games, positions, players, results = self.inspect(path)
        item = {
            "id": digest[:12], "name": path.stem, "filename": path.name,
            "path": str(path.resolve()), "sha256": digest, "games": games,
            "positions": positions, "players": players[:8], "results": results,
            "size": len(raw), "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        registry.insert(0, item)
        self._write_registry(registry)
        return item

    @staticmethod
    def inspect(path: Path) -> tuple[int, int, list[str], dict[str, int]]:
        games = positions = 0
        players: set[str] = set()
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games += 1
                positions += sum(1 for _ in game.mainline_moves())
                players.update(filter(None, [game.headers.get("White"), game.headers.get("Black")]))
                result = game.headers.get("Result", "*")
                results[result] = results.get(result, 0) + 1
        if not games:
            raise ValueError("Nenhuma partida PGN válida foi encontrada.")
        return games, positions, sorted(players), results

    def list(self) -> list[dict]:
        return self._registry()

    def list_guided(self) -> list[dict]:
        try:
            return json.loads(self.guided_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def add_guided(self, fen: str, move_uci: str, note: str = "", priority: int = 3) -> dict:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
        except ValueError as exc:
            raise ValueError("Posição ou lance guiado inválido.") from exc
        if move not in board.legal_moves:
            raise ValueError("O lance ensinado precisa ser legal nesta posição.")
        digest = hashlib.sha256(f"{board.fen()}|{move.uci()}".encode()).hexdigest()[:12]
        examples = self.list_guided()
        existing = next((item for item in examples if item["id"] == digest), None)
        if existing:
            existing.update(note=note.strip()[:200], priority=max(1, min(10, int(priority))))
            item = existing
        else:
            item = {"id": digest, "fen": board.fen(), "move": move.uci(), "san": board.san(move),
                    "note": note.strip()[:200], "priority": max(1, min(10, int(priority))),
                    "created_at": datetime.now(timezone.utc).isoformat()}
            examples.insert(0, item)
        self.guided_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        return item

    def delete_guided(self, example_id: str) -> None:
        examples = [item for item in self.list_guided() if item["id"] != example_id]
        self.guided_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    def guided_position(self, fen: str | None = None) -> dict:
        try:
            board = chess.Board(fen) if fen else chess.Board()
        except ValueError as exc:
            raise ValueError("FEN inválida.") from exc
        return {"fen": board.fen(), "turn": "white" if board.turn else "black",
                "legal_moves": [move.uci() for move in board.legal_moves], "check": board.is_check()}

    def load_positions(self, ids: list[str] | None = None, max_positions: int = 20000,
                       include_guided: bool = False):
        selected = [d for d in self._registry() if ids is None or d["id"] in ids]
        splits = {"train": [[], []], "val": [[], []], "test": [[], []]}
        game_index = 0
        for dataset in selected:
            with Path(dataset["path"]).open("r", encoding="utf-8-sig", errors="replace") as handle:
                while sum(len(v[1]) for v in splits.values()) < max_positions:
                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break
                    bucket_value = int(hashlib.sha256(f"{dataset['sha256']}:{game_index}".encode()).hexdigest()[:8], 16) % 100
                    split = "train" if bucket_value < 70 else "val" if bucket_value < 85 else "test"
                    board = game.board()
                    for move in game.mainline_moves():
                        splits[split][0].append(encode_board(board))
                        splits[split][1].append(move_to_id(move))
                        board.push(move)
                    game_index += 1
        if include_guided:
            examples = self.list_guided()
            for example in examples:
                board = chess.Board(example["fen"])
                repetitions = max(1, min(10, int(example.get("priority", 3))))
                for _ in range(repetitions):
                    splits["train"][0].append(encode_board(board))
                    splits["train"][1].append(move_to_id(chess.Move.from_uci(example["move"])))
            guided_hash = hashlib.sha256(self.guided_path.read_bytes()).hexdigest()
            if examples:
                selected.append({"id": "guided", "name": "Mentor ChessLab", "sha256": guided_hash,
                                 "games": 0, "positions": len(examples)})
        packed = {}
        for name, (features, labels) in splits.items():
            packed[name] = (
                np.asarray(features, dtype=np.float32),
                np.asarray(labels, dtype=np.int32),
            )
        return packed, selected
