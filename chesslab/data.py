from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import chess.pgn
import numpy as np

from .encoding import encode_board, move_to_id


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.upload_dir = root / "uploads"
        self.registry_path = root / "datasets" / "registry.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")
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

    def load_positions(self, ids: list[str] | None = None, max_positions: int = 20000):
        selected = [d for d in self._registry() if not ids or d["id"] in ids]
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
        packed = {}
        for name, (features, labels) in splits.items():
            packed[name] = (
                np.asarray(features, dtype=np.float32),
                np.asarray(labels, dtype=np.int32),
            )
        return packed, selected

