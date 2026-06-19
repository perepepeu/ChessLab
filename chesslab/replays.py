from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn


class ReplayStore:
    """Persistent PGN episode archive shared by training and championships."""

    def __init__(self, root: Path):
        self.root = root / "replays"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.lock = threading.RLock()
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, items: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_game(self, game: chess.pgn.Game, source: str, metadata: dict | None = None) -> dict:
        metadata = metadata or {}
        replay_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        path = self.root / f"{replay_id}.pgn"
        text = str(game)
        path.write_text(text + "\n", encoding="utf-8")
        item = {
            "id": replay_id,
            "source": source,
            "title": metadata.get("title") or game.headers.get("Event", "Partida gravada"),
            "white": game.headers.get("White", "Brancas"),
            "black": game.headers.get("Black", "Pretas"),
            "result": game.headers.get("Result", "*"),
            "plies": sum(1 for _ in game.mainline_moves()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": metadata.get("run_id"),
            "tournament_id": metadata.get("tournament_id"),
            "path": str(path.resolve()),
            "tags": metadata.get("tags", []),
        }
        with self.lock:
            items = self._read()
            items.insert(0, item)
            self._write(items)
        return item

    def list(self, limit: int = 200) -> list[dict]:
        with self.lock:
            return self._read()[:limit]

    def detail(self, replay_id: str) -> dict:
        with self.lock:
            item = next((entry for entry in self._read() if entry["id"] == replay_id), None)
        if not item:
            raise ValueError("Replay não encontrado.")
        with Path(item["path"]).open("r", encoding="utf-8") as handle:
            game = chess.pgn.read_game(handle)
        if game is None:
            raise ValueError("O PGN deste replay está corrompido.")
        board = game.board()
        fens = [board.fen()]
        moves, sans = [], []
        for move in game.mainline_moves():
            sans.append(board.san(move))
            moves.append(move.uci())
            board.push(move)
            fens.append(board.fen())
        return {**item, "headers": dict(game.headers), "moves": moves, "sans": sans, "fens": fens, "pgn": str(game)}

