from __future__ import annotations

import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import chess.pgn
import numpy as np

from .encoding import encode_board, move_to_id


class DatasetStore:
    def __init__(self, root: Path):
        self.root = root
        self.upload_dir = root / "uploads"
        self.players_dir = root / "players"
        self.registry_path = root / "datasets" / "registry.json"
        self.guided_path = root / "datasets" / "guided-examples.json"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.players_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]", encoding="utf-8")
        if not self.guided_path.exists():
            self.guided_path.write_text("[]", encoding="utf-8")
        self.bootstrap()
        self.organize_existing()

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
            if path.resolve() != Path(existing["path"]).resolve() and path.is_relative_to(self.upload_dir):
                path.unlink(missing_ok=True)
            return existing
        games, positions, players, results, player_counts = self.inspect(path)
        primary_player, appearances, confidence = self.identify_primary_player(player_counts, games)
        organized_path, folder_name = self.organize_file(path, primary_player, games, digest)
        item = {
            "id": digest[:12], "name": path.stem, "filename": organized_path.name,
            "path": str(organized_path.resolve()), "sha256": digest, "games": games,
            "positions": positions, "players": players[:8], "results": results,
            "primary_player": primary_player, "primary_player_games": appearances,
            "player_confidence": round(confidence, 4), "player_folder": folder_name,
            "size": len(raw), "imported_at": datetime.now(timezone.utc).isoformat(),
        }
        registry.insert(0, item)
        self._write_registry(registry)
        return item

    @staticmethod
    def inspect(path: Path) -> tuple[int, int, list[str], dict[str, int], Counter]:
        games = positions = 0
        players: set[str] = set()
        player_counts: Counter = Counter()
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games += 1
                positions += sum(1 for _ in game.mainline_moves())
                game_players = [name.strip() for name in [game.headers.get("White", ""), game.headers.get("Black", "")] if name.strip() and name.strip() != "?"]
                players.update(game_players)
                player_counts.update(game_players)
                result = game.headers.get("Result", "*")
                results[result] = results.get(result, 0) + 1
        if not games:
            raise ValueError("Nenhuma partida PGN válida foi encontrada.")
        return games, positions, sorted(players), results, player_counts

    @staticmethod
    def identify_primary_player(player_counts: Counter, games: int) -> tuple[str, int, float]:
        ranked = player_counts.most_common()
        if not ranked:
            return "Coleções mistas", 0, 0.0
        name, appearances = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        confidence = appearances / max(1, games)
        if appearances > second and confidence >= 0.5:
            return name, appearances, confidence
        return "Coleções mistas", appearances, confidence

    @staticmethod
    def player_slug(name: str) -> str:
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_name).strip("-").lower()
        return slug[:70] or "colecoes-mistas"

    def organized_filename(self, primary_player: str, games: int, digest: str) -> str:
        suffix = "partida" if games == 1 else "partidas"
        return f"{self.player_slug(primary_player)}-{games}-{suffix}-{digest[:8]}.pgn"

    def organize_file(self, path: Path, primary_player: str, games: int, digest: str) -> tuple[Path, str]:
        folder_name = self.player_slug(primary_player)
        folder = self.players_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        destination = folder / self.organized_filename(primary_player, games, digest)
        if path.resolve() != destination.resolve():
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(path.read_bytes()).digest():
                    path.unlink(missing_ok=True)
                else:
                    destination = folder / f"{path.stem}-{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}{path.suffix}"
                    shutil.move(str(path), str(destination))
            else:
                shutil.move(str(path), str(destination))
        return destination, f"Jogadores/{folder_name}"

    def organize_existing(self) -> None:
        registry = self._registry()
        changed = False
        for item in registry:
            path = Path(item.get("path", ""))
            if not path.exists():
                continue
            if not item.get("primary_player"):
                games, _, players, _, counts = self.inspect(path)
                primary, appearances, confidence = self.identify_primary_player(counts, games)
                item.update(primary_player=primary, primary_player_games=appearances,
                            player_confidence=round(confidence, 4), players=players[:8])
                changed = True
            managed_path = path.is_relative_to(self.upload_dir.resolve()) or path.is_relative_to(self.players_dir.resolve())
            expected_name = self.organized_filename(item["primary_player"], int(item["games"]), item["sha256"])
            expected_folder = (self.players_dir / self.player_slug(item["primary_player"])).resolve()
            if managed_path and (path.parent.resolve() != expected_folder or path.name != expected_name):
                organized_path, folder_name = self.organize_file(path, item["primary_player"], int(item["games"]), item["sha256"])
                item.update(path=str(organized_path.resolve()), filename=organized_path.name, player_folder=folder_name)
                changed = True
            elif not item.get("player_folder"):
                item["player_folder"] = f"Jogadores/{self.player_slug(item['primary_player'])}"
                changed = True
        if changed:
            self._write_registry(registry)

    def list(self) -> list[dict]:
        return self._registry()

    def rename(self, dataset_id: str, name: str) -> dict:
        clean_name = " ".join(str(name).strip().split())[:80]
        if not clean_name:
            raise ValueError("Informe um nome para o dataset.")
        registry = self._registry()
        item = next((entry for entry in registry if entry["id"] == dataset_id), None)
        if not item:
            raise ValueError("Dataset não encontrado.")
        item["name"] = clean_name
        self._write_registry(registry)
        return item

    def delete(self, dataset_id: str) -> dict:
        registry = self._registry()
        item = next((entry for entry in registry if entry["id"] == dataset_id), None)
        if not item:
            raise ValueError("Dataset não encontrado.")
        path = Path(item["path"]).resolve()
        managed = path.is_relative_to(self.upload_dir.resolve()) or path.is_relative_to(self.players_dir.resolve())
        if not managed:
            raise ValueError("O arquivo está fora da biblioteca gerenciada e não pode ser apagado.")
        registry = [entry for entry in registry if entry["id"] != dataset_id]
        self._write_registry(registry)
        path.unlink(missing_ok=True)
        if path.parent != self.players_dir and path.parent.is_relative_to(self.players_dir):
            try:
                path.parent.rmdir()
            except OSError:
                pass
        return item

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
