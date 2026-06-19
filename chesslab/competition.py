from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn

from .model import PolicyNetwork
from .replays import ReplayStore
from .search import choose_with_search


class TournamentManager:
    def __init__(self, root: Path, replays: ReplayStore):
        self.root = root
        self.models_dir = root / "models"
        self.tournaments_dir = root / "tournaments"
        self.tournaments_dir.mkdir(exist_ok=True)
        self.replays = replays
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state = self._idle()

    @staticmethod
    def _idle() -> dict:
        return {"running": False, "stage": "Pronto", "progress": 0, "games": [], "standings": [], "error": None}

    def status(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def start(self, config: dict) -> dict:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Já existe um campeonato em andamento.")
            model_ids = [Path(value).name for value in config.get("model_ids", [])]
            if not model_ids:
                raise ValueError("Escolha ao menos um checkpoint.")
            rounds = max(1, min(1000, int(config.get("rounds", 1))))
            max_plies = max(20, min(300, int(config.get("max_plies", 100))))
            tournament_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            normalized = {"model_ids": model_ids, "rounds": rounds, "max_plies": max_plies,
                          "temperature": max(.05, min(1.5, float(config.get("temperature", .25)))),
                          "search_level": config.get("search_level", "policy") if config.get("search_level") in {"policy", "tactical", "search2"} else "policy"}
            self.stop_event.clear()
            self.state = {**self._idle(), "running": True, "stage": "Montando chaves", "id": tournament_id,
                          "config": normalized, "started_at": datetime.now(timezone.utc).isoformat()}
            self.thread = threading.Thread(target=self._run, args=(tournament_id, normalized), daemon=True)
            self.thread.start()
            return self.status()

    def stop(self) -> None:
        self.stop_event.set()
        with self.lock:
            self.state["stage"] = "Encerrando a partida atual"

    def _run(self, tournament_id: str, config: dict) -> None:
        try:
            players = self._load_players(config["model_ids"])
            if len(players) == 1:
                base = players[0]
                clone, _ = PolicyNetwork.load(self.models_dir / f"{base['model_id']}.npz")
                players.append({"id": base["id"] + "::clone", "model_id": base["model_id"], "name": base["name"] + " Clone", "model": clone})
                players[0]["name"] += " Principal"
            standings = {p["id"]: {"id": p["id"], "name": p["name"], "played": 0, "wins": 0,
                                           "draws": 0, "losses": 0, "points": 0.0, "elo": 1200.0} for p in players}
            pairings = []
            for round_number in range(1, config["rounds"] + 1):
                for i in range(len(players)):
                    for j in range(i + 1, len(players)):
                        pairings.extend([(round_number, players[i], players[j]), (round_number, players[j], players[i])])
            total = len(pairings)
            for index, (round_number, white, black) in enumerate(pairings, start=1):
                if self.stop_event.is_set():
                    break
                with self.lock:
                    self.state.update(stage=f"Rodada {round_number} · partida {index}/{total}", progress=int((index - 1) * 100 / total))
                game = self._play_game(white, black, config)
                replay = self.replays.save_game(game, "championship", {"tournament_id": tournament_id,
                    "title": f"Campeonato · {white['name']} × {black['name']}", "tags": [config["search_level"]]})
                result = game.headers["Result"]
                self._score(standings, white["id"], black["id"], result)
                record = {"replay_id": replay["id"], "round": round_number, "white": white["name"],
                          "black": black["name"], "result": result, "plies": replay["plies"]}
                with self.lock:
                    self.state["games"].insert(0, record)
                    self.state["standings"] = sorted(standings.values(), key=lambda p: (p["points"], p["elo"]), reverse=True)
            with self.lock:
                self.state.update(running=False, stage="Concluído" if not self.stop_event.is_set() else "Interrompido", progress=100)
            (self.tournaments_dir / f"{tournament_id}.json").write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            with self.lock:
                self.state.update(running=False, stage="Falha", error=str(exc))

    def _load_players(self, model_ids: list[str]) -> list[dict]:
        players = []
        for model_id in model_ids:
            path = self.models_dir / f"{model_id}.npz"
            if not path.exists():
                raise ValueError(f"Checkpoint {model_id} não encontrado.")
            model, metadata = PolicyNetwork.load(path)
            players.append({"id": model_id, "model_id": model_id, "name": metadata.get("name", model_id), "model": model})
        return players

    def _play_game(self, white: dict, black: dict, config: dict) -> chess.pgn.Game:
        game = chess.pgn.Game()
        game.headers.update({"Event": "ChessLab Championship", "Date": datetime.now().strftime("%Y.%m.%d"),
                             "White": white["name"], "Black": black["name"], "Round": "-", "Result": "*"})
        board, node = game.board(), game
        for _ in range(config["max_plies"]):
            if board.is_game_over(claim_draw=True):
                break
            player = white if board.turn == chess.WHITE else black
            move = choose_with_search(player["model"], board, config["search_level"], config["temperature"], sample=True)
            if move is None:
                break
            node = node.add_variation(move)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"
        game.headers["Termination"] = outcome.termination.name if outcome else "Ply limit"
        return game

    @staticmethod
    def _score(standings: dict, white_id: str, black_id: str, result: str) -> None:
        white, black = standings[white_id], standings[black_id]
        white["played"] += 1; black["played"] += 1
        actual_white = 1.0 if result == "1-0" else 0.0 if result == "0-1" else 0.5
        actual_black = 1.0 - actual_white
        if actual_white == 1:
            white["wins"] += 1; black["losses"] += 1
        elif actual_white == 0:
            black["wins"] += 1; white["losses"] += 1
        else:
            white["draws"] += 1; black["draws"] += 1
        white["points"] += actual_white; black["points"] += actual_black
        expected_white = 1 / (1 + 10 ** ((black["elo"] - white["elo"]) / 400))
        white["elo"] = round(white["elo"] + 16 * (actual_white - expected_white), 1)
        black["elo"] = round(black["elo"] + 16 * (actual_black - (1 - expected_white)), 1)
