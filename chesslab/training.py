from __future__ import annotations

import hashlib
import json
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .data import DatasetStore
from .encoding import encode_board, material_score, move_to_id
from .model import PolicyNetwork
from .replays import ReplayStore


class TrainingManager:
    def __init__(self, root: Path, datasets: DatasetStore, replays: ReplayStore | None = None):
        self.root = root
        self.datasets = datasets
        self.models_dir = root / "models"
        self.runs_dir = root / "runs"
        self.models_dir.mkdir(exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.replays = replays or ReplayStore(root)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.model = PolicyNetwork()
        self.active_model_id: str | None = None
        self.state = self._idle_state()
        self._restore_latest()

    def _restore_latest(self) -> None:
        """Resume the newest valid checkpoint so a restart does not forget the active brain."""
        candidates = sorted(self.models_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for metadata_path in candidates:
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                model_id = metadata["id"]
                self.model, _ = PolicyNetwork.load(self.models_dir / f"{model_id}.npz")
                self.active_model_id = model_id
                self.state["logs"] = [{"time": "agora", "message": f"Checkpoint {self.model.name} restaurado."}]
                return
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    @staticmethod
    def _idle_state() -> dict:
        return {"running": False, "stage": "Pronto", "progress": 0, "loss": None, "epoch": 0,
                "metrics": [], "logs": [{"time": "agora", "message": "Laboratório pronto para treinar."}], "error": None}

    def start(self, config: dict) -> dict:
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Já existe um treinamento em andamento.")
            self.stop_event.clear()
            run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            config = self._normalize_config(config)
            self.state = {**self._idle_state(), "running": True, "stage": "Preparando dados", "run_id": run_id, "config": config}
            self.thread = threading.Thread(target=self._run, args=(run_id, config), daemon=True)
            self.thread.start()
            return self.status()

    @staticmethod
    def _normalize_config(config: dict) -> dict:
        layers = config.get("hidden_layers", [96, 64])
        if isinstance(layers, str):
            layers = [int(v.strip()) for v in layers.split(",") if v.strip()]
        return {
            "name": str(config.get("name", "Aurora")).strip()[:40] or "Aurora",
            "mode": config.get("mode", "hybrid") if config.get("mode") in {"imitation", "selfplay", "hybrid", "guided"} else "hybrid",
            "hidden_layers": [max(8, min(256, int(v))) for v in layers[:3]] or [64],
            "epochs": max(1, min(50, int(config.get("epochs", 4)))),
            "batch_size": max(4, min(256, int(config.get("batch_size", 32)))),
            "learning_rate": max(0.00001, min(0.05, float(config.get("learning_rate", 0.001)))),
            "selfplay_episodes": max(1, min(100, int(config.get("selfplay_episodes", 8)))),
            "temperature": max(0.05, min(2.0, float(config.get("temperature", 0.7)))),
            "max_positions": max(64, min(100000, int(config.get("max_positions", 10000)))),
            "seed": int(config.get("seed", 42)),
            "dataset_ids": list(config["dataset_ids"]) if "dataset_ids" in config else None,
            "include_guided": bool(config.get("include_guided", True)),
            "base_model_id": Path(str(config.get("base_model_id", ""))).name or None,
        }

    def stop(self) -> None:
        self.stop_event.set()
        self._log("Parada solicitada; concluindo o lote atual.")

    def status(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.state))

    def _log(self, message: str) -> None:
        with self.lock:
            self.state.setdefault("logs", []).insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "message": message})
            self.state["logs"] = self.state["logs"][:30]

    def _set(self, **values) -> None:
        with self.lock:
            self.state.update(values)

    def _run(self, run_id: str, config: dict) -> None:
        started = time.time()
        try:
            if config["base_model_id"]:
                base_path = self.models_dir / f"{config['base_model_id']}.npz"
                if not base_path.exists():
                    raise ValueError("O modelo base selecionado não existe mais.")
                self.model, _ = PolicyNetwork.load(base_path)
                self.model.name = config["name"]
                config["hidden_layers"] = self.model.hidden_layers
                self._log(f"Fine-tuning iniciado a partir de {config['base_model_id']}.")
            else:
                self.model = PolicyNetwork(config["hidden_layers"], config["seed"], config["name"])
            packed = None
            selected = []
            if config["mode"] in {"imitation", "hybrid", "guided"}:
                dataset_ids = [] if config["mode"] == "guided" else config["dataset_ids"]
                use_guided = config["mode"] == "guided" or config["include_guided"]
                packed, selected = self.datasets.load_positions(dataset_ids, config["max_positions"], use_guided)
                if len(packed["train"][1]) == 0:
                    raise ValueError("Importe e selecione ao menos um PGN antes do treino por imitação.")
                self._log(f"{len(packed['train'][1]):,} posições de treino carregadas de {len(selected)} conjunto(s).")
                self._train_imitation(packed, config)
            if not self.stop_event.is_set() and config["mode"] in {"selfplay", "hybrid"}:
                self._train_selfplay(config, run_id)
            evaluation = self._evaluate(packed, config)
            model_id = f"{config['name'].lower().replace(' ', '-')}-{run_id}"
            model_path = self.models_dir / f"{model_id}.npz"
            metadata = {"id": model_id, "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                        "parent_model_id": config["base_model_id"],
                        "evaluation": evaluation, "duration_seconds": round(time.time() - started, 2), "run_id": run_id}
            self.model.save(model_path, metadata)
            (self.models_dir / f"{model_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            self._write_run_manifest(run_id, config, selected, metadata)
            self.active_model_id = model_id
            self._set(running=False, stage="Concluído", progress=100, evaluation=evaluation, model_id=model_id)
            self._log(f"Checkpoint {config['name']} salvo com sucesso.")
        except Exception as exc:
            self._set(running=False, stage="Interrompido" if self.stop_event.is_set() else "Falha", error=str(exc))
            self._log(str(exc))

    def _train_imitation(self, packed, config) -> None:
        x_train, y_train = packed["train"]
        batch = config["batch_size"]
        epochs = config["epochs"]
        rng = np.random.default_rng(config["seed"])
        total_batches = max(1, epochs * int(np.ceil(len(y_train) / batch)))
        completed = 0
        self._set(stage="Imitação supervisionada")
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(y_train))
            losses = []
            for start in range(0, len(order), batch):
                if self.stop_event.is_set():
                    return
                indices = order[start:start + batch]
                loss = self.model.train_batch(x_train[indices], y_train[indices], config["learning_rate"])
                losses.append(loss)
                completed += 1
                progress = int(70 * completed / total_batches) if config["mode"] == "hybrid" else int(90 * completed / total_batches)
                self._set(progress=progress, loss=round(float(np.mean(losses)), 4), epoch=epoch)
            val_accuracy = self._policy_accuracy(*packed["val"], limit=500)
            metric = {"step": epoch, "loss": round(float(np.mean(losses)), 4), "validation": round(val_accuracy, 4), "stage": "imitação"}
            with self.lock:
                self.state["metrics"].append(metric)
            self._log(f"Época {epoch}/{epochs}: loss {metric['loss']} · validação {metric['validation'] * 100:.1f}%")

    def _train_selfplay(self, config, run_id: str) -> None:
        episodes = config["selfplay_episodes"]
        self._set(stage="Autojogo por reforço")
        for episode in range(1, episodes + 1):
            if self.stop_event.is_set():
                return
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers.update({"Event": "ChessLab Self-play", "Date": datetime.now().strftime("%Y.%m.%d"),
                                 "Round": str(episode), "White": self.model.name, "Black": self.model.name, "Result": "*"})
            node = game
            trajectory: list[tuple[np.ndarray, int, chess.Color, float]] = []
            entropies: list[float] = []
            for _ in range(120):
                if board.is_game_over(claim_draw=True):
                    break
                color = board.turn
                moves, probabilities = self.model.policy(board, config["temperature"])
                if not moves:
                    break
                entropies.append(float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)))))
                move = moves[int(self.model.rng.choice(len(moves), p=probabilities))]
                before = material_score(board, color)
                state = encode_board(board)
                node = node.add_variation(move)
                board.push(move)
                shaping = np.tanh((material_score(board, color) - before) / 5.0) * 0.08
                trajectory.append((state, move_to_id(move), color, float(shaping)))
            outcome = board.outcome(claim_draw=True)
            winner = outcome.winner if outcome else None
            game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"
            game.headers["Termination"] = outcome.termination.name if outcome else "Ply limit"
            if trajectory:
                features = np.stack([t[0] for t in trajectory])
                targets = np.asarray([t[1] for t in trajectory], dtype=np.int32)
                rewards = np.asarray([(1.0 if winner == t[2] else -1.0 if winner is not None else 0.0) + t[3] for t in trajectory], dtype=np.float32)
                advantages = rewards - rewards.mean() if len(rewards) > 1 else rewards
                loss = self.model.train_batch(features, targets, config["learning_rate"] * 0.35, advantages)
            else:
                loss = 0.0
            base = 70 if config["mode"] == "hybrid" else 0
            span = 20 if config["mode"] == "hybrid" else 90
            self._set(progress=base + int(span * episode / episodes), loss=round(float(loss), 4), epoch=episode)
            replay = self.replays.save_game(game, "selfplay", {"run_id": run_id,
                "title": f"Treino {self.model.name} · episódio {episode}", "tags": [config["mode"]]})
            metric = {"step": episode, "loss": round(float(loss), 4), "reward": float(1 if winner else 0),
                      "plies": len(trajectory), "entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
                      "replay_id": replay["id"], "stage": "autojogo"}
            with self.lock:
                self.state["metrics"].append(metric)
            self._log(f"Autojogo {episode}/{episodes}: {len(trajectory)} lances · replay salvo")

    def _policy_accuracy(self, x: np.ndarray, y: np.ndarray, limit: int = 1000) -> float:
        if not len(y):
            return 0.0
        indices = np.arange(min(len(y), limit))
        predictions = np.argmax(self.model.forward(x[indices]), axis=1)
        return float(np.mean(predictions == y[indices]))

    def _evaluate(self, packed, config) -> dict:
        self._set(stage="Avaliação independente", progress=94)
        accuracy = self._policy_accuracy(*packed["test"], limit=1000) if packed else 0.0
        return {"test_policy_accuracy": round(accuracy, 4), "parameters": self.model.parameter_count,
                "trained_positions": self.model.trained_positions, "seed": config["seed"]}

    def _write_run_manifest(self, run_id, config, selected, metadata) -> None:
        digest = hashlib.sha256("".join(d["sha256"] for d in selected).encode()).hexdigest() if selected else "selfplay-generated"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        manifest = {
            "experiment_id": run_id, "hypothesis": "Imitation followed by bounded self-play improves held-out policy agreement.",
            "decision": "Whether to retain this checkpoint as a playable challenger.", "code_version": "local-workspace",
            "environment_version": "python-chess-standard-v1", "evaluator_version": "heldout-policy-v1",
            "data": {"source": [d["name"] for d in selected] or ["self-play"], "period": "user supplied",
                     "hash": digest, "split_policy": "game-aware deterministic 70/15/15"},
            "runtime": {"hardware": platform.processor() or "CPU", "software": platform.python_version(), "numeric_precision": "fp32"},
            "seeds": [config["seed"], config["seed"] + 1, config["seed"] + 2],
            "baseline": {"name": "random legal policy", "version": "v1"},
            "candidate": {"name": config["name"], "configuration": config,
                          "initial_checkpoint": config["base_model_id"] or "none"},
            "budget": {"training_steps": self.model.step, "environment_interactions": config["selfplay_episodes"], "wall_clock_limit_minutes": 0},
            "evaluation": {"primary_metric": "held-out policy agreement", "held_out_suite": "pgn-game-split-v1",
                           "minimum_effect": 0.0, "maximum_regression": 0.05, "confidence_level": 0.95},
            "artifacts": {"checkpoint_uri": str((self.models_dir / f"{metadata['id']}.npz").resolve()),
                          "raw_results_uri": str((run_dir / "result.json").resolve()), "telemetry_uri": str(run_dir.resolve())},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "result.json").write_text(json.dumps({"metrics": self.state.get("metrics", []), **metadata}, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_models(self) -> list[dict]:
        models = []
        for path in sorted(self.models_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["active"] = item.get("id") == self.active_model_id
                item["size"] = (self.models_dir / f"{item['id']}.npz").stat().st_size
                models.append(item)
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        return models

    def import_checkpoint(self, file_storage) -> dict:
        raw = file_storage.read()
        if not raw:
            raise ValueError("O arquivo do modelo está vazio.")
        digest = hashlib.sha256(raw).hexdigest()[:10]
        safe_stem = Path(file_storage.filename or "modelo-base.npz").stem.lower().replace(" ", "-")[:40]
        model_id = f"{safe_stem}-{digest}"
        path = self.models_dir / f"{model_id}.npz"
        path.write_bytes(raw)
        try:
            model, embedded = PolicyNetwork.load(path)
        except Exception as exc:
            path.unlink(missing_ok=True)
            raise ValueError("Este arquivo não é um checkpoint ChessLab válido.") from exc
        metadata = {
            "id": model_id, "created_at": datetime.now(timezone.utc).isoformat(), "imported": True,
            "parent_model_id": embedded.get("parent_model_id"),
            "config": embedded.get("config", {"name": model.name, "mode": "imported", "hidden_layers": model.hidden_layers}),
            "evaluation": embedded.get("evaluation", {"parameters": model.parameter_count,
                "trained_positions": model.trained_positions, "test_policy_accuracy": 0.0}),
            "duration_seconds": embedded.get("duration_seconds", 0), "run_id": embedded.get("run_id"),
        }
        (self.models_dir / f"{model_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def load_model(self, model_id: str) -> dict:
        path = self.models_dir / f"{Path(model_id).name}.npz"
        if not path.exists():
            raise ValueError("Checkpoint não encontrado.")
        self.model, metadata = PolicyNetwork.load(path)
        self.active_model_id = model_id
        return metadata
