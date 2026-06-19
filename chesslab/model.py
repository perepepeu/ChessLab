from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import chess
import numpy as np

from .encoding import INPUT_SIZE, OUTPUT_SIZE, encode_board, legal_move_groups


class PolicyNetwork:
    """A compact NumPy MLP whose every weight can be inspected and persisted."""

    def __init__(self, hidden_layers: Iterable[int] = (96, 64), seed: int = 42, name: str = "Aurora"):
        self.hidden_layers = [int(max(8, min(256, n))) for n in hidden_layers]
        self.seed = int(seed)
        self.name = name
        self.rng = np.random.default_rng(self.seed)
        sizes = [INPUT_SIZE, *self.hidden_layers, OUTPUT_SIZE]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(self.rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros(fan_out, dtype=np.float32))
        self.step = 0
        self.trained_positions = 0
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.weights, self.biases)))

    def forward(self, x: np.ndarray, return_activations: bool = False):
        if x.ndim == 1:
            x = x[None, :]
        activations = [x.astype(np.float32, copy=False)]
        current = activations[0]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            current = current @ weight + bias
            if index < len(self.weights) - 1:
                current = np.maximum(current, 0.0)
            activations.append(current)
        return (current, activations) if return_activations else current

    def train_batch(self, x: np.ndarray, targets: np.ndarray, learning_rate: float, advantages=None) -> float:
        logits, activations = self.forward(x, return_activations=True)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(np.clip(shifted, -40, 40))
        probs = exp / (exp.sum(axis=1, keepdims=True) + 1e-8)
        rows = np.arange(len(targets))
        chosen = np.clip(probs[rows, targets], 1e-8, 1.0)
        signal = np.ones(len(targets), dtype=np.float32) if advantages is None else np.asarray(advantages, dtype=np.float32)
        loss = float(np.mean(-np.log(chosen) * signal))
        grad = probs
        grad[rows, targets] -= 1.0
        grad *= signal[:, None] / max(1, len(targets))

        grad_w: list[np.ndarray] = [np.empty(0)] * len(self.weights)
        grad_b: list[np.ndarray] = [np.empty(0)] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            grad_w[layer] = activations[layer].T @ grad
            grad_b[layer] = grad.sum(axis=0)
            if layer:
                grad = grad @ self.weights[layer].T
                grad[activations[layer] <= 0] = 0
        self._adam_update(grad_w, grad_b, float(learning_rate))
        self.trained_positions += len(targets)
        return loss

    def _adam_update(self, grad_w, grad_b, learning_rate: float) -> None:
        self.step += 1
        beta1, beta2 = 0.9, 0.999
        for i in range(len(self.weights)):
            np.clip(grad_w[i], -2.0, 2.0, out=grad_w[i])
            np.clip(grad_b[i], -2.0, 2.0, out=grad_b[i])
            self._m_w[i] = beta1 * self._m_w[i] + (1 - beta1) * grad_w[i]
            self._v_w[i] = beta2 * self._v_w[i] + (1 - beta2) * (grad_w[i] ** 2)
            self._m_b[i] = beta1 * self._m_b[i] + (1 - beta1) * grad_b[i]
            self._v_b[i] = beta2 * self._v_b[i] + (1 - beta2) * (grad_b[i] ** 2)
            mw = self._m_w[i] / (1 - beta1 ** self.step)
            vw = self._v_w[i] / (1 - beta2 ** self.step)
            mb = self._m_b[i] / (1 - beta1 ** self.step)
            vb = self._v_b[i] / (1 - beta2 ** self.step)
            self.weights[i] -= learning_rate * mw / (np.sqrt(vw) + 1e-8)
            self.biases[i] -= learning_rate * mb / (np.sqrt(vb) + 1e-8)

    def policy(self, board: chess.Board, temperature: float = 0.25) -> tuple[list[chess.Move], np.ndarray]:
        groups = legal_move_groups(board)
        if not groups:
            return [], np.array([], dtype=np.float32)
        logits = self.forward(encode_board(board))[0]
        ids = list(groups)
        legal_logits = np.array([logits[i] for i in ids], dtype=np.float64)
        legal_logits = (legal_logits - legal_logits.max()) / max(0.05, float(temperature))
        probs = np.exp(np.clip(legal_logits, -40, 40))
        probs /= probs.sum()
        moves = [next((m for m in groups[i] if m.promotion == chess.QUEEN), groups[i][0]) for i in ids]
        return moves, probs.astype(np.float32)

    def choose_move(self, board: chess.Board, temperature: float = 0.15, sample: bool = False) -> chess.Move | None:
        moves, probabilities = self.policy(board, temperature)
        if not moves:
            return None
        index = int(self.rng.choice(len(moves), p=probabilities)) if sample else int(np.argmax(probabilities))
        return moves[index]

    def save(self, path: Path, metadata: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"w{i}": w for i, w in enumerate(self.weights)}
        payload.update({f"b{i}": b for i, b in enumerate(self.biases)})
        payload["metadata"] = np.array(json.dumps({
            "name": self.name, "seed": self.seed, "hidden_layers": self.hidden_layers,
            "step": self.step, "trained_positions": self.trained_positions, **(metadata or {}),
        }, ensure_ascii=False))
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: Path) -> tuple["PolicyNetwork", dict]:
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"]))
        model = cls(metadata["hidden_layers"], metadata["seed"], metadata.get("name", path.stem))
        for i in range(len(model.weights)):
            model.weights[i] = archive[f"w{i}"].astype(np.float32)
            model.biases[i] = archive[f"b{i}"].astype(np.float32)
        model.step = int(metadata.get("step", 0))
        model.trained_positions = int(metadata.get("trained_positions", 0))
        return model, metadata

    def snapshot(self, board: chess.Board | None = None, max_nodes: int = 14) -> dict:
        board = board or chess.Board()
        _, activations = self.forward(encode_board(board), return_activations=True)
        labels = ["Entrada", *[f"Oculta {i + 1}" for i in range(len(self.hidden_layers))], "Política"]
        sampled: list[np.ndarray] = []
        layers = []
        for i, activation in enumerate(activations):
            count = activation.shape[1]
            if i == 0:
                active = np.flatnonzero(np.abs(activation[0]) > 0)
                indices = active[:max_nodes] if len(active) else np.arange(min(count, max_nodes))
            elif i == len(activations) - 1:
                indices = np.argsort(activation[0])[-max_nodes:]
            else:
                indices = np.argsort(np.abs(activation[0]))[-max_nodes:]
            sampled.append(indices)
            values = activation[0, indices]
            layers.append({"name": labels[i], "total": int(count), "nodes": [
                {"id": f"{i}:{int(idx)}", "index": int(idx), "activation": float(value)}
                for idx, value in zip(indices, values)
            ]})
        edges = []
        for layer, weight in enumerate(self.weights):
            for source in sampled[layer]:
                for target in sampled[layer + 1]:
                    value = float(weight[int(source), int(target)])
                    edges.append({"source": f"{layer}:{int(source)}", "target": f"{layer + 1}:{int(target)}", "weight": value})
        return {"layers": layers, "edges": edges, "parameters": self.parameter_count, "trained_positions": self.trained_positions}

