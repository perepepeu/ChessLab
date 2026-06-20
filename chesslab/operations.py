from __future__ import annotations

import threading


class OperationGate:
    """Coordinates mutually exclusive compute-heavy operations."""

    def __init__(self):
        self.lock = threading.RLock()
        self.owner: str | None = None

    def acquire(self, owner: str) -> None:
        with self.lock:
            if self.owner is not None:
                raise ValueError(f"A operação '{self.owner}' já está em andamento.")
            self.owner = owner

    def release(self, owner: str) -> None:
        with self.lock:
            if self.owner == owner:
                self.owner = None

    def current(self) -> str | None:
        with self.lock:
            return self.owner
