from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable

import chess


class GameSessionStore:
    """Thread-safe in-memory game sessions with TTL and LRU eviction."""

    def __init__(self, max_sessions: int = 100, ttl_seconds: float = 12 * 60 * 60,
                 clock: Callable[[], float] = time.monotonic):
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.clock = clock
        self.lock = threading.RLock()
        self._sessions: OrderedDict[str, dict] = OrderedDict()

    def put(self, session_id: str, board: chess.Board, strength: str) -> None:
        with self.lock:
            now = self.clock()
            self._evict_expired(now)
            self._sessions.pop(session_id, None)
            self._sessions[session_id] = {"board": board, "strength": strength, "updated_at": now}
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)

    def get(self, session_id: str) -> dict | None:
        with self.lock:
            now = self.clock()
            self._evict_expired(now)
            session = self._sessions.pop(session_id, None)
            if session is None:
                return None
            session["updated_at"] = now
            self._sessions[session_id] = session
            return session

    def _evict_expired(self, now: float) -> None:
        expired = [session_id for session_id, session in self._sessions.items()
                   if now - session["updated_at"] > self.ttl_seconds]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        with self.lock:
            self._evict_expired(self.clock())
            return len(self._sessions)
