from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

import chess


class GameSessionStore:
    """Persistent, thread-safe game sessions with TTL and LRU eviction."""

    def __init__(self, database_path: str | Path = ":memory:", max_sessions: int = 100,
                 ttl_seconds: float = 12 * 60 * 60, clock: Callable[[], float] = time.time):
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.clock = clock
        self.lock = threading.RLock()
        if database_path != ":memory:":
            path = Path(database_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            database_path = str(path)
        self.connection = sqlite3.connect(database_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS game_sessions (
                session_id TEXT PRIMARY KEY,
                fen TEXT NOT NULL,
                strength TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_seq INTEGER NOT NULL
            )
        """)
        self.connection.commit()
        with self.lock:
            self._evict_expired(self.clock())
            self._evict_lru()

    def _next_sequence(self) -> int:
        row = self.connection.execute("SELECT COALESCE(MAX(access_seq), 0) + 1 AS value FROM game_sessions").fetchone()
        return int(row["value"])

    def put(self, session_id: str, board: chess.Board, strength: str) -> None:
        with self.lock:
            now = self.clock()
            self._evict_expired(now)
            sequence = self._next_sequence()
            self.connection.execute("""
                INSERT INTO game_sessions(session_id, fen, strength, created_at, updated_at, access_seq)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    fen=excluded.fen,
                    strength=excluded.strength,
                    updated_at=excluded.updated_at,
                    access_seq=excluded.access_seq
            """, (session_id, board.fen(), strength, now, now, sequence))
            self._evict_lru()
            self.connection.commit()

    def get(self, session_id: str) -> dict | None:
        with self.lock:
            now = self.clock()
            self._evict_expired(now)
            row = self.connection.execute(
                "SELECT fen, strength FROM game_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE game_sessions SET updated_at = ?, access_seq = ? WHERE session_id = ?",
                (now, self._next_sequence(), session_id),
            )
            self.connection.commit()
            return {"board": chess.Board(row["fen"]), "strength": row["strength"], "updated_at": now}

    def _evict_expired(self, now: float) -> None:
        self.connection.execute("DELETE FROM game_sessions WHERE updated_at < ?", (now - self.ttl_seconds,))

    def _evict_lru(self) -> None:
        self.connection.execute("""
            DELETE FROM game_sessions
            WHERE session_id IN (
                SELECT session_id FROM game_sessions
                ORDER BY access_seq DESC
                LIMIT -1 OFFSET ?
            )
        """, (self.max_sessions,))

    def __len__(self) -> int:
        with self.lock:
            self._evict_expired(self.clock())
            count = self.connection.execute("SELECT COUNT(*) AS value FROM game_sessions").fetchone()["value"]
            self.connection.commit()
            return int(count)

    def close(self) -> None:
        with self.lock:
            self.connection.close()
