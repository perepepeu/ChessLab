from __future__ import annotations

from io import BytesIO

import pytest
from flask.testing import FlaskClient

import app as app_module
from chesslab.sessions import GameSessionStore


@pytest.fixture()
def client(tmp_path, monkeypatch) -> FlaskClient:
    session_store = GameSessionStore(tmp_path / "sessions.sqlite3", max_sessions=10, ttl_seconds=60)
    monkeypatch.setattr(app_module, "games", session_store)
    monkeypatch.setattr(
        app_module,
        "choose_with_search",
        lambda model, board, level="policy", temperature=0.15, sample=False: next(iter(board.legal_moves), None),
    )
    app_module.app.config.update(TESTING=True)
    with app_module.app.test_client() as test_client:
        yield test_client
    session_store.close()


def test_play_new_and_move_round_trip(client: FlaskClient):
    created = client.post("/api/play/new", json={"color": "white", "strength": "policy"})
    assert created.status_code == 200
    game = created.get_json()["game"]
    assert "e2e4" in game["legal_moves"]

    moved = client.post("/api/play/move", json={"session_id": game["session_id"], "move": "e2e4"})
    assert moved.status_code == 200
    updated = moved.get_json()["game"]
    assert updated["ai_move"] is not None
    assert updated["fen"] != game["fen"]


def test_play_move_rejects_expired_session(client: FlaskClient):
    response = client.post("/api/play/move", json={"session_id": "missing", "move": "e2e4"})
    assert response.status_code == 400
    assert response.get_json() == {"ok": False, "error": "Partida expirada. Inicie uma nova partida."}


def test_training_start_passes_zero_base_model(client: FlaskClient, monkeypatch):
    received = {}

    def fake_start(config):
        received.update(config)
        return {"running": True, "stage": "Preparando dados"}

    monkeypatch.setattr(app_module.trainer, "start", fake_start)
    response = client.post("/api/training/start", json={"name": "Zero", "base_model_id": None})
    assert response.status_code == 200
    assert response.get_json()["training"]["running"] is True
    assert received["base_model_id"] is None


def test_dataset_import_rejects_fake_pgn(client: FlaskClient):
    response = client.post(
        "/api/datasets/import",
        data={"files": (BytesIO(b"not a chess database"), "fake.pgn")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "cabeçalho PGN" in response.get_json()["error"]
