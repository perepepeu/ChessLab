from __future__ import annotations

import uuid
from pathlib import Path

import chess
from flask import Flask, jsonify, render_template, request, send_file

from chesslab.data import DatasetStore
from chesslab.competition import TournamentManager
from chesslab.replays import ReplayStore
from chesslab.search import choose_with_search
from chesslab.sessions import GameSessionStore
from chesslab.training import TrainingManager


ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
datasets = DatasetStore(ROOT / "data")
replays = ReplayStore(ROOT)
trainer = TrainingManager(ROOT, datasets, replays)
tournament = TournamentManager(ROOT, replays)
games = GameSessionStore(max_sessions=100, ttl_seconds=12 * 60 * 60)


def ok(**payload):
    return jsonify({"ok": True, **payload})


@app.errorhandler(Exception)
def handle_error(error):
    code = getattr(error, "code", 400)
    return jsonify({"ok": False, "error": str(error)}), code if isinstance(code, int) else 400


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/dashboard")
def dashboard():
    items = datasets.list()
    models = trainer.list_models()
    return ok(datasets=items, models=models, training=trainer.status(), summary={
        "games": sum(d["games"] for d in items), "positions": sum(d["positions"] for d in items),
        "models": len(models), "active_model": trainer.model.name,
        "parameters": trainer.model.parameter_count, "trained_positions": trainer.model.trained_positions,
        "guided_examples": len(datasets.list_guided()), "replays": len(replays.list()),
    })


@app.get("/api/datasets")
def list_datasets():
    return ok(datasets=datasets.list())


@app.patch("/api/datasets/<dataset_id>")
def rename_dataset(dataset_id):
    if trainer.status()["running"]:
        raise ValueError("Aguarde o treino atual terminar antes de alterar datasets.")
    body = request.get_json(force=True) or {}
    return ok(dataset=datasets.rename(dataset_id, body.get("name", "")), datasets=datasets.list())


@app.delete("/api/datasets/<dataset_id>")
def delete_dataset(dataset_id):
    if trainer.status()["running"]:
        raise ValueError("Aguarde o treino atual terminar antes de apagar datasets.")
    return ok(deleted=datasets.delete(dataset_id), datasets=datasets.list())


@app.post("/api/datasets/import")
def import_dataset():
    files = request.files.getlist("files")
    if not files:
        raise ValueError("Selecione pelo menos um arquivo .pgn.")
    imported = []
    for file in files:
        if not (file.filename or "").lower().endswith(".pgn"):
            raise ValueError(f"{file.filename}: somente arquivos .pgn são aceitos.")
        imported.append(datasets.register_upload(file))
    return ok(datasets=imported)


@app.get("/api/guided")
def list_guided():
    return ok(examples=datasets.list_guided())


@app.get("/api/guided/position")
def guided_position():
    return ok(position=datasets.guided_position(request.args.get("fen")))


@app.post("/api/guided/examples")
def add_guided_example():
    body = request.get_json(force=True) or {}
    item = datasets.add_guided(body.get("fen", ""), body.get("move", ""), body.get("note", ""), body.get("priority", 3))
    board = chess.Board(item["fen"])
    board.push(chess.Move.from_uci(item["move"]))
    return ok(example=item, next_position=datasets.guided_position(board.fen()), examples=datasets.list_guided())


@app.delete("/api/guided/examples/<example_id>")
def delete_guided_example(example_id):
    datasets.delete_guided(example_id)
    return ok(examples=datasets.list_guided())


@app.post("/api/training/start")
def start_training():
    return ok(training=trainer.start(request.get_json(force=True) or {}))


@app.get("/api/training/status")
def training_status():
    return ok(training=trainer.status())


@app.post("/api/training/stop")
def stop_training():
    trainer.stop()
    return ok(training=trainer.status())


@app.get("/api/models")
def list_models():
    return ok(models=trainer.list_models())


@app.post("/api/models/import")
def import_model():
    file = request.files.get("file")
    if not file or not (file.filename or "").lower().endswith(".npz"):
        raise ValueError("Selecione um checkpoint .npz do ChessLab.")
    return ok(model=trainer.import_checkpoint(file), models=trainer.list_models())


@app.post("/api/models/<model_id>/load")
def load_model(model_id):
    return ok(model=trainer.load_model(model_id))


@app.delete("/api/models/<model_id>")
def delete_model(model_id):
    if trainer.status()["running"]:
        raise ValueError("Pare o treinamento antes de apagar um modelo.")
    return ok(result=trainer.delete_model(model_id), models=trainer.list_models())


@app.get("/api/models/<model_id>/download")
def download_model(model_id):
    path = ROOT / "models" / f"{Path(model_id).name}.npz"
    if not path.exists():
        raise ValueError("Checkpoint não encontrado.")
    return send_file(path, as_attachment=True, download_name=path.name)


@app.get("/api/network")
def network():
    fen = request.args.get("fen")
    board = chess.Board(fen) if fen else chess.Board()
    return ok(network=trainer.model.snapshot(board), model={"name": trainer.model.name, "id": trainer.active_model_id})


@app.get("/api/replays")
def list_replays():
    return ok(replays=replays.list())


@app.get("/api/replays/<replay_id>")
def replay_detail(replay_id):
    return ok(replay=replays.detail(replay_id))


@app.post("/api/tournament/start")
def start_tournament():
    return ok(tournament=tournament.start(request.get_json(force=True) or {}))


@app.get("/api/tournament/status")
def tournament_status():
    return ok(tournament=tournament.status())


@app.post("/api/tournament/stop")
def stop_tournament():
    tournament.stop()
    return ok(tournament=tournament.status())


def board_payload(board: chess.Board, session_id: str, ai_move: str | None = None):
    outcome = board.outcome(claim_draw=True)
    return {"session_id": session_id, "fen": board.fen(), "turn": "white" if board.turn else "black",
            "legal_moves": [m.uci() for m in board.legal_moves], "ai_move": ai_move,
            "game_over": board.is_game_over(claim_draw=True), "result": outcome.result() if outcome else None,
            "check": board.is_check()}


@app.post("/api/play/new")
def new_game():
    body = request.get_json(silent=True) or {}
    color = body.get("color", "white")
    strength = body.get("strength", "tactical") if body.get("strength") in {"policy", "tactical", "search2"} else "tactical"
    session_id = uuid.uuid4().hex
    board = chess.Board()
    ai_move = None
    if color == "black":
        move = choose_with_search(trainer.model, board, strength)
        if move:
            board.push(move)
            ai_move = move.uci()
    games.put(session_id, board, strength)
    return ok(game=board_payload(board, session_id, ai_move))


@app.post("/api/play/move")
def play_move():
    body = request.get_json(force=True)
    session_id = body.get("session_id", "")
    session = games.get(session_id)
    if session is None:
        raise ValueError("Partida expirada. Inicie uma nova partida.")
    board = session["board"]
    try:
        move = chess.Move.from_uci(body.get("move", ""))
    except ValueError as exc:
        raise ValueError("Lance inválido.") from exc
    if move not in board.legal_moves:
        raise ValueError("Esse lance não é legal nesta posição.")
    board.push(move)
    ai_uci = None
    if not board.is_game_over(claim_draw=True):
        ai_move = choose_with_search(trainer.model, board, session["strength"])
        if ai_move:
            board.push(ai_move)
            ai_uci = ai_move.uci()
    return ok(game=board_payload(board, session_id, ai_uci))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
