from __future__ import annotations

import gc
import json
import sqlite3
import sys
import threading
import time
import uuid
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


FRONTEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, DATA_CONFIG, GENERATION_CONFIG  # noqa: E402
from test import MusicGenTester  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DATABASE_PATH = PROJECT_ROOT / "musicgen_history.sqlite3"

app = FastAPI(title="MusicGenAI", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/audio", StaticFiles(directory=OUTPUT_DIR), name="audio")

MODEL_LOCK = threading.Lock()
_tester: MusicGenTester | None = None
_checkpoint_path: str | None = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    session_id: str | None = None


@contextmanager
def database():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with database() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                audio_url TEXT,
                duration REAL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON messages(session_id, id);
            """
        )


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def select_checkpoint() -> Path:
    candidates = [
        CHECKPOINT_DIR / DATA_CONFIG["run_name"] / "best_musicgen_model.pt",
        CHECKPOINT_DIR / DATA_CONFIG["run_name"] / "latest_musicgen_model.pt",
        CHECKPOINT_DIR / "best_musicgen_model.pt",
        CHECKPOINT_DIR / "latest_musicgen_model.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Chưa tìm thấy checkpoint. Hãy chạy train.py trước khi tạo nhạc."
    )


def release_model() -> None:
    global _tester, _checkpoint_path
    _tester = None
    _checkpoint_path = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tester() -> tuple[MusicGenTester, Path]:
    global _tester, _checkpoint_path
    checkpoint = select_checkpoint()
    checkpoint_string = str(checkpoint)
    if _tester is None or _checkpoint_path != checkpoint_string:
        release_model()
        _tester = MusicGenTester(checkpoint_string)
        _checkpoint_path = checkpoint_string
    return _tester, checkpoint


def create_or_get_session(connection, session_id: str | None, prompt: str) -> str:
    if session_id:
        existing = connection.execute(
            "SELECT id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if existing:
            return session_id

    session_id = uuid.uuid4().hex
    words = prompt.split()
    title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")
    timestamp = now_iso()
    connection.execute(
        "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, timestamp, timestamp),
    )
    return session_id


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    checkpoint = None
    try:
        checkpoint = str(select_checkpoint().relative_to(PROJECT_ROOT))
    except FileNotFoundError:
        pass
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "checkpoint": checkpoint,
    }


@app.post("/api/generate")
def generate(request: GenerateRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt không được để trống.")

    with database() as connection:
        session_id = create_or_get_session(
            connection, request.session_id, prompt
        )
        timestamp = now_iso()
        connection.execute(
            """
            INSERT INTO messages(session_id, role, content, created_at)
            VALUES (?, 'user', ?, ?)
            """,
            (session_id, prompt, timestamp),
        )
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (timestamp, session_id),
        )

    filename = f"music_{session_id[:8]}_{int(time.time())}.wav"
    output_path = OUTPUT_DIR / filename
    started_at = time.perf_counter()
    try:
        with MODEL_LOCK:
            tester, checkpoint = get_tester()
            tester.generate_music(
                prompt=prompt,
                output_path=str(output_path),
                duration_sec=GENERATION_CONFIG["duration_sec"],
                temperature=GENERATION_CONFIG["temperature"],
                top_k=GENERATION_CONFIG["top_k"],
                guidance_scale=GENERATION_CONFIG["guidance_scale"],
            )
    except torch.OutOfMemoryError as exc:
        release_model()
        raise HTTPException(
            status_code=503,
            detail="GPU hết bộ nhớ. Hãy đóng ứng dụng đang sử dụng GPU rồi thử lại.",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - started_at
    audio_url = f"/audio/{filename}"
    metadata = {
        "mode": "model",
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "generation_seconds": round(elapsed, 2),
        "temperature": GENERATION_CONFIG["temperature"],
        "top_k": GENERATION_CONFIG["top_k"],
        "guidance_scale": GENERATION_CONFIG["guidance_scale"],
    }
    with database() as connection:
        timestamp = now_iso()
        connection.execute(
            """
            INSERT INTO messages(
                session_id, role, content, audio_url, duration, metadata, created_at
            ) VALUES (?, 'assistant', ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                "Bản nhạc vừa tạo",
                audio_url,
                GENERATION_CONFIG["duration_sec"],
                json.dumps(metadata, ensure_ascii=False),
                timestamp,
            ),
        )
        connection.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (timestamp, session_id),
        )

    return {
        "status": "success",
        "session_id": session_id,
        "audio_url": audio_url,
        "duration": GENERATION_CONFIG["duration_sec"],
        "metadata": metadata,
    }


@app.get("/api/history")
def history():
    with database() as connection:
        rows = connection.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS message_count
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.id
            GROUP BY s.id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/history/{session_id}")
def history_detail(session_id: str):
    with database() as connection:
        session = connection.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
        messages = connection.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    message_list = []
    for row in messages:
        item = dict(row)
        item["metadata"] = json.loads(item["metadata"] or "{}")
        message_list.append(item)
    result = dict(session)
    result["messages"] = message_list
    return result


@app.delete("/api/history/{session_id}")
def delete_history(session_id: str):
    with database() as connection:
        audio_rows = connection.execute(
            "SELECT audio_url FROM messages WHERE session_id = ? AND audio_url IS NOT NULL",
            (session_id,),
        ).fetchall()
        connection.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        deleted = connection.execute(
            "DELETE FROM sessions WHERE id = ?", (session_id,)
        ).rowcount
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy hội thoại.")
    for row in audio_rows:
        filename = Path(row["audio_url"]).name
        audio_path = (OUTPUT_DIR / filename).resolve()
        if audio_path.parent == OUTPUT_DIR.resolve() and audio_path.is_file():
            audio_path.unlink()
    return {"status": "success"}


@app.delete("/api/history")
def clear_history():
    with database() as connection:
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM sessions")
    for audio_path in OUTPUT_DIR.glob("music_*.wav"):
        audio_path.unlink()
    return {"status": "success"}


initialize_database()


if __name__ == "__main__":
    threading.Timer(
        1.2, lambda: webbrowser.open("http://127.0.0.1:7860")
    ).start()
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="info")
