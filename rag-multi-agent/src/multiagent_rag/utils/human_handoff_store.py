import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import BaseMessage

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sessions.db")
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_handoff_queue (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id           TEXT NOT NULL,
                query                TEXT NOT NULL,
                final_answer         TEXT,
                emotion              TEXT,
                emotion_confidence   REAL,
                response_confidence  REAL,
                escalation_reason    TEXT,
                intent               TEXT,
                conversation_summary TEXT,
                chat_history_json    TEXT,
                latency_ms_json      TEXT,
                status               TEXT NOT NULL DEFAULT 'pending',
                created_at           TEXT NOT NULL,
                actioned_at          TEXT
            )
        """)
        conn.commit()


_init_db()


def enqueue_handoff(session_id: str, query: str, final_answer: str, emotion: str, emotion_confidence: float,
        response_confidence: float, escalation_reason: str, intent: str, chat_history: List[BaseMessage],
        conversation_summary: Optional[str], latency_ms: dict, ) -> int:
    history_serialized = json.dumps(
        [{"role": m.type, "content": m.content} for m in chat_history if hasattr(m, "content")])
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        with _get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO human_handoff_queue
                    (session_id, query, final_answer, emotion, emotion_confidence,
                     response_confidence, escalation_reason, intent, conversation_summary,
                     chat_history_json, latency_ms_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """, (
                session_id, query, final_answer, emotion, emotion_confidence, response_confidence, escalation_reason,
                intent, conversation_summary, history_serialized, json.dumps(latency_ms), now,), )
            conn.commit()
            return cursor.lastrowid


def get_pending_handoffs() -> list:
    with _get_connection() as conn:
        rows = conn.execute("SELECT id, session_id, query, emotion, escalation_reason, intent, created_at "
                            "FROM human_handoff_queue WHERE status = 'pending' ORDER BY created_at ASC").fetchall()
    return [dict(r) for r in rows]


def get_handoff_detail(handoff_id: int) -> Optional[dict]:
    with _get_connection() as conn:
        row = conn.execute("SELECT * FROM human_handoff_queue WHERE id = ?", (handoff_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["chat_history"] = json.loads(item.pop("chat_history_json", "[]"))
    item["latency_ms"] = json.loads(item.pop("latency_ms_json", "{}"))
    return item


def mark_handled(handoff_id: int):
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            conn.execute("UPDATE human_handoff_queue SET status = 'handled', actioned_at = ? WHERE id = ?",
                (now, handoff_id), )
            conn.commit()


def get_handoff_stats() -> dict:
    with _get_connection() as conn:
        rows = conn.execute("SELECT status, COUNT(*) as count FROM human_handoff_queue GROUP BY status").fetchall()
    return {r["status"]: r["count"] for r in rows}
