import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.messages import BaseMessage

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing_dataset", "sessions.db")
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    with _get_connection() as conn:
        try:
            cursor = conn.execute("PRAGMA table_info(human_handoff_queue)")
            columns = cursor.fetchall()
            if columns:
                id_col = next((c for c in columns if c["name"] == "id"), None)
                if id_col and id_col["type"].upper() == "INTEGER":
                    logger.warning("Detected legacy INTEGER id schema. Recreating handoff table as TEXT.")
                    conn.execute("DROP TABLE human_handoff_queue")
        except Exception as e:
            logger.error(f"Migration check failed: {e}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_handoff_queue (
                id                   TEXT PRIMARY KEY,
                session_id           TEXT NOT NULL,
                phone_number         TEXT,
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
                status               TEXT NOT NULL DEFAULT 'ringing',
                created_at           TEXT NOT NULL,
                answered_at          TEXT,
                actioned_at          TEXT
            )
        """)
        for col, definition in [
            ("answered_at", "TEXT"),
            ("phone_number", "TEXT"),
            ("actioned_at", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE human_handoff_queue ADD COLUMN {col} {definition}")
            except Exception:
                pass
        conn.commit()


_init_db()


def enqueue_handoff(
    session_id: str,
    query: str,
    final_answer: str,
    emotion: str,
    emotion_confidence: float,
    response_confidence: float,
    escalation_reason: str,
    intent: str,
    chat_history: List[BaseMessage],
    conversation_summary: Optional[str],
    latency_ms: dict,
    phone_number: Optional[str] = None,
) -> str:
    handoff_id = str(uuid.uuid4())
    history_serialized = json.dumps(
        [{"role": m.type, "content": m.content} for m in chat_history if hasattr(m, "content")]
    )
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        with _get_connection() as conn:
            conn.execute(
                """
                INSERT INTO human_handoff_queue
                    (id, session_id, phone_number, query, final_answer, emotion, emotion_confidence,
                     response_confidence, escalation_reason, intent, conversation_summary,
                     chat_history_json, latency_ms_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ringing', ?)
                """,
                (
                    str(handoff_id), str(session_id), str(phone_number) if phone_number else None,
                    str(query), str(final_answer), str(emotion), 
                    float(emotion_confidence) if emotion_confidence is not None else 0.0,
                    float(response_confidence) if response_confidence is not None else 0.0, 
                    str(escalation_reason) if escalation_reason else "",
                    str(intent), str(conversation_summary) if conversation_summary else "", 
                    history_serialized, json.dumps(latency_ms), now,
                ),
            )
            conn.commit()
    logger.info(f"Handoff enqueued id={handoff_id} session={session_id} phone={phone_number}")
    return handoff_id


def get_active_handoffs() -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, phone_number, query, emotion, escalation_reason,
                   intent, status, created_at, answered_at
            FROM human_handoff_queue
            WHERE status IN ('ringing', 'answered')
            ORDER BY created_at ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_ended_handoffs() -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, session_id, phone_number, query, emotion, escalation_reason,
                   intent, status, created_at, answered_at, actioned_at
            FROM human_handoff_queue
            WHERE status = 'ended'
            ORDER BY actioned_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_handoff_detail(handoff_id: str) -> Optional[dict]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM human_handoff_queue WHERE id = ?", (handoff_id,)
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["chat_history"] = json.loads(item.pop("chat_history_json", "[]"))
    item["latency_ms"] = json.loads(item.pop("latency_ms_json", "{}"))
    return item


def mark_answered(handoff_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            conn.execute(
                "UPDATE human_handoff_queue SET status='answered', answered_at=? WHERE id=? AND status='ringing'",
                (now, handoff_id),
            )
            conn.commit()


def mark_ended(handoff_id: str):
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            conn.execute(
                "UPDATE human_handoff_queue SET status='ended', actioned_at=? "
                "WHERE id=? AND status IN ('ringing','answered')",
                (now, handoff_id),
            )
            conn.commit()


def mark_handled(handoff_id: str):
    mark_ended(handoff_id)


def get_handoff_stats() -> dict:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM human_handoff_queue GROUP BY status"
        ).fetchall()
    return {r["status"]: r["count"] for r in rows}


def get_pending_handoffs() -> list:
    return get_active_handoffs()
