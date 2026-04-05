import os
import sqlite3
import threading
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "sessions.db")
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_db():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_history (
                session_id  TEXT NOT NULL,
                turn_index  INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                PRIMARY KEY (session_id, turn_index)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_summary (
                session_id  TEXT PRIMARY KEY,
                summary     TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)
        conn.commit()


_init_db()


def _serialize_message(msg: BaseMessage) -> dict:
    if isinstance(msg, HumanMessage):
        role = "human"
    elif isinstance(msg, AIMessage):
        role = "ai"
    elif isinstance(msg, SystemMessage):
        role = "system"
    else:
        role = "system"
    return {"role": role, "content": msg.content}


def _deserialize_message(role: str, content: str) -> BaseMessage:
    if role == "human":
        return HumanMessage(content=content)
    elif role == "ai":
        return AIMessage(content=content)
    else:
        return SystemMessage(content=content)


def load_history(session_id: str) -> List[BaseMessage]:
    try:
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_history "
                "WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        return [_deserialize_message(r["role"], r["content"]) for r in rows]
    except Exception as e:
        logger.error(f"Failed to load history for session {session_id}: {e}")
        return []


def save_history(session_id: str, history: List[BaseMessage]):
    try:
        serialized = [_serialize_message(m) for m in history]
        with _lock:
            with _get_connection() as conn:
                conn.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
                conn.executemany(
                    "INSERT INTO session_history (session_id, turn_index, role, content) VALUES (?, ?, ?, ?)",
                    [(session_id, i, s["role"], s["content"]) for i, s in enumerate(serialized)],
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to save history for session {session_id}: {e}")


def load_summary(session_id: str) -> Optional[str]:
    try:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT summary FROM session_summary WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["summary"] if row else None
    except Exception as e:
        logger.error(f"Failed to load summary for session {session_id}: {e}")
        return None


def save_summary(session_id: str, summary: str):
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with _lock:
            with _get_connection() as conn:
                conn.execute(
                    "INSERT INTO session_summary (session_id, summary, updated_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, updated_at=excluded.updated_at",
                    (session_id, summary, now),
                )
                conn.commit()
    except Exception as e:
        logger.error(f"Failed to save summary for session {session_id}: {e}")


def delete_session(session_id: str):
    try:
        with _lock:
            with _get_connection() as conn:
                conn.execute("DELETE FROM session_history WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM session_summary WHERE session_id = ?", (session_id,))
                conn.commit()
        logger.info(f"Session {session_id} data deleted from store")
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
