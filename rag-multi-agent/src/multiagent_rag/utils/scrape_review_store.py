import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "data", "sessions.db"
)
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(_DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_review_table():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_review (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                url          TEXT NOT NULL,
                title        TEXT,
                preview_text TEXT,
                full_text    TEXT NOT NULL,
                ai_score     INTEGER NOT NULL,
                ai_reason    TEXT,
                seed_url     TEXT,
                status       TEXT NOT NULL DEFAULT 'pending',
                created_at   TEXT NOT NULL,
                actioned_at  TEXT
            )
        """)
        conn.commit()


init_review_table()


def queue_page(
        url: str,
        title: str,
        preview_text: str,
        full_text: str,
        ai_score: int,
        ai_reason: str,
        seed_url: str,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scrape_review
                    (url, title, preview_text, full_text, ai_score, ai_reason, seed_url, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (url, title, preview_text, full_text, ai_score, ai_reason, seed_url, now),
            )
            conn.commit()
            return cursor.lastrowid


def get_pending_queue() -> list:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT id, url, title, preview_text, ai_score, ai_reason, seed_url, created_at "
            "FROM scrape_review WHERE status = 'pending' ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_queue_item(item_id: int) -> Optional[dict]:
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM scrape_review WHERE id = ?", (item_id,)
        ).fetchone()
    return dict(row) if row else None


def approve_item(item_id: int) -> Optional[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            conn.execute(
                "UPDATE scrape_review SET status = 'approved', actioned_at = ? WHERE id = ?",
                (now, item_id),
            )
            conn.commit()
    return get_queue_item(item_id)


def reject_item(item_id: int) -> Optional[dict]:
    now = datetime.now(timezone.utc).isoformat()
    with _lock:
        with _get_connection() as conn:
            conn.execute(
                "UPDATE scrape_review SET status = 'rejected', actioned_at = ? WHERE id = ?",
                (now, item_id),
            )
            conn.commit()
    return get_queue_item(item_id)


def get_stats() -> dict:
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as count FROM scrape_review GROUP BY status"
        ).fetchall()
    return {r["status"]: r["count"] for r in rows}
