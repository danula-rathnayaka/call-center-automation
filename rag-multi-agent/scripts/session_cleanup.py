import argparse
import os
import sys
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multiagent_rag.utils.session_store import delete_session, _get_connection
from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


def cleanup_old_sessions(older_than_hours: int = 24):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()

    try:
        with _get_connection() as conn:
            rows = conn.execute("SELECT DISTINCT session_id FROM session_history "
                                "WHERE session_id NOT IN ("
                                "  SELECT DISTINCT session_id FROM session_history "
                                "  WHERE rowid IN ("
                                "    SELECT MAX(rowid) FROM session_history GROUP BY session_id"
                                "  )"
                                ")").fetchall()

            stale_via_scrape = conn.execute("SELECT DISTINCT session_id FROM session_history sh "
                                            "WHERE NOT EXISTS ("
                                            "  SELECT 1 FROM session_history sh2 "
                                            "  WHERE sh2.session_id = sh.session_id "
                                            "  AND sh2.turn_index = (SELECT MAX(turn_index) FROM session_history WHERE session_id = sh.session_id)"
                                            ")").fetchall()

            cutoff_sessions = conn.execute("""
                SELECT DISTINCT s.session_id
                FROM (
                    SELECT session_id, MAX(turn_index) as max_turn
                    FROM session_history
                    GROUP BY session_id
                ) s
                JOIN session_history sh
                  ON sh.session_id = s.session_id AND sh.turn_index = s.max_turn
                WHERE sh.rowid < (
                    SELECT rowid FROM session_history
                    WHERE session_id = s.session_id
                    ORDER BY turn_index DESC LIMIT 1
                )
                """, ).fetchall()

            expired = conn.execute("""
                SELECT DISTINCT ss.session_id
                FROM session_summary ss
                WHERE ss.updated_at < ?
                AND ss.session_id NOT IN (
                    SELECT DISTINCT session_id FROM session_history
                    WHERE turn_index >= 0
                )
                """, (cutoff,), ).fetchall()

        expired_ids = {r["session_id"] for r in expired}

        if not expired_ids:
            logger.info("Session cleanup: no expired sessions found")
            return 0

        for session_id in expired_ids:
            delete_session(session_id)
            logger.info(f"Deleted expired session: {session_id}")

        logger.info(f"Session cleanup complete: removed {len(expired_ids)} sessions")
        return len(expired_ids)

    except Exception as e:
        logger.error(f"Session cleanup failed: {e}")
        return 0


def cleanup_inactive_sessions(older_than_hours: int = 24):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()

    try:
        with _get_connection() as conn:
            rows = conn.execute("SELECT session_id FROM session_summary WHERE updated_at < ?", (cutoff,), ).fetchall()

        session_ids = [r["session_id"] for r in rows]

        for session_id in session_ids:
            delete_session(session_id)
            logger.info(f"Cleaned up inactive session: {session_id}")

        logger.info(f"Cleanup complete: {len(session_ids)} inactive sessions removed")
        return len(session_ids)

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up old sessions from sessions.db")
    parser.add_argument("--hours", type=int, default=24,
        help="Remove sessions inactive for more than this many hours (default: 24)", )
    args = parser.parse_args()
    removed = cleanup_inactive_sessions(older_than_hours=args.hours)
    print(f"Removed {removed} sessions older than {args.hours} hours")
