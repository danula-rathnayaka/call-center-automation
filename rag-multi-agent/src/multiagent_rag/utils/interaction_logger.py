import json
import os
from datetime import datetime, timezone

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionLogger:

    def __init__(self):
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        self._log_dir = os.path.join(project_root, "logs")
        self._log_file = os.path.join(self._log_dir, "interactions.jsonl")
        os.makedirs(self._log_dir, exist_ok=True)

    def log_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        emotion: str = "neutral",
        emotion_confidence: float = 0.0,
        response_confidence: float = 0.0,
        should_escalate: bool = False,
        intent: str = "unknown",
        retrieved_docs_count: int = 0,
        latency_ms: dict = None,
    ):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "query": query,
            "response": response,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "response_confidence": response_confidence,
            "should_escalate": should_escalate,
            "intent": intent,
            "retrieved_docs_count": retrieved_docs_count,
            "latency_ms": latency_ms or {},
        }

        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Interaction logged for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

    def get_session_history(self, session_id: str) -> list:
        interactions = []
        try:
            if not os.path.exists(self._log_file):
                return []

            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("session_id") == session_id:
                        interactions.append(entry)
        except Exception as e:
            logger.error(f"Failed to read session history: {str(e)}")

        return interactions

    def get_all_logs(self, limit: int = 100) -> list:
        interactions = []
        try:
            if not os.path.exists(self._log_file):
                return []

            with open(self._log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    interactions.append(json.loads(line))

            return interactions[-limit:][::-1]
        except Exception as e:
            logger.error(f"Failed to read interaction logs: {str(e)}")
            return []
