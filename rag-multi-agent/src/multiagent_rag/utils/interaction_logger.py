import json
import os
from datetime import datetime

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class InteractionLogger:
    """
    Logs all RAG interactions (queries, responses, emotions, confidence scores)
    to a JSON lines file for transparency and future performance analysis.
    """

    def __init__(self):
        # Resolve logs directory relative to the project root
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
    ):
        """
        Log a single interaction to the JSONL file.

        Args:
            session_id: Unique session identifier
            query: The user's original query
            response: The generated response
            emotion: Detected emotion of the query
            emotion_confidence: Confidence of emotion detection
            response_confidence: Confidence score of the response
            should_escalate: Whether the system recommends human escalation
            intent: The classified intent (technical, casual, etc.)
            retrieved_docs_count: Number of documents retrieved
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "query": query,
            "response": response,
            "emotion": emotion,
            "emotion_confidence": emotion_confidence,
            "response_confidence": response_confidence,
            "should_escalate": should_escalate,
            "intent": intent,
            "retrieved_docs_count": retrieved_docs_count,
        }

        try:
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Interaction logged for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to log interaction: {str(e)}")

    def get_session_history(self, session_id: str) -> list:
        """
        Retrieve all logged interactions for a specific session.

        Args:
            session_id: The session ID to filter by

        Returns:
            List of interaction dicts for the given session, ordered by timestamp
        """
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
        """
        Retrieve the most recent interaction logs.

        Args:
            limit: Maximum number of entries to return (most recent first)

        Returns:
            List of interaction dicts, ordered by most recent first
        """
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

            # Return most recent first
            return interactions[-limit:][::-1]
        except Exception as e:
            logger.error(f"Failed to read interaction logs: {str(e)}")
            return []
