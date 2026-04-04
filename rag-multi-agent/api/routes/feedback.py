import json
import os
import threading
from datetime import datetime, timezone

from api.schemas import FeedbackRequest, FeedbackResponse
from fastapi import APIRouter, HTTPException

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["Feedback"])

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_log_dir = os.path.join(_project_root, "logs")
_feedback_file = os.path.join(_log_dir, "feedback.jsonl")
os.makedirs(_log_dir, exist_ok=True)

_write_lock = threading.Lock()


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Submit customer feedback on an AI response",
    description=(
        "Records a thumbs-up/thumbs-down/neutral rating for a specific AI response. "
        "Include the `session_id`, the exact `query` and `response` text, and a `rating` of `positive`, `negative`, or `neutral`. "
        "Optionally provide `correct_answer` (what the right answer should have been) and a `comment`. "
        "This data is appended to a JSONL log file and can feed future model fine-tuning or quality reviews. "
        "Show a feedback widget (thumbs up/down) on each chat bubble and call this endpoint on interaction."
    ),
)
async def submit_feedback(request: FeedbackRequest):
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "session_id": request.session_id,
        "query": request.query, "response": request.response, "rating": request.rating,
        "correct_answer": request.correct_answer, "comment": request.comment, }
    try:
        with _write_lock:
            with open(_feedback_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Feedback logged for session {request.session_id} - rating: {request.rating}")
        return FeedbackResponse(status="recorded",
            message="Feedback has been recorded. Thank you for helping improve the system.")
    except Exception as e:
        logger.error(f"Failed to log feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get(
    "/stats",
    summary="Get aggregated feedback rating counts",
    description=(
        "Returns the total number of feedback submissions broken down by rating: `positive`, `negative`, `neutral`. "
        "Use this on an admin metrics dashboard to monitor overall response quality trends over time."
    ),
)
async def get_feedback_stats():
    try:
        if not os.path.exists(_feedback_file):
            return {"total": 0, "positive": 0, "negative": 0, "neutral": 0}
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        with open(_feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                rating = entry.get("rating", "neutral")
                if rating in counts:
                    counts[rating] += 1
        return {"total": sum(counts.values()), **counts}
    except Exception as e:
        logger.error(f"Failed to read feedback stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve feedback stats: {str(e)}")
