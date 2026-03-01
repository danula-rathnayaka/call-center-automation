from typing import Optional, List
from pydantic import BaseModel, Field


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    query: str = Field(..., description="The user's query text", min_length=1)
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Auto-generated if not provided."
    )


class ChatResponse(BaseModel):
    """Response body from the chat endpoint."""
    response: str = Field(..., description="The AI-generated response")
    session_id: str = Field(..., description="The session ID for this conversation")
    emotion: str = Field("neutral", description="Detected emotion of the query")
    emotion_confidence: float = Field(0.0, description="Confidence of emotion detection")
    confidence_score: float = Field(0.0, description="Confidence score of the response")
    should_escalate: bool = Field(False, description="Whether the system recommends human escalation")
    intent: str = Field("unknown", description="Classified intent of the query")


# ─── Ingestion ────────────────────────────────────────────────────────────────

class URLIngestionRequest(BaseModel):
    """Request body for URL-based ingestion."""
    url: str = Field(..., description="URL to scrape and ingest")


class IngestionResponse(BaseModel):
    """Response body from ingestion endpoints."""
    status: str = Field(..., description="Ingestion status (completed, failed, skipped)")
    message: str = Field("", description="Additional details about the ingestion")


# ─── Knowledge Base ──────────────────────────────────────────────────────────

class KnowledgeBaseStatus(BaseModel):
    """Response body for knowledge base status."""
    index_name: str = Field(..., description="Name of the Pinecone index")
    total_vectors: int = Field(0, description="Approximate number of vectors in the index")
    status: str = Field("unknown", description="Index status")


class KnowledgeResetResponse(BaseModel):
    """Response body for knowledge base reset."""
    status: str
    message: str


# ─── Session History ──────────────────────────────────────────────────────────

class SessionMessage(BaseModel):
    """A single message in a session history."""
    timestamp: str
    query: str
    response: str
    emotion: str = "neutral"
    confidence_score: float = 0.0


class SessionHistoryResponse(BaseModel):
    """Response body for session history."""
    session_id: str
    messages: List[SessionMessage]
    total_messages: int


# ─── Health ──────────────────────────────────────────────────────────────────

class ComponentStatus(BaseModel):
    """Status of a single system component."""
    name: str
    status: str
    details: str = ""


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""
    status: str = Field(..., description="Overall system status: healthy, degraded, or unhealthy")
    components: List[ComponentStatus]
