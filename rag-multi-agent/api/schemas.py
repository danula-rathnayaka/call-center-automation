from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None)


class RetrievedChunk(BaseModel):
    content: str = Field("")
    source: str = Field("")
    chunk_type: str = Field("")


class EmotionResult(BaseModel):
    emotion: str = Field("neutral")
    confidence: float = Field(0.0)


class ConfidenceResult(BaseModel):
    score: float = Field(0.0)
    should_escalate: bool = Field(False)
    reason: str = Field("")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    transcribed_text: str = ""
    emotion: EmotionResult = Field(default_factory=EmotionResult)
    intent: str = "unknown"
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    confidence: ConfidenceResult = Field(default_factory=ConfidenceResult)
    latency_ms: dict = Field(default_factory=dict)


class URLIngestionRequest(BaseModel):
    url: str


class IngestionResponse(BaseModel):
    status: str
    message: str = ""
    chunks_ingested: int = 0


class BatchIngestionResponse(BaseModel):
    results: List[IngestionResponse]
    total_files: int
    successful: int
    failed: int
    duplicates: int


class KnowledgeBaseStatus(BaseModel):
    index_name: str
    total_vectors: int = 0
    status: str = "unknown"


class KnowledgeResetResponse(BaseModel):
    status: str
    message: str


class SessionMessage(BaseModel):
    timestamp: str
    query: str
    response: str
    emotion: str = "neutral"
    confidence_score: float = 0.0


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[SessionMessage]
    total_messages: int


class ComponentStatus(BaseModel):
    name: str
    status: str
    details: str = ""


class HealthResponse(BaseModel):
    status: str
    components: List[ComponentStatus]


class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(positive|negative|neutral)$")
    correct_answer: Optional[str] = Field(None)
    comment: Optional[str] = Field(None)


class FeedbackResponse(BaseModel):
    status: str
    message: str
