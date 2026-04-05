from typing import Optional, List, Union

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(None)
    phone_number: Optional[str] = Field(None, description="Customer's phone number, captured at session start")


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
    handoff_uuid: Optional[str] = Field(
        None,
        description="UUID of the enqueued handoff. Only present when confidence.should_escalate=true. "
                    "Use this to track the call on the agent dashboard — do NOT display to the customer."
    )


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


class DocumentEntry(BaseModel):
    source: str
    type: str
    document_hash: str = ""


class IngestedDocumentsResponse(BaseModel):
    total: int
    documents: List[DocumentEntry]


class IngestedURLsResponse(BaseModel):
    total: int
    urls: List[DocumentEntry]


class DeleteKnowledgeResponse(BaseModel):
    status: str
    source: str
    vectors_deleted: int
    message: str


class ToolEntry(BaseModel):
    tool_name: str
    description: str
    api_url: str
    http_method: str


class ToolListResponse(BaseModel):
    total: int
    tools: List[ToolEntry]


class DeleteToolResponse(BaseModel):
    status: str
    tool_name: str
    message: str


class HandoffQueueItem(BaseModel):
    id: Union[str, int]
    session_id: str
    phone_number: Optional[str] = None
    query: str
    emotion: Optional[str] = "neutral"
    escalation_reason: Optional[str] = ""
    intent: Optional[str] = "unknown"
    status: str
    created_at: str
    answered_at: Optional[str] = None
    actioned_at: Optional[str] = None


class HandoffQueueResponse(BaseModel):
    total: int
    ringing: int
    answered: int
    items: List[HandoffQueueItem]


class HandoffHistoryResponse(BaseModel):
    total: int
    items: List[HandoffQueueItem]


class HandoffDashboardResponse(BaseModel):
    ringing: int
    answered: int
    ended: int
    active_calls: List[HandoffQueueItem]


class HandoffActionResponse(BaseModel):
    status: str
    handoff_id: str
    session_id: str
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


class ToolParameter(BaseModel):
    name: str = Field(...)
    type: str = Field(...)
    description: str = Field(...)


class ToolRegistrationRequest(BaseModel):
    tool_name: str = Field(...)
    description: str = Field(...)
    api_url: str = Field(...)
    http_method: str = Field(default="POST")
    parameters: List[ToolParameter] = Field(default_factory=list)
