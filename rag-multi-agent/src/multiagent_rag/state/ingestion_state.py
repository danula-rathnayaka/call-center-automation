from typing import TypedDict, List, Dict, Any, Optional


class ScrapedPage(TypedDict):
    url: str
    title: str
    text: str
    filter_status: str
    ai_score: int
    ai_reason: str


class IngestionState(TypedDict):
    file_path: str
    chunks: List[Dict[str, Any]]
    document_hash: str
    status: str
    scraped_pages: Optional[List[ScrapedPage]]
