from typing import TypedDict, List, Dict, Any


class IngestionState(TypedDict):
    file_path: str
    chunks: List[Dict[str, Any]]
    status: str
