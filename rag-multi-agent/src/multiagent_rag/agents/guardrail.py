import re

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(everything|all|your)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"act\s+as\s+(a|an)\s+(?!customer|user)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your\s+)?(instructions|programming|rules)",
    r"repeat\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions|rules)",
]

PII_PATTERNS = [
    r"\b\d{9}[VvXx]\b",
    r"\b\d{12}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
]


class Guardrail:

    def __init__(self):
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
        ]
        self._pii_patterns = [
            re.compile(p) for p in PII_PATTERNS
        ]

    def validate(self, query: str) -> dict:
        if not query or not query.strip():
            return {"safe": False, "reason": "Empty query received"}

        query_lower = query.lower().strip()

        for pattern in self._injection_patterns:
            if pattern.search(query_lower):
                logger.warning(f"Prompt injection detected in query: {query[:80]}")
                return {
                    "safe": False,
                    "reason": "Query contains potentially unsafe instructions"
                }

        if len(query) > 2000:
            logger.warning(f"Query exceeds maximum length: {len(query)} chars")
            return {
                "safe": False,
                "reason": "Query exceeds maximum allowed length"
            }

        return {"safe": True, "reason": ""}

    def sanitize_response(self, response: str) -> str:
        sanitized = response
        for pattern in self._pii_patterns:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized
