import re

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)",
    r"disregard\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(everything|all|your)\s+(instructions|rules|prompts)",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"new\s+instructions?\s*:",
    r"system\s*prompt\s*:",
    r"act\s+as\s+(a|an)\s+(?!customer|user|agent)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"override\s+(your\s+)?(instructions|programming|rules)",
    r"repeat\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)",
    r"what\s+(is|are)\s+your\s+(system\s+)?(prompt|instructions|rules)",
    r"bypassing",
    r"developer\s+mode",
    r"\bdan\b",
    r"give\s+me\s+all\s+(the\s+)?data",
    r"delete\s+(the\s+)?database",
    r"list\s+all\s+users",
    r"show\s+me\s+all\s+(customer|user)\s+(data|records|details)",
    r"export\s+(all\s+)?(data|records)",
    r"(drop|truncate|delete)\s+(table|database|index)",
]

_PII_PHONE = re.compile(r"\b0\d{9}\b|\b\d{10}\b|\b\+\d{1,3}[\s\-]?\d{7,12}\b")
_PII_NIC = re.compile(r"\b\d{9}[VvXx]\b|\b\d{12}\b")
_PII_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PII_CARD = re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")


class PIIHandler:

    def sanitize_for_rag(self, text: str) -> str:
        sanitized = _PII_PHONE.sub("[REDACTED]", text)
        sanitized = _PII_NIC.sub("[REDACTED]", sanitized)
        sanitized = _PII_EMAIL.sub("[REDACTED]", sanitized)
        return sanitized

    def sanitize_response(self, response: str) -> str:
        sanitized = _PII_PHONE.sub("[REDACTED]", response)
        sanitized = _PII_NIC.sub("[REDACTED]", sanitized)
        sanitized = _PII_EMAIL.sub("[REDACTED]", sanitized)
        sanitized = _PII_CARD.sub("[REDACTED]", sanitized)
        return sanitized


class SafetyGuardrail:

    def __init__(self):
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS
        ]

    def validate(self, query: str) -> dict:
        if not query or not query.strip():
            return {
                "safe": False,
                "reason": "I didn't catch that. Could you please repeat your question?",
                "block_type": "empty",
            }

        if len(query) > 2000:
            return {
                "safe": False,
                "reason": "Your message is too long. Could you please keep it shorter and I will do my best to help?",
                "block_type": "length",
            }

        if _PII_CARD.search(query):
            logger.warning(f"Card number detected in query: {query[:40]}")
            return {
                "safe": False,
                "reason": "Payment card numbers cannot be shared over this channel. Please contact us through a secure channel for card-related queries.",
                "block_type": "card",
            }

        query_lower = query.lower().strip()
        for pattern in self._injection_patterns:
            if pattern.search(query_lower):
                logger.warning(f"Prompt injection attempt detected: {query[:80]}")
                return {
                    "safe": False,
                    "reason": "I am not able to process that kind of request. Please ask me something related to our telecom services.",
                    "block_type": "injection",
                }

        return {"safe": True, "reason": "", "block_type": None}


class Guardrail:

    def __init__(self):
        self.safety = SafetyGuardrail()
        self.pii = PIIHandler()

    def validate(self, query: str) -> dict:
        return self.safety.validate(query)

    def sanitize_response(self, response: str) -> str:
        return self.pii.sanitize_response(response)
