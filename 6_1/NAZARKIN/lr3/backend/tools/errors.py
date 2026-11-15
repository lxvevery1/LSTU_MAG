"""Structured errors returned by the tool gateway."""
from __future__ import annotations

from typing import Any, Dict


class ToolError(Exception):
    def __init__(self, code: str, message: str, details: Dict[str, Any] | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }
