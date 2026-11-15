"""HTTP client for the backend tool gateway."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List

from . import config


class BackendError(Exception):
    def __init__(self, code: str, message: str, details: Dict[str, Any] | None = None,
                 retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.retryable = retryable

    def as_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "details": self.details, "retryable": self.retryable}


class BackendClient:
    def __init__(self, base_url: str = config.BACKEND_URL,
                 token: str = config.INTERNAL_TOKEN, timeout: float = 90):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._tools: List[Dict[str, Any]] | None = None

    def _request(self, method: str, path: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token:
            headers["X-Internal-Token"] = self.token
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = {}
            error = parsed.get("error") or {}
            raise BackendError(
                error.get("code") or "BACKEND_HTTP_ERROR",
                error.get("message") or text[:500] or str(exc),
                error.get("details") or {},
                bool(error.get("retryable")),
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendError("BACKEND_UNAVAILABLE", f"{type(exc).__name__}: {exc}",
                               retryable=True)
        except Exception as exc:
            raise BackendError("BACKEND_ERROR", f"{type(exc).__name__}: {exc}")

    def tools(self) -> List[Dict[str, Any]]:
        if self._tools is not None:
            return self._tools
        payload = self._request("GET", "/tools")
        self._tools = payload.get("tools") or []
        return self._tools

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> Dict[str, Any]:
        return self._request("GET", "/stats")

    def call(self, tool: str, arguments: Dict[str, Any],
             conversation_id: str = "") -> Dict[str, Any]:
        return self._request("POST", "/tools/call", {
            "tool": tool,
            "arguments": arguments,
            "conversation_id": conversation_id,
        })

    def confirm(self, action_id: str, decision: str,
                conversation_id: str = "") -> Dict[str, Any]:
        return self._request("POST", "/tools/confirm", {
            "action_id": action_id,
            "decision": decision,
            "conversation_id": conversation_id,
        })
