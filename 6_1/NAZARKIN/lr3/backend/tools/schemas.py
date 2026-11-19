"""JSON schemas exposed to the LLM.  The LLM can only call these functions."""
from __future__ import annotations

from typing import Any, Dict, List

LABELS = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
ENTITY_TYPES = ["thread", "watch_item", "research_task"]
WRITE_ENTITY_TYPES = ["watch_item", "research_task"]
WATCH_KINDS = ["thread", "ticker", "query"]
WATCH_STATUSES = ["active", "paused", "cancelled", "archived"]
TASK_STATUSES = ["draft", "queued", "in_progress", "done", "cancelled", "archived"]
METRICS = ["count_posts", "label_distribution", "ticker_frequency", "engagement_stats"]
GROUP_BY = ["year", "label", "ticker"]


def _obj(properties: Dict[str, Any], required: List[str] | None = None) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Semantic search over r/WallStreetBets threads. Returns compact "
                "post cards with post_id, title, snippet, score, label, tickers, "
                "year and Reddit URL. Use it whenever the user asks to find posts."
            ),
            "parameters": _obj({
                "query": {"type": "string", "minLength": 1, "maxLength": 500},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
                "tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "labels": {"type": "array", "items": {"type": "string", "enum": LABELS}, "maxItems": 4},
                "year_from": {"type": "integer", "minimum": 2019, "maximum": 2030},
                "year_to": {"type": "integer", "minimum": 2019, "maximum": 2030},
            }, ["query"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get",
            "description": (
                "Get one object by id: a Reddit thread (full body/chunks), a "
                "watch item, or a research task. Never invent ids; use ids from "
                "previous search results or from the conversation."
            ),
            "parameters": _obj({
                "entity_type": {"type": "string", "enum": ENTITY_TYPES},
                "id": {"type": "string", "minLength": 1, "maxLength": 120},
                "include_chunks": {"type": "boolean", "default": True},
            }, ["entity_type", "id"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create",
            "description": (
                "Create a watch item or a research task. This is a WRITE operation: "
                "the backend only stages it and returns a confirmation preview. "
                "The user must confirm before anything is stored."
            ),
            "parameters": _obj({
                "entity_type": {"type": "string", "enum": WRITE_ENTITY_TYPES},
                "data": {"type": "object"},
            }, ["entity_type", "data"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update",
            "description": (
                "Update editable fields of a watch item or research task. WRITE "
                "operation, requires confirmation. Status is changed only through "
                "change_status, not through update."
            ),
            "parameters": _obj({
                "entity_type": {"type": "string", "enum": WRITE_ENTITY_TYPES},
                "id": {"type": "string", "minLength": 1, "maxLength": 120},
                "patch": {"type": "object"},
            }, ["entity_type", "id", "patch"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel",
            "description": (
                "Cancel a watch item or research task (soft cancel, keeps history). "
                "DANGEROUS write operation: always requires explicit confirmation."
            ),
            "parameters": _obj({
                "entity_type": {"type": "string", "enum": WRITE_ENTITY_TYPES},
                "id": {"type": "string", "minLength": 1, "maxLength": 120},
                "reason": {"type": "string", "maxLength": 500},
            }, ["entity_type", "id"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check",
            "description": (
                "Check a service, a watch item or a research task. For a watch item "
                "it runs the saved thread/ticker/query and returns matches, including "
                "new matches since the previous check. Read-side-effect: updates "
                "last_checked_at/last_seen_ids."
            ),
            "parameters": _obj({
                "target": {"type": "string", "enum": ["service", "watch_item", "research_task"]},
                "id": {"type": "string", "maxLength": 120},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            }, ["target"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Calculate aggregate statistics over the indexed threads. Only the "
                "listed metrics are allowed; free-form SQL is never accepted. "
                "Use it for counts, label distributions, ticker frequency and "
                "engagement statistics."
            ),
            "parameters": _obj({
                "metric": {"type": "string", "enum": METRICS},
                "filters": {
                    "type": "object",
                    "properties": {
                        "tickers": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                        "labels": {"type": "array", "items": {"type": "string", "enum": LABELS}, "maxItems": 4},
                        "year_from": {"type": "integer", "minimum": 2019, "maximum": 2030},
                        "year_to": {"type": "integer", "minimum": 2019, "maximum": 2030},
                        "min_post_score": {"type": "integer", "minimum": 0},
                    },
                    "additionalProperties": False,
                },
                "group_by": {"type": "string", "enum": GROUP_BY},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
            }, ["metric"]),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "change_status",
            "description": (
                "Move a watch item or research task to another status using the "
                "backend's transition matrix. WRITE operation, requires confirmation. "
                "Use cancel for cancellation instead of this tool."
            ),
            "parameters": _obj({
                "entity_type": {"type": "string", "enum": WRITE_ENTITY_TYPES},
                "id": {"type": "string", "minLength": 1, "maxLength": 120},
                "to_status": {"type": "string", "enum": WATCH_STATUSES + TASK_STATUSES},
                "reason": {"type": "string", "maxLength": 500},
            }, ["entity_type", "id", "to_status"]),
        },
    },
]


def tool_names() -> List[str]:
    return [item["function"]["name"] for item in TOOLS]


def tool_by_name(name: str) -> Dict[str, Any] | None:
    for item in TOOLS:
        if item["function"]["name"] == name:
            return item
    return None
