"""Configuration for the WSB Thread Radar backend."""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LR3_ROOT = HERE.parent
RUNTIME_DIR = Path(os.environ.get("LR3_RUNTIME_DIR", str(LR3_ROOT / "runtime")))
IS_DB = Path(os.environ.get("LR3_IS_DB", str(RUNTIME_DIR / "is.db")))
INDEX_DIR = os.environ.get("LR3_INDEX_DIR", "")
MODEL_DIR = os.environ.get("LR3_MODEL_DIR", "")

HOST = os.environ.get("LR3_BACKEND_HOST", "127.0.0.1")
PORT = int(os.environ.get("LR3_BACKEND_PORT", "8000"))
INTERNAL_TOKEN = os.environ.get("LR3_INTERNAL_TOKEN", "")
VERSION = "1.0.0"

# Read/write policy.  Kept in one place so both the tool gateway and the
# prompt builder can use the same source of truth.
READ_TOOLS = {"search", "get", "calculate"}
READ_SIDE_EFFECT_TOOLS = {"check"}
WRITE_TOOLS = {"create", "update", "cancel", "change_status"}
DANGEROUS_TOOLS = {"cancel", "change_status"}

WATCH_STATUSES = ("active", "paused", "cancelled", "archived")
TASK_STATUSES = ("draft", "queued", "in_progress", "done", "cancelled", "archived")

WATCH_TRANSITIONS = {
    "active": {"paused", "cancelled", "archived"},
    "paused": {"active", "cancelled", "archived"},
    "cancelled": {"archived"},
    "archived": set(),
}
TASK_TRANSITIONS = {
    "draft": {"queued", "cancelled"},
    "queued": {"in_progress", "cancelled"},
    "in_progress": {"done", "cancelled"},
    "done": {"archived"},
    "cancelled": {"archived"},
    "archived": set(),
}
