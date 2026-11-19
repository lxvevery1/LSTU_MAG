"""Configuration for the LLM gateway."""
from __future__ import annotations

import os

HOST = os.environ.get("LR3_LLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("LR3_LLM_PORT", "8010"))
BACKEND_URL = os.environ.get("LR3_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
INTERNAL_TOKEN = os.environ.get("LR3_INTERNAL_TOKEN", "")

# OpenAI-compatible provider.  The default is the public OpenCode inference
# endpoint; it can be replaced by any compatible endpoint without code changes.
PROVIDER = os.environ.get("LLM_PROVIDER", "openai_compatible")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://opencode.ai/inference/openai/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "public")
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash-free")
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1200"))
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "70"))
MAX_TOOL_STEPS = int(os.environ.get("LLM_MAX_TOOL_STEPS", "6"))
# If the real provider is unavailable (rate limit, network), the deterministic
# fallback still drives the same backend tools so the demo remains usable.
FALLBACK_MOCK = os.environ.get("LLM_FALLBACK_MOCK", "1").lower() not in {"0", "false", "no"}
SYSTEM_PROMPT = os.environ.get("LLM_SYSTEM_PROMPT", "")
