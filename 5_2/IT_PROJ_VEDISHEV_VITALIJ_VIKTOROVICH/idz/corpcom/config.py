"""Конфигурация CorpCom — локальный сервер в корпоративной сети."""

import os

HOST = os.environ.get("CORPCOM_HOST", "0.0.0.0")
PORT = int(os.environ.get("CORPCOM_PORT", "8765"))
AUTH_TOKEN = os.environ.get("CORPCOM_TOKEN", "corp-secret-change-me")

# Голос: PCM 16-bit mono
SAMPLE_RATE = 16_000
CHANNELS = 1
VOICE_CHUNK_MS = 20  # меньше задержка для голоса в реальном времени
VOICE_CHUNK_SAMPLES = SAMPLE_RATE * VOICE_CHUNK_MS // 1000

DEFAULT_ROOM = "general"
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MiB
