#!/usr/bin/env python3
"""Shared helpers for the WSB direction project.

All paths in configs are resolved relative to the project directory
(``stocks_predicts/``), not relative to the current working directory.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLASSES = ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
DIRECTIONS = ["DOWN", "FLAT", "UP"]
DIRECTION_TO_ID = {d: i for i, d in enumerate(DIRECTIONS)}
ID_TO_DIRECTION = {i: d for d, i in DIRECTION_TO_ID.items()}

# TZ section 1.1 / 3.5
INFLUENCE_WEIGHTS = {
    "ANALYSIS_DD": 1.0,
    "FINANCIAL_CLAIM": 0.6,
    "GENUINE_OPINION": 0.5,
    "ADVICE": 0.5,
    "NOISE_MEME": 0.1,
    "UNCERTAIN": 0.0,
}


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


def load_config(path: str | Path = "configs/data.yaml") -> dict:
    p = project_path(path)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def get_path(cfg: dict, key: str, default: str | None = None) -> Path:
    value = cfg.get(key, default)
    if value is None:
        raise KeyError(f"missing path {key!r} in config")
    return project_path(value)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def read_json(path: str | Path) -> Any:
    with open(project_path(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int | None = 2) -> None:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def read_jsonl(path: str | Path) -> list[dict]:
    p = project_path(path)
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def append_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str, prefix: str = "") -> str:
    return "sha256:" + hashlib.sha256((prefix + text).encode("utf-8")).hexdigest()


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9$ ]+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _NON_WORD_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def post_text(post: dict) -> str:
    title = (post.get("title") or "").strip()
    body = (post.get("body") or post.get("selftext") or "").strip()
    if body.lower() in ("[deleted]", "[removed]"):
        body = ""
    return (title + "\n\n" + body).strip() if body else title


def clean_body(value: str | None) -> str:
    value = (value or "").strip()
    if value.lower() in ("[deleted]", "[removed]", "[removed by moderator]"):
        return ""
    return value


def timestamp_to_utc(ts: int | float) -> datetime:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def year_of(ts: int | float) -> int:
    return timestamp_to_utc(ts).year


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b and not math.isclose(float(b), 0.0) else default


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stable_hash(text: str, seed: int = 0) -> int:
    h = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def data_version(cfg: dict) -> str:
    return cfg.get("data_version", "wsb16k-v1")


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        project_path(p).mkdir(parents=True, exist_ok=True)


def class_probs_from_row(row: pd.Series) -> list[float]:
    return [float(row.get(f"prob_{c}", 0.0) or 0.0) for c in CLASSES]


def influence_weight(label: str) -> float:
    return float(INFLUENCE_WEIGHTS.get(label, 0.0))
