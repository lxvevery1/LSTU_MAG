"""Общие утилиты: пути, аннотации, топология кисти, метрика PCK.

Протокол PCK (Simon et al., CVPR 2017, «Hand Keypoint Detection in Single
Images using Multiview Bootstrapping»):
    расстояние до точки нормируется на 0.7 * head_size (примерная длина
    вытянутой кисти), PCK@alpha = доля точек с ошибкой <= alpha * (0.7 * head_size).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figs"
MODELS = ROOT / "models"

# порядок точек совпадает и с CMU, и с MediaPipe Hands (21 точка)
HAND_LABELS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (2, 5), (5, 9), (9, 13), (13, 17),
]
FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

PCK_ALPHAS = np.round(np.arange(0.02, 0.51, 0.02), 2)
HEAD_SCALE = 0.7  # из статьи: 0.7 * head_size ~= длина вытянутой кисти


# --- записи манифеста -------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_records(split: str) -> list[dict]:
    """Читает data/processed/<split>.jsonl и переводит координаты в numpy."""
    recs = read_jsonl(PROCESSED / f"{split}.jsonl")
    for r in recs:
        for key in ("kp_px", "kp_norm", "vis", "body_norm"):
            if key in r and r[key] is not None:
                r[key] = np.asarray(r[key], dtype=np.float32)
    return recs


def image_path(rec: dict) -> Path:
    return ROOT / rec["image"]


def kp_array(rec: dict) -> np.ndarray:
    """(21, 2) пиксельные координаты GT с учётом реального размера изображения."""
    return np.asarray(rec["kp_px"], dtype=np.float32)


# --- масштаб для PCK --------------------------------------------------------
def hand_scale_px(rec: dict) -> float:
    """Нормировка ошибки: 0.7 * head_size (px).

    Если head_size отсутствует или некорректен, откатываемся на максимальную
    сторону bbox по GT-точкам — это фиксируется в отчёте как fallback.
    """
    head = float(rec.get("head_size") or 0.0)
    if head > 1.0:
        return HEAD_SCALE * head
    pts = kp_array(rec)
    return float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))


def point_errors_px(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(21,) евклидовы ошибки в пикселях."""
    return np.linalg.norm(np.asarray(pred, np.float32) - np.asarray(gt, np.float32), axis=-1)


def valid_mask(rec: dict) -> np.ndarray:
    """bool-маска точек с валидной разметкой (visibility > 0)."""
    vis = rec.get("vis")
    if vis is None:
        return np.ones(len(HAND_LABELS), dtype=bool)
    return np.asarray(vis, dtype=np.float32) > 0


# --- PCK --------------------------------------------------------------------
def pck_curve(errors_norm: np.ndarray, alphas: np.ndarray = PCK_ALPHAS) -> np.ndarray:
    """Доля точек с ошибкой <= alpha для каждого alpha.

    errors_norm: (N, 21) ошибки, уже поделённые на 0.7*head_size.
    NaN — невалидная GT-точка (исключается), inf — пропущенная детекция (идёт в 0).
    """
    errors_norm = np.asarray(errors_norm, dtype=np.float64)
    ok = np.isfinite(errors_norm) | np.isinf(errors_norm)
    if not ok.any():
        return np.zeros(len(alphas))
    return np.array([np.mean(errors_norm[ok] <= a) for a in alphas])


def pck_at(errors_norm: np.ndarray, alpha: float) -> float:
    e = np.asarray(errors_norm, dtype=np.float64)
    ok = np.isfinite(e) | np.isinf(e)
    return float(np.mean(e[ok] <= alpha)) if ok.any() else 0.0


def curve_auc(alphas: np.ndarray, values: np.ndarray) -> float:
    """Площадь под PCK-кривой (нормирована на длину интервала)."""
    return float(np.trapezoid(values, alphas) / (alphas[-1] - alphas[0]))


def summarize_errors(errors_px: np.ndarray, scale_px: float) -> dict:
    """Набор чисел для таблиц: PCK@0.1/0.2, EPE, доля промахов."""
    e = np.asarray(errors_px, dtype=np.float64)
    finite = np.isfinite(e)
    return {
        "pck@0.1": pck_at(e / scale_px, 0.1),
        "pck@0.2": pck_at(e / scale_px, 0.2),
        "epe_median_px": float(np.median(e[finite])) if finite.any() else float("nan"),
        "epe_p90_px": float(np.percentile(e[finite], 90)) if finite.any() else float("nan"),
        "miss_rate": float((~finite).mean()),
    }


# --- геометрия --------------------------------------------------------------
def square_roi(center_xy, size: float, image_wh) -> tuple[int, int, int, int]:
    """Квадратный ROI (x0, y0, x1, y1), обрезанный по границам изображения."""
    w, h = image_wh
    cx, cy = center_xy
    half = size / 2.0
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    return max(0, x0), max(0, y0), min(int(w), x1), min(int(h), y1)


def points_bbox(pts: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """xyxy по точкам с относительным паддингом от размера bbox."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    s = max(x1 - x0, y1 - y0) * pad
    return np.array([x0 - s, y0 - s, x1 + s, y1 + s], dtype=np.float32)