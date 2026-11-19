"""Ядро инференса: MediaPipe Hands + ROI по позе тела.

Пайплайн одного кадра:
    кадр → PoseLandmarker (lite) → запястья
         → квадратный ROI вокруг запястья (размер от ширины плеч / длины предплечья)
         → HandLandmarker на кропе → 21 точка в координатах исходного кадра
         → (опционально) полный кадр как fallback, слияние результатов
         → сглаживание EMA для видео

Почему так: MediaPipe всегда работает с квадратным ROI. При подаче полного кадра
детектор ладони видит кисть размером в единицы процентов кадра и теряет её,
поэтому мелкие кисти лучше искать в отдельном кропе (см. results/dataset_stats.md).

Режимы MediaPipe:
    VIDEO  — detect_for_video(ts) с трекингом между кадрами: файлы и вебка;
    IMAGE  — detect() без трекинга: оценка на фотографиях.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import HAND_EDGES, MODELS

POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16
POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW = 13, 14
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12

COLORS = {
    "left": (90, 200, 255),    # BGR: оранжево-жёлтый
    "right": (255, 170, 60),   # BGR: голубой
    "gt": (60, 210, 60),
}


@dataclass
class Hand:
    points: np.ndarray          # (21, 2) в пикселях исходного кадра
    handedness: str = "Unknown"  # "Left" / "Right"
    score: float = 0.0
    source: str = "roi"          # "roi" | "full"

    @property
    def wrist(self) -> np.ndarray:
        return self.points[0]


@dataclass
class EstimatorConfig:
    mode: str = "VIDEO"          # "VIDEO" | "IMAGE"
    num_hands: int = 2
    detection_confidence: float = 0.3
    presence_confidence: float = 0.3
    tracking_confidence: float = 0.4
    use_pose_roi: bool = True
    full_frame_fallback: bool = True
    pose_size: int = 512
    full_size: int = 960
    roi_size: int = 512
    roi_hand_ratio: float = 3.0   # кроп ≈ 3 × bbox кисти (калибровка на 120 кадрах)
    ensemble_full: bool = False   # + проход по полному кадру (точность вместо скорости)
    pose_every: int = 1           # пересчитывать позу раз в N кадров (realtime)
    merge_radius_frac: float = 0.02  # радиус склейки дублей, доля меньшей стороны кадра
    merge_radius_min: float = 10.0   # ... но не меньше, px
    wrist_visibility: float = 0.5


class HandPoseEstimator:
    """Обёртка над MediaPipe Tasks: поза тела → ROI → 21 точка кисти."""

    def __init__(self, cfg: EstimatorConfig | None = None,
                 hand_model: str | None = None,
                 pose_model: str | None = None):
        self.cfg = cfg or EstimatorConfig()
        hand_model = hand_model or str(MODELS / "hand_landmarker.task")
        pose_model = pose_model or str(MODELS / "pose_landmarker_lite.task")
        self._frame_idx = 0
        self._pose_calls = 0
        self._last_rois: list[tuple[float, float, float]] = []

        def base_options(path: str):
            return mp_python.BaseOptions(model_asset_path=path)

        run_mode = vision.RunningMode.IMAGE if self.cfg.mode.upper() == "IMAGE" else vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=base_options(hand_model),
                running_mode=run_mode,
                num_hands=self.cfg.num_hands,
                min_hand_detection_confidence=self.cfg.detection_confidence,
                min_hand_presence_confidence=self.cfg.presence_confidence,
                min_tracking_confidence=self.cfg.tracking_confidence,
            )
        )
        self.pose = None
        if self.cfg.use_pose_roi:
            # поза нужна раз в кадр и без трекинга — IMAGE-режим предсказуемее
            self.pose = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=base_options(pose_model),
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                )
            )

    # --- служебное ---------------------------------------------------------
    def _timestamp(self) -> int:
        """Монотонный ts в мс; на кадр резервируем 10 мс под несколько ROI."""
        self._frame_idx += 1
        return self._frame_idx * 100

    def _detect_hands(self, bgr: np.ndarray, ts: int) -> list[Hand]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.cfg.mode.upper() == "IMAGE":
            res = self.hands.detect(img)
        else:
            res = self.hands.detect_for_video(img, ts)
        h, w = bgr.shape[:2]
        out: list[Hand] = []
        for i, lm in enumerate(res.hand_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
            label = "Unknown"
            score = 0.0
            if i < len(res.handedness) and res.handedness[i]:
                cat = res.handedness[i][0]
                label, score = cat.category_name, float(cat.score)
            out.append(Hand(pts, label, score))
        return out

    @staticmethod
    def _letterbox(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
        """Квадратный canvas с сохранением пропорций: (canvas, scale, pad_x, pad_y)."""
        h, w = bgr.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((size, size, 3), np.uint8)
        px, py = (size - nw) // 2, (size - nh) // 2
        canvas[py:py + nh, px:px + nw] = resized
        return canvas, scale, px, py

    # --- ROI по позе -------------------------------------------------------
    def _pose_rois(self, bgr: np.ndarray) -> list[tuple[float, float, float]]:
        """Список (cx, cy, size) в пикселях исходного кадра.

        Калибровка по 120 тестовым кадрам: bbox кисти ≈ 0.45 · ширина плеч
        ≈ 0.72 · длина предплечья, а лучший кроп для MediaPipe ≈ 3 · bbox
        (см. results/pck_summary.md, свип по размеру кропа).
        """
        canvas, scale, px, py = self._letterbox(bgr, self.cfg.pose_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        res = self.pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.pose_landmarks:
            return []
        h, w = bgr.shape[:2]
        lm = res.pose_landmarks[0]

        def to_px(i: int):
            p = lm[i]
            vis = float(getattr(p, "visibility", 1.0) or 0.0)
            if vis < self.cfg.wrist_visibility:
                return None
            return np.array([(p.x * self.cfg.pose_size - px) / scale,
                             (p.y * self.cfg.pose_size - py) / scale], dtype=np.float32)

        sl, sr = to_px(POSE_LEFT_SHOULDER), to_px(POSE_RIGHT_SHOULDER)
        shoulder_w = float(np.linalg.norm(sl - sr)) if sl is not None and sr is not None else 0.0

        rois = []
        for wrist_i, elbow_i in ((POSE_LEFT_WRIST, POSE_LEFT_ELBOW),
                                 (POSE_RIGHT_WRIST, POSE_RIGHT_ELBOW)):
            wrist, elbow = to_px(wrist_i), to_px(elbow_i)
            if wrist is None:
                continue
            candidates = []
            if shoulder_w > 1.0:
                candidates.append(self.cfg.roi_hand_ratio / 0.45 * shoulder_w)
            if elbow is not None:
                candidates.append(self.cfg.roi_hand_ratio / 0.72 * float(np.linalg.norm(wrist - elbow)))
            size = float(np.median(candidates)) if candidates else 0.45 * min(w, h)
            size = float(np.clip(size, 0.10 * min(w, h), 0.95 * min(w, h)))
            center = wrist
            if elbow is not None:
                d = wrist - elbow
                n = float(np.linalg.norm(d))
                if n > 1e-6:
                    center = wrist + 0.12 * size * (d / n)  # сдвиг к пальцам, но не к краю кропа
            rois.append((float(center[0]), float(center[1]), size))
        return rois

    # --- основной вызов ----------------------------------------------------
    def estimate(self, bgr: np.ndarray, timestamp_ms: int | None = None) -> list[Hand]:
        ts = timestamp_ms if timestamp_ms is not None else self._timestamp()
        h, w = bgr.shape[:2]

        hands: list[Hand] = []
        used = 0

        if self.pose is not None:
            every = max(1, int(self.cfg.pose_every))
            if not self._last_rois or self._pose_calls % every == 0:
                self._last_rois = self._pose_rois(bgr)
            self._pose_calls += 1
            for cx, cy, size in self._last_rois:
                side_px = int(round(size))
                x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))
                x1, y1 = x0 + side_px, y0 + side_px
                # квадратный canvas с паддингом по границам кадра
                canvas = np.zeros((side_px, side_px, 3), np.uint8)
                x0c, y0c = max(0, x0), max(0, y0)
                x1c, y1c = min(w, x1), min(h, y1)
                if x1c - x0c < 16 or y1c - y0c < 16:
                    continue
                canvas[y0c - y0:y1c - y0, x0c - x0:x1c - x0] = bgr[y0c:y1c, x0c:x1c]
                k = self.cfg.roi_size / side_px
                canvas = cv2.resize(canvas, None, fx=k, fy=k,
                                    interpolation=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA)
                for hand in self._detect_hands(canvas, ts + used):
                    hand.points = hand.points / k + np.array([x0, y0], np.float32)
                    hand.source = "roi"
                    hands.append(hand)
                used += 1

        if (not hands and (self.cfg.full_frame_fallback or self.cfg.ensemble_full)) \
                or (hands and self.cfg.ensemble_full):
            hands += self._detect_full(bgr, ts + used)

        return self._merge(hands, image_wh=(w, h),
                           radius_frac=self.cfg.merge_radius_frac,
                           radius_min=self.cfg.merge_radius_min)

    def _detect_full(self, bgr: np.ndarray, ts: int) -> list[Hand]:
        """Проход по всему кадру (letterbox в квадрат `full_size`)."""
        canvas, scale, px, py = self._letterbox(bgr, self.cfg.full_size)
        out = []
        for hand in self._detect_hands(canvas, ts):
            hand.points = (hand.points - np.array([px, py], np.float32)) / scale
            hand.source = "full"
            out.append(hand)
        return out

    @staticmethod
    def _merge(hands: list[Hand], image_wh=None,
               radius_frac: float = 0.02, radius_min: float = 10.0) -> list[Hand]:
        """Убирает дубли: ROI и полный кадр могут найти одну и ту же кисть.

        Радиус склейки адаптивный (2 % меньшей стороны кадра, но не меньше 10 px):
        фиксированные 60 px на 4K-кадре склеивали бы разные кисти, а на 480p —
        не склеивали бы одну и ту же. Слишком большой радиус (≥5 %) заметно
        снижает пользу ансамбля: полнокадровый проход перестаёт добавлять
        альтернативную гипотезу запястья.
        """
        radius_px = radius_min
        if image_wh is not None:
            radius_px = max(radius_min, radius_frac * min(image_wh))
        merged: list[Hand] = []
        for hand in sorted(hands, key=lambda x: -x.score):
            if all(np.linalg.norm(hand.wrist - m.wrist) > radius_px for m in merged):
                merged.append(hand)
        return merged

    def close(self) -> None:
        self.hands.close()
        if self.pose is not None:
            self.pose.close()


# --- сглаживание и отрисовка ------------------------------------------------
@dataclass
class LandmarkSmoother:
    """EMA по точкам, отдельное состояние на каждую сторону кисти."""

    alpha: float = 0.5
    state: dict = field(default_factory=dict)

    def apply(self, hands: list[Hand]) -> list[Hand]:
        for hand in hands:
            prev = self.state.get(hand.handedness)
            if prev is not None and len(prev) == len(hand.points):
                hand.points = (self.alpha * hand.points + (1 - self.alpha) * prev).astype(np.float32)
            self.state[hand.handedness] = hand.points.copy()
        return hands


def draw_hands(bgr: np.ndarray, hands: list[Hand], labels: bool = True) -> np.ndarray:
    out = bgr.copy()
    for hand in hands:
        color = COLORS["left"] if hand.handedness == "Left" else COLORS["right"]
        pts = hand.points.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
        for i, p in enumerate(pts):
            r = 5 if i in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(out, tuple(p), r, color, -1, cv2.LINE_AA)
        if labels:
            x, y = pts[:, 0].min(), pts[:, 1].max() + 18
            text = f"{hand.handedness} {hand.score:.2f} [{hand.source}]"
            cv2.putText(out, text, (int(x), int(min(y, out.shape[0] - 6))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out