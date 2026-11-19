"""PCK-оценка MediaPipe Hands на 846 тестовых изображениях.

Конфигурации (абляции):
    full      — полный кадр в исходном разрешении (letterbox в квадрат)
    full720   — полный кадр, даунскейл до 1280×720
    full480   — полный кадр, даунскейл до 640×480 (цена realtime-упрощения)
    oracle    — кроп по GT-bbox кисти (верхняя граница лендмаркера)
    pose      — ROI по запястью из PoseLandmarker (предлагаемый пайплайн)
    pose_full — pose + дополнительный проход по полному кадру (ансамбль)
    tiles2    — 2×2 тайла с перекрытием 10 %

Метрика (Simon et al., CVPR 2017): ошибка нормируется на 0.7 · head_size,
PCK@α = доля точек с нормированной ошибкой ≤ α. Пропуск детекции = +inf.

Запуск:
    python src/eval_mediapipe.py                        # все конфигурации
    python src/eval_mediapipe.py --configs pose,full --limit 100
    python src/eval_mediapipe.py --configs pose --vis 12   # + картинки GT/pred

Результаты:
    results/preds/<config>.jsonl   — ошибки по каждому изображению
    results/pck_eval.json          — сводка
    results/pck_summary.md         — таблица для отчёта
    results/figs/pck_*.png         — кривые и разрезы
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from common import (
    FIGS,
    FINGERS,
    HAND_EDGES,
    HAND_LABELS,
    MODELS,
    PCK_ALPHAS,
    RESULTS,
    curve_auc,
    hand_scale_px,
    image_path,
    kp_array,
    load_records,
    pck_at,
    pck_curve,
    points_bbox,
    valid_mask,
)
from pipeline import HandPoseEstimator, EstimatorConfig

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "font.family": "Liberation Serif",
                     "axes.unicode_minus": False})


# --- детекторы под каждую конфигурацию --------------------------------------
class Detectors:
    """Ленивое создание нужных инстансов MediaPipe (все в IMAGE-режиме)."""

    def __init__(self) -> None:
        self._lm = None
        self._estimators: dict[str, HandPoseEstimator] = {}

    @property
    def landmarker(self):
        if self._lm is None:
            self._lm = vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(MODELS / "hand_landmarker.task")),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                )
            )
        return self._lm

    def estimator(self, tag: str, **cfg) -> HandPoseEstimator:
        if tag not in self._estimators:
            self._estimators[tag] = HandPoseEstimator(EstimatorConfig(mode="IMAGE", **cfg))
        return self._estimators[tag]

    def close(self) -> None:
        if self._lm is not None:
            self._lm.close()
        for e in self._estimators.values():
            e.close()


def _square(img: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Паддинг до квадрата (без ресайза): canvas, scale=1, pad_x, pad_y."""
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), np.uint8)
    px, py = (s - w) // 2, (s - h) // 2
    canvas[py:py + h, px:px + w] = img
    return canvas, 1.0, px, py


def _hands_from(landmarker, bgr: np.ndarray) -> list[np.ndarray]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = bgr.shape[:2]
    return [np.array([[p.x * w, p.y * h] for p in lm], np.float32) for lm in res.hand_landmarks]


def _detect_full(det: Detectors, img: np.ndarray, size: int | None = None) -> list[np.ndarray]:
    k = 1.0
    if size is not None:
        h, w = img.shape[:2]
        k = size / max(h, w)
        img = cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))))
    canvas, _, px, py = _square(img)
    out = []
    for pts in _hands_from(det.landmarker, canvas):
        out.append((pts - np.array([px, py], np.float32)) / k)  # сначала ресайз, потом паддинг
    return out


def _detect_crop(det: Detectors, img: np.ndarray, box, out_size: int = 512) -> list[np.ndarray]:
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    canvas, _, px, py = _square(img[y0:y1, x0:x1])
    k = out_size / canvas.shape[0]
    canvas = cv2.resize(canvas, None, fx=k, fy=k)
    out = []
    for pts in _hands_from(det.landmarker, canvas):
        # сначала паддинг, потом ресайз: crop = pts / k - pad
        out.append(pts / k - np.array([px, py], np.float32) + np.array([x0, y0], np.float32))
    return out


def _detect_tiles(det: Detectors, img: np.ndarray, n: int = 2, overlap: float = 0.1) -> list[np.ndarray]:
    h, w = img.shape[:2]
    tw, th = int(w / (n - overlap * (n - 1))), int(h / (n - overlap * (n - 1)))
    step_x, step_y = tw - int(overlap * tw), th - int(overlap * th)
    out: list[np.ndarray] = []
    for i in range(n):
        for j in range(n):
            x0, y0 = j * step_x, i * step_y
            out += _detect_crop(det, img, (x0, y0, x0 + tw, y0 + th), out_size=512)
    return out


_POSE_CFG = dict(use_pose_roi=True, pose_size=512, roi_size=512,
                 full_size=960, roi_hand_ratio=3.0)


def _run_estimator(det: Detectors, tag: str, img: np.ndarray, **extra) -> list[np.ndarray]:
    est = det.estimator(tag, **{**_POSE_CFG, **extra})
    return [h.points for h in est.estimate(img)]


CONFIG_RUNNERS = {
    "full": lambda d, img, gt: _detect_full(d, img),
    "full720": lambda d, img, gt: _detect_full(d, img, size=1280),
    "full480": lambda d, img, gt: _detect_full(d, img, size=640),
    "oracle": lambda d, img, gt: _detect_crop(d, img, points_bbox(gt, pad=1.8)),  # 4.6× bbox
    "tiles2": lambda d, img, gt: _detect_tiles(d, img, n=2),
    # предлагаемый пайплайн: поза → ROI вокруг запястья → кисть в кропе
    "pose": lambda d, img, gt: _run_estimator(d, "pose", img),
    # он же + страховочный проход по полному кадру (точность вместо скорости)
    "pose_full": lambda d, img, gt: _run_estimator(d, "pose_full", img, ensemble_full=True),
}


# --- оценка ------------------------------------------------------------------
@dataclass
class Row:
    name: str
    group: str
    side: str
    rel_hand_size: float
    scale_px: float
    errors_norm: np.ndarray
    errors_px: np.ndarray
    dt_ms: float
    n_pred: int
    matched: bool


def evaluate_config(det: Detectors, config: str, records: list[dict], vis: int = 0) -> list[Row]:
    runner = CONFIG_RUNNERS[config]
    rows: list[Row] = []
    vis_picks: list[tuple[dict, list[np.ndarray]]] = []

    for i, rec in enumerate(records):
        img = cv2.imread(str(image_path(rec)))
        if img is None:
            raise RuntimeError(f"не читается изображение: {rec['image']}")
        gt = kp_array(rec)
        scale = hand_scale_px(rec)
        mask = valid_mask(rec)

        t0 = time.perf_counter()
        preds = runner(det, img, gt)
        dt = (time.perf_counter() - t0) * 1000

        if preds:
            best = min(preds, key=lambda p: float(np.linalg.norm(p[0] - gt[0])))
            err_px = np.linalg.norm(best - gt, axis=1)
        else:
            err_px = np.full(len(gt), np.inf, np.float32)
        err_norm = np.where(mask, err_px / scale, np.nan)

        rows.append(Row(
            name=rec["name"], group=rec["group"], side=rec["side"],
            rel_hand_size=float(max(np.ptp(gt[:, 0]), np.ptp(gt[:, 1])) / max(rec["width"], rec["height"])),
            scale_px=scale, errors_norm=err_norm, errors_px=err_px, dt_ms=dt,
            n_pred=len(preds), matched=bool(preds),
        ))
        if vis and len(vis_picks) < vis:
            vis_picks.append((rec, preds))

        if (i + 1) % 100 == 0:
            print(f"    {config}: {i + 1}/{len(records)}")

    if vis_picks:
        render_pairs(vis_picks, config)
    return rows


def stack_errors(rows: list[Row]) -> np.ndarray:
    """(N, 21): NaN — пропущенные/невалидные точки, inf — промах по детекции."""
    return np.stack([r.errors_norm for r in rows])


def summarize(rows: list[Row], config: str) -> dict:
    e = stack_errors(rows)
    dt = np.array([r.dt_ms for r in rows])
    finite = [r.errors_px[np.isfinite(r.errors_px)] for r in rows]
    flat = np.concatenate(finite) if any(len(v) for v in finite) else np.array([np.nan])
    out = {
        "config": config,
        "images": len(rows),
        "detection_rate": float(np.mean([r.matched for r in rows])),
        "pck@0.1": pck_at(e, 0.1),
        "pck@0.2": pck_at(e, 0.2),
        "auc": curve_auc(PCK_ALPHAS, pck_curve(e)),
        "epe_median_px": float(np.median(flat)),
        "latency_p50_ms": float(np.median(dt)),
        "latency_p90_ms": float(np.percentile(dt, 90)),
        "extra_detections": float(np.mean([max(0, r.n_pred - 1) for r in rows])),
    }
    by_size = {}
    for label, sel in (("small<5%", [r for r in rows if r.rel_hand_size < 0.05]),
                       ("mid 5-10%", [r for r in rows if 0.05 <= r.rel_hand_size < 0.10]),
                       ("large>=10%", [r for r in rows if r.rel_hand_size >= 0.10])):
        if sel:
            by_size[label] = pck_at(stack_errors(sel), 0.2)
    out["pck@0.2_by_hand_size"] = by_size
    return out


# --- визуализация ------------------------------------------------------------
def render_pairs(picks: list[tuple[dict, list[np.ndarray]]], config: str) -> None:
    tiles = []
    for rec, preds in picks:
        img = cv2.imread(str(image_path(rec)))
        gt = kp_array(rec)
        best = min(preds, key=lambda p: float(np.linalg.norm(p[0] - gt[0]))) if preds else None
        box = points_bbox(gt, pad=0.45)
        x0, y0, x1, y1 = [int(v) for v in box]
        h, w = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        crop = img[y0:y1, x0:x1].copy()
        off = np.array([x0, y0], np.float32)
        p = (gt - off).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(crop, tuple(p[a]), tuple(p[b]), (60, 210, 60), 2)
        if best is not None:
            q = (best - off).astype(int)
            for a, b in HAND_EDGES:
                cv2.line(crop, tuple(q[a]), tuple(q[b]), (60, 60, 235), 2)
        err = np.linalg.norm(best - gt, axis=1) / hand_scale_px(rec) if best is not None else None
        title = f"PCK@.2={np.mean(err <= 0.2):.2f}" if err is not None else "miss"
        tile = cv2.resize(crop, (300, 300))
        cv2.rectangle(tile, (0, 0), (118, 20), (0, 0, 0), -1)
        cv2.putText(tile, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    FIGS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGS / f"vis_{config}.png"), np.vstack(rows))


def plot_curves(all_errors: dict[str, np.ndarray]) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for cfg, e in all_errors.items():
        ax.plot(PCK_ALPHAS, pck_curve(e), marker="o", ms=2.5, lw=1.4, label=cfg)
    ax.axvline(0.2, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("α (ошибка / (0.7 · head_size))")
    ax.set_ylabel("PCK")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("PCK-кривые MediaPipe по конфигурациям")
    fig.tight_layout()
    fig.savefig(FIGS / "pck_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    names = [n.replace("_MCP", "").replace("_PIP", "").replace("_DIP", "").replace("_TIP", "T")
             .replace("THUMB", "TH").replace("INDEX", "I").replace("MIDDLE", "M")
             .replace("RING", "R").replace("PINKY", "P") for n in HAND_LABELS]
    for cfg, e in all_errors.items():
        vals = [pck_at(e[:, j], 0.2) for j in range(len(HAND_LABELS))]
        ax.plot(vals, marker="o", ms=3, lw=1.2, label=cfg)
    ax.set_xticks(range(len(HAND_LABELS)), names, rotation=60, fontsize=6)
    ax.set_ylabel("PCK@0.2")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.set_title("Точность по суставам")
    fig.tight_layout()
    fig.savefig(FIGS / "pck_per_joint.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    for cfg, e in all_errors.items():
        v = np.sort(e[np.isfinite(e)])
        ax.plot(v, np.linspace(0, 1, len(v)), lw=1.4, label=cfg)
    ax.axvline(0.2, color="gray", ls="--", lw=0.8)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("нормированная ошибка точки")
    ax.set_ylabel("доля точек")
    ax.legend(fontsize=7)
    ax.set_title("CDF ошибок")
    fig.tight_layout()
    fig.savefig(FIGS / "error_cdf.png")
    plt.close(fig)


def write_summary(summaries: list[dict], all_errors: dict[str, np.ndarray]) -> None:
    n_images = summaries[0]["images"] if summaries else 0
    lines = [
        f"# PCK на тестовой выборке ({n_images} изображений, CMU Panoptic Hand DB)",
        "",
        "Нормировка: `0.7 · head_size` (Simon et al., CVPR 2017). Пропуск = промах по всем 21 точке.",
        "EPE med считается только по найденным точкам; промахи видны в колонке detection.",
        "",
        "| конфигурация | PCK@0.1 | PCK@0.2 | AUC | EPE med, px | detection | задержка p50, мс | лишних кистей |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(summaries, key=lambda x: -x["pck@0.2"]):
        lines.append(
            f"| {s['config']} | {s['pck@0.1']:.3f} | {s['pck@0.2']:.3f} | {s['auc']:.3f} | "
            f"{s['epe_median_px']:.1f} | {s['detection_rate']:.1%} | {s['latency_p50_ms']:.0f} | "
            f"{s['extra_detections']:.2f} |"
        )
    lines += ["", "## PCK@0.2 по размеру кисти (доля кадра)", ""]
    labels = sorted({k for s in summaries for k in s["pck@0.2_by_hand_size"]})
    lines.append("| конфигурация | " + " | ".join(labels) + " |")
    lines.append("|---" * (len(labels) + 1) + "|")
    for s in summaries:
        cells = [f"{s['pck@0.2_by_hand_size'][k]:.3f}" if k in s["pck@0.2_by_hand_size"] else "—"
                 for k in labels]
        lines.append(f"| {s['config']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Разрезы",
        "",
        "- `results/figs/pck_curve.png` — PCK(α)",
        "- `results/figs/pck_per_joint.png` — точность по суставам",
        "- `results/figs/error_cdf.png` — CDF нормированной ошибки",
        "- `results/figs/vis_<config>.png` — примеры GT (зелёный) и предсказания (красный)",
        "- `results/preds/<config>.jsonl` — ошибки по каждому изображению",
        "",
    ]
    (RESULTS / "pck_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIG_RUNNERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="оценить только первые N изображений")
    ap.add_argument("--vis", type=int, default=0, help="сколько примеров отрисовать на конфигурацию")
    ap.add_argument("--summarize-only", action="store_true",
                    help="не считать заново, только пересобрать сводку по results/preds")
    args = ap.parse_args()

    records = load_records(args.split)
    if args.limit:
        records = records[: args.limit]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(configs) - set(CONFIG_RUNNERS)
    if unknown:
        raise SystemExit(f"неизвестные конфигурации: {sorted(unknown)}")

    eval_path = RESULTS / "pck_eval.json"
    summaries: dict[str, dict] = {}
    if eval_path.exists():
        for s in json.loads(eval_path.read_text(encoding="utf-8")):
            summaries[s["config"]] = s

    if not args.summarize_only:
        det = Detectors()
        try:
            for cfg in configs:
                print(f"  → {cfg} ({len(records)} изображений)")
                rows = evaluate_config(det, cfg, records, vis=args.vis)
                summaries[cfg] = summarize(rows, cfg)
                RESULTS.mkdir(parents=True, exist_ok=True)
                (RESULTS / "preds").mkdir(exist_ok=True)
                with open(RESULTS / "preds" / f"{cfg}.jsonl", "w", encoding="utf-8") as f:
                    for r in rows:
                        f.write(json.dumps({
                            "name": r.name, "group": r.group, "side": r.side,
                            "rel_hand_size": round(r.rel_hand_size, 5),
                            "scale_px": round(r.scale_px, 3), "dt_ms": round(r.dt_ms, 1),
                            "n_pred": r.n_pred,
                            "errors_norm": [None if not np.isfinite(v) else round(float(v), 5)
                                            for v in r.errors_norm],
                            "errors_px": [None if not np.isfinite(v) else round(float(v), 2)
                                          for v in r.errors_px],
                        }) + "\n")
                print(f"    PCK@0.2 = {summaries[cfg]['pck@0.2']:.3f}, "
                      f"p50 {summaries[cfg]['latency_p50_ms']:.0f} ms")
        finally:
            det.close()

    # все ошибки — из сохранённых предсказаний (включая конфигурации прошлых прогонов)
    all_errors: dict[str, np.ndarray] = {}
    for p in sorted((RESULTS / "preds").glob("*.jsonl")):
        rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        all_errors[p.stem] = np.array(
            [[np.inf if v is None else v for v in r["errors_norm"]] for r in rows], dtype=np.float64)

    eval_path.write_text(
        json.dumps(list(summaries.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    plot_curves(all_errors)
    write_summary(list(summaries.values()), all_errors)


if __name__ == "__main__":
    main()