"""Пост-рендер видео: детекция кистей и запись результата в mp4.

Читает видеофайл, прогоняет кадр через `HandPoseEstimator`, рисует скелет
и пишет H.264 (через ffmpeg; если его нет — mp4v средствами OpenCV).

Примеры:
    python src/infer_video.py video.mp4                       # полное разрешение
    python src/infer_video.py video.mp4 --width 960 -o out.mp4
    python src/infer_video.py video.mp4 --pose-every 3        # быстрее
    python src/infer_video.py video.mp4 --no-pose             # только полный кадр
    python src/infer_video.py video.mp4 --ensemble            # pose + полный кадр

По умолчанию результат: results/renders/<имя>_hands.mp4
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import RESULTS
from pipeline import EstimatorConfig, HandPoseEstimator, LandmarkSmoother, draw_hands

RENDERS = RESULTS / "renders"


# --- запись -----------------------------------------------------------------
class FFmpegWriter:
    """Потоковая запись кадров в H.264 через ffmpeg (+ звук исходника)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 source: Path | None = None, crf: int = 20) -> None:
        self.path = path
        self.size = size
        path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
            "-i", "-",
        ]
        if source is not None:
            cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg завершился с кодом {code}")


class CvWriter:
    """Запасной вариант, если ffmpeg недоступен (кодек mp4v)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float,
                 source: Path | None = None, crf: int = 20) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self.writer.isOpened():
            raise RuntimeError(f"не удалось открыть VideoWriter: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def open_writer(path: Path, size, fps, source: Path | None, crf: int):
    if shutil.which("ffmpeg"):
        return FFmpegWriter(path, size, fps, source, crf)
    print("[!] ffmpeg не найден — пишу mp4v (совместимость хуже)")
    return CvWriter(path, size, fps, source, crf)


# --- утилиты ----------------------------------------------------------------
def fit_width(frame: np.ndarray, width: int | None) -> np.ndarray:
    if not width or width >= frame.shape[1]:
        return frame
    k = width / frame.shape[1]
    return cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)


def put_hud(frame: np.ndarray, lines: list[str]) -> None:
    y = 26
    for text in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    ap.add_argument("--width", type=int, default=0, help="ширина кадра (0 — как в исходнике)")
    ap.add_argument("--stride", type=int, default=1,
                    help="обрабатывать каждый N-й кадр (видео ускоряется в N раз)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-pose", action="store_true", help="без PoseLandmarker, только полный кадр")
    ap.add_argument("--ensemble", action="store_true", help="pose + дополнительный проход по кадру")
    ap.add_argument("--pose-every", type=int, default=1, help="пересчёт позы раз в N кадров")
    ap.add_argument("--roi-size", type=int, default=512, help="размер кропа кисти")
    ap.add_argument("--pose-size", type=int, default=512, help="размер входа позы")
    ap.add_argument("--full-size", type=int, default=960, help="размер полного кадра")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA-сглаживание (0 — выкл.)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--dump", type=Path, default=None,
                    help="сохранить точки по кадрам в JSONL")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"нет файла: {args.input}")
        return 1

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"не открывается видео: {args.input}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out_path = args.output or (RENDERS / f"{args.input.stem}_hands.mp4")

    cfg = EstimatorConfig(
        mode="VIDEO",
        use_pose_roi=not args.no_pose,
        pose_every=args.pose_every,
        ensemble_full=args.ensemble,
        pose_size=args.pose_size,
        roi_size=args.roi_size,
        full_size=args.full_size,
    )
    est = HandPoseEstimator(cfg)
    smoother = LandmarkSmoother(alpha=args.smooth) if args.smooth > 0 else None

    dump = None
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        dump = open(args.dump, "w", encoding="utf-8")
    writer = None
    frames_done = 0
    t_start = time.perf_counter()
    t_infer = 0.0
    last_hands = []

    pbar = tqdm(total=total or None, unit="кадр", desc="render")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and frames_done >= args.max_frames:
                break
            frames_done += 1
            if args.stride > 1 and (frames_done - 1) % args.stride != 0:
                continue

            frame = fit_width(frame, args.width)
            ts = int(round((frames_done - 1) * 1000.0 / fps))
            t0 = time.perf_counter()
            hands = est.estimate(frame, timestamp_ms=ts)
            t_infer += time.perf_counter() - t0
            if smoother is not None:
                hands = smoother.apply(hands)
            last_hands = hands

            vis = draw_hands(frame, hands)
            proc_fps = (frames_done) / max(1e-9, time.perf_counter() - t_start)
            put_hud(vis, [
                f"hands: {len(hands)}  |  {proc_fps:.1f} fps (render)",
                f"pose: {'off' if args.no_pose else f'every {args.pose_every}'}"
                f"{'  + full' if args.ensemble else ''}  |  smooth {args.smooth}",
            ])

            if writer is None:
                h, w = vis.shape[:2]
                writer = open_writer(out_path, (w, h), fps / max(1, args.stride),
                                     source=args.input, crf=args.crf)
            writer.write(vis)

            if dump is not None:
                dump.write(json.dumps({
                    "frame": frames_done - 1,
                    "t_ms": ts,
                    "hands": [{
                        "handedness": h.handedness,
                        "score": round(h.score, 4),
                        "source": h.source,
                        "points": [[round(float(x), 2), round(float(y), 2)] for x, y in h.points],
                    } for h in hands],
                }) + "\n")

            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        est.close()
        if writer is not None:
            writer.close()
        if dump is not None:
            dump.close()

    wall = time.perf_counter() - t_start
    print(f"\nкадров: {frames_done}")
    print(f"инференс: {t_infer / max(1, frames_done) * 1000:.0f} мс/кадр "
          f"({frames_done / max(1e-9, t_infer):.1f} fps)")
    print(f"всего: {wall:.1f} c, {frames_done / max(1e-9, wall):.2f} fps")
    print(f"записано: {out_path}")
    if dump is not None:
        print(f"точки: {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())