"""Realtime-демо с вебкамеры.

Окно: скелет кисти поверх видео, FPS, источник предсказания (roi/full).
Клавиши: q / Esc — выход, s — скриншот, p — пауза, f — вкл/выкл сглаживание,
         r — начать/остановить запись.

Для скорости поза пересчитывается не каждый кадр (`--pose-every`), а кадр
уменьшается до 480p. На CPU без GPU это ~5–10 FPS; на машине с GPU — плавно.

Примеры:
    python src/demo_live.py                    # камера 0, 640×480
    python src/demo_live.py --camera 1 --width 960 --height 540
    python src/demo_live.py --fast             # pose раз в 5 кадров, ROI 256
    python src/demo_live.py --record demo.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline import EstimatorConfig, HandPoseEstimator, LandmarkSmoother, draw_hands
from infer_video import open_writer, put_hud


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="индекс камеры или путь к видео")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--pose-every", type=int, default=3)
    ap.add_argument("--roi-size", type=int, default=384)
    ap.add_argument("--pose-size", type=int, default=256)
    ap.add_argument("--full-size", type=int, default=640)
    ap.add_argument("--no-pose", action="store_true", help="только полный кадр")
    ap.add_argument("--ensemble", action="store_true", help="pose + полный кадр")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--fast", action="store_true", help="пресет: pose_every=5, roi 256")
    ap.add_argument("--record", type=Path, default=None, help="писать mp4 (клавиша r)")
    args = ap.parse_args()

    if args.fast:
        args.pose_every, args.roi_size = 5, 256
        args.pose_size, args.full_size = 224, 512

    src = int(args.camera) if str(args.camera).isdigit() else args.camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"не открывается источник: {args.camera}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if hasattr(cv2, "CAP_PROP_FOURCC"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    ok, frame = cap.read()
    if not ok:
        print("камера не отдаёт кадры")
        cap.release()
        return 1
    real_w, real_h = frame.shape[1], frame.shape[0]
    print(f"источник: {args.camera}  {real_w}×{real_h}")

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
    smoother = LandmarkSmoother(alpha=args.smooth)
    smooth_on = args.smooth > 0

    writer = None
    recording = False
    fps_smooth = 0.0
    t_prev = time.perf_counter()
    idx = 0
    paused = False
    shot = 0
    hands = []

    print("q/Esc — выход, s — скриншот, p — пауза, f — сглаживание, "
          "r — запись" + (f" → {args.record}" if args.record else " (нужен --record)"))

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("поток кадров закончился")
                    break
                idx += 1
                t0 = time.perf_counter()
                hands = est.estimate(frame, timestamp_ms=int(idx * 1000 / 30))
                if smooth_on:
                    hands = smoother.apply(hands)
                dt = time.perf_counter() - t0
                now = time.perf_counter()
                inst = 1.0 / max(1e-6, now - t_prev)
                t_prev = now
                fps_smooth = inst if fps_smooth == 0 else 0.85 * fps_smooth + 0.15 * inst

                vis = draw_hands(frame, hands)
                put_hud(vis, [
                    f"{fps_smooth:4.1f} fps (inference {dt * 1000:.0f} ms)",
                    f"hands {len(hands)}  |  pose "
                    f"{'off' if args.no_pose else f'1/{args.pose_every}'}"
                    f"{' + full' if args.ensemble else ''}  |  smooth {'on' if smooth_on else 'off'}",
                ])
                if recording and writer is not None:
                    writer.write(vis)
                cv2.imshow("hand keypoints — demo", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                shot += 1
                path = Path(f"screenshot_{shot:02d}.png")
                cv2.imwrite(str(path), vis)
                print(f"скриншот: {path}")
            elif key == ord("p"):
                paused = not paused
                print("пауза" if paused else "продолжаю")
            elif key == ord("f"):
                smooth_on = not smooth_on
                if smooth_on:
                    smoother.state.clear()
                print(f"сглаживание: {'вкл' if smooth_on else 'выкл'}")
            elif key == ord("r") and args.record is not None:
                if not recording:
                    h, w = frame.shape[:2]
                    writer = open_writer(args.record, (w, h), 30.0)
                    recording = True
                    print(f"запись: {args.record}")
                else:
                    recording = False
                    if writer is not None:
                        writer.close()
                        writer = None
                    print("запись остановлена")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        est.close()
        if writer is not None:
            writer.close()
    print("готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())