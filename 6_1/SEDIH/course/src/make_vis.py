"""Галереи предсказаний для отчёта: GT (зелёный) против предсказания (красный).

Пример:
    python src/make_vis.py                 # vis_pose.png и vis_full.png, 8 примеров
    python src/make_vis.py --n 12 --seed 1
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from common import ROOT, load_records
from eval_mediapipe import CONFIG_RUNNERS, Detectors, render_pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--configs", default="pose,full")
    args = ap.parse_args()

    recs = load_records("test")
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(recs), size=min(args.n, len(recs)), replace=False)
    subset = [recs[i] for i in picks]

    det = Detectors()
    try:
        for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
            runner = CONFIG_RUNNERS[cfg]
            pairs = []
            for rec in subset:
                img = cv2.imread(str(ROOT / rec["image"]))
                preds = runner(det, img, np.asarray(rec["kp_px"], dtype=np.float32))
                pairs.append((rec, preds))
            render_pairs(pairs, cfg)
            print(f"записано: results/figs/vis_{cfg}.png")
    finally:
        det.close()


if __name__ == "__main__":
    main()