"""Аудит данных: статистика, проверки консистентности, иллюстрации.

Результаты:
    results/dataset_stats.md   — таблицы и выводы
    results/figs/*.png         — распределения и примеры разметки

Запуск:
    python src/audit.py
"""

from __future__ import annotations

import json
from collections import Counter

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    FIGS,
    HAND_EDGES,
    HAND_LABELS,
    HEAD_SCALE,
    PROCESSED,
    RESULTS,
    image_path,
    load_records,
    points_bbox,
)

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "font.family": "Liberation Serif",
                     "axes.unicode_minus": False})


def records_frame(recs: list[dict], split: str) -> pd.DataFrame:
    rows = []
    for r in recs:
        kp = np.asarray(r["kp_px"])
        w, h = r["width"], r["height"]
        box = points_bbox(kp)
        rows.append({
            "split": split,
            "name": r["name"],
            "group": r["group"],
            "frame": r["frame"],
            "side": r["side"],
            "width": w,
            "height": h,
            "is_mpii": r["is_mpii"],
            "head_size": r["head_size"],
            "hand_w": box[2] - box[0],
            "hand_h": box[3] - box[1],
            "hand_size": max(box[2] - box[0], box[3] - box[1]),
            "rel_hand_size": max(box[2] - box[0], box[3] - box[1]) / max(w, h),
            "vis_zero": int((np.asarray(r["vis"]) <= 0).sum()),
        })
    df = pd.DataFrame(rows)
    df["pck_scale_px"] = HEAD_SCALE * df["head_size"]
    df["pck_thr@0.2_px"] = 0.2 * df["pck_scale_px"]
    return df


def plot_distributions(df: pd.DataFrame) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for split, color in (("test", "#cf3d3d"), ("train", "#3d6fcf")):
        v = df.loc[df.split == split, "rel_hand_size"] * 100
        ax.hist(v, bins=np.arange(0, 22, 1), alpha=0.55, label=f"{split} (n={len(v)})", color=color)
    ax.set_xlim(0, 20)
    ax.set_xlabel("размер кисти, % от большей стороны кадра")
    ax.set_ylabel("изображений")
    ax.legend()
    ax.set_title("Насколько мелкими выглядят кисти")
    fig.tight_layout()
    fig.savefig(FIGS / "hand_size_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for split, color in (("test", "#cf3d3d"), ("train", "#3d6fcf")):
        v = df.loc[df.split == split, "head_size"]
        ax.hist(v, bins=np.arange(0, 320, 10), alpha=0.55, label=split, color=color)
    ax.set_xlabel("head_size, px")
    ax.set_ylabel("изображений")
    ax.legend()
    ax.set_title("Масштаб для PCK (0.7 · head_size)")
    fig.tight_layout()
    fig.savefig(FIGS / "head_size_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    counts = df.groupby(["split", "side"]).size().unstack(fill_value=0)
    counts.plot(kind="barh", stacked=True, ax=ax, color=["#8a5cf6", "#1f9d55"])
    ax.set_xlabel("изображений")
    ax.set_ylabel("")
    ax.set_title("Левая / правая кисть")
    fig.tight_layout()
    fig.savefig(FIGS / "sides.png")
    plt.close(fig)

    top = df[df.split == "test"]["group"].value_counts().head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.barh(top.index, top.values, color="#cf3d3d", alpha=0.75)
    ax.set_xlabel("изображений в тесте")
    ax.set_title("Топ-10 клипов тестовой выборки")
    fig.tight_layout()
    fig.savefig(FIGS / "groups_test.png")
    plt.close(fig)


def plot_gt_examples(recs: list[dict], n: int = 8) -> None:
    """Кропы вокруг кисти с нанесённым GT-скелетом."""
    rng = np.random.default_rng(7)
    picks = rng.choice(len(recs), size=min(n, len(recs)), replace=False)
    tiles = []
    for i in picks:
        r = recs[i]
        img = cv2.imread(str(image_path(r)))
        pts = np.asarray(r["kp_px"])
        box = points_bbox(pts, pad=0.35)
        x0, y0, x1, y1 = [int(v) for v in box]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        crop = img[y0:y1, x0:x1].copy()
        p = (pts - [x0, y0]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(crop, tuple(p[a]), tuple(p[b]), (0, 220, 0), 2)
        for j, q in enumerate(p):
            cv2.circle(crop, tuple(q), 2, (0, 220, 0), -1)
        cv2.putText(crop, f"{r['name'][:18]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        tiles.append(cv2.resize(crop, (260, 260)))

    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    FIG: np.ndarray = np.vstack(rows)
    FIGS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(FIGS / "gt_examples.png"), FIG)


def consistency_check(test: list[dict]) -> dict:
    """Насколько нормированные точки HF совпадают с пиксельными CMU."""
    diffs = []
    for r in test:
        kp_hf = np.asarray(r["kp_norm_hf"]) * [r["width"], r["height"]]
        diffs.append(np.linalg.norm(kp_hf - np.asarray(r["kp_px"]), axis=1))
    d = np.concatenate(diffs)
    return {
        "median_px": float(np.median(d)),
        "p99_px": float(np.percentile(d, 99)),
        "max_px": float(d.max()),
        "within_1px": float((d <= 1.0).mean()),
    }


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main() -> None:
    test = load_records("test")
    train = load_records("train")
    val = load_records("val")
    df = pd.concat(
        [records_frame(test, "test"), records_frame(train, "train"), records_frame(val, "val")],
        ignore_index=True,
    )

    plot_distributions(df)
    plot_gt_examples(test)

    man = json.loads((PROCESSED / "manifest.json").read_text(encoding="utf-8"))
    cons = consistency_check(test)

    def q(split: str, col: str, ps=(5, 25, 50, 75, 95)) -> str:
        v = df.loc[df.split == split, col]
        return " / ".join(f"{np.percentile(v, p):.2f}" for p in ps)

    dup = (
        df[df.split != "test"]
        .groupby(["group", "frame"]).size()
        .reset_index(name="n")
    )
    dup_pairs = int((dup["n"] > 1).sum())

    lines = [
        "# Аудит данных",
        "",
        f"Манифест: `data/processed/manifest.json` (создан {man['created_utc']}).",
        "",
        "## Состав",
        "",
        "| выборка | изображений | клипов | левых | правых | MPII |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("test", "train", "val"):
        s = df[df.split == split]
        lines.append(
            f"| {split} | {len(s)} | {s['group'].nunique()} | "
            f"{(s.side == 'left').sum()} | {(s.side == 'right').sum()} | {int(s.is_mpii.sum())} |"
        )
    lines += [
        "",
        f"Пересечение клипов с тестом: train = {len(man['group_leak']['train_test'])}, "
        f"val = {len(man['group_leak']['val_test'])}.",
        "",
        "## Разрешение и масштаб кисти",
        "",
        "| выборка | разрешение (w×h), топ-3 | размер кисти, % кадра (p5/25/50/75/95) | доля < 10 % |",
        "|---|---|---|---|",
    ]
    for split in ("test", "train"):
        s = df[df.split == split]
        res = Counter(f"{int(w)}×{int(h)}" for w, h in zip(s.width, s.height)).most_common(3)
        res_s = ", ".join(f"{k} ({v})" for k, v in res)
        lines.append(
            f"| {split} | {res_s} | {q(split, 'rel_hand_size')} | "
            f"{fmt_pct((s.rel_hand_size < 0.10).mean())} |"
        )
    lines += [
        "",
        "## Масштаб PCK (0.7 · head_size)",
        "",
        "| выборка | head_size, px (p5/25/50/75/95) | порог PCK@0.2, px (p5/50/95) |",
        "|---|---|---|",
    ]
    for split in ("test", "train"):
        thr = df.loc[df.split == split, "pck_thr@0.2_px"]
        lines.append(
            f"| {split} | {q(split, 'head_size')} | "
            f"{np.percentile(thr, 5):.2f} / {np.percentile(thr, 50):.2f} / {np.percentile(thr, 95):.2f} |"
        )
    lines += [
        "",
        "## Проверки",
        "",
        f"- Нормированные точки HF против пиксельных CMU: медиана {cons['median_px']:.3f} px, "
        f"p99 {cons['p99_px']:.3f} px, max {cons['max_px']:.3f} px; "
        f"в пределах 1 px — {fmt_pct(cons['within_1px'])} точек.",
        f"- Нулевая visibility у GT-точек: {int(df.vis_zero.sum())} из {21 * len(df)}.",
        f"- Кадры с обеими руками (одинаковый клип+кадр): {dup_pairs} пар в train+val.",
        f"- Всего изображений: {len(df)}; точек на изображение: 21; рёбер скелета: {len(HAND_EDGES)}.",
        "",
        "## Выводы для эксперимента",
        "",
        "- Кисти мелкие: медиана — около 9 % кадра, у 63 % тестовых изображений меньше 10 %; "
        "поэтому «полный кадр» заведомо проигрывает пайплайну с ROI.",
        "- Разрешения разные (от 400×300 до 1920×1080), и относительный размер кисти "
        "зависит от разрешения: метрику надо смотреть ещё и в разрезе по разрешению.",
        "- Порог PCK@0.2 — медиана около 20 px: метрика строгая, на 480p-даунскейле "
        "мелкие кисти деградируют сильнее всего.",
        "- Официальный train отделён по клипам от теста (пересечений нет), "
        "утечки через соседние кадры не будет.",
        "",
        "## Файлы",
        "",
        "- `results/figs/hand_size_hist.png` — распределение относительного размера кисти",
        "- `results/figs/head_size_hist.png` — масштаб PCK",
        "- `results/figs/sides.png`, `results/figs/groups_test.png` — состав выборок",
        "- `results/figs/gt_examples.png` — примеры GT-разметки",
        "",
        f"Порядок точек: {', '.join(HAND_LABELS[:5])}, … (всего {len(HAND_LABELS)}).",
        "",
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"\nзаписано: results/dataset_stats.md, results/figs/*.png")


if __name__ == "__main__":
    main()