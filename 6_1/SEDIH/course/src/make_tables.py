"""Генерация LaTeX-таблиц для отчёта из результатов экспериментов.

Читает `results/pck_eval.json` и `results/preds/*.jsonl`, пишет
`report/tables/*.tex` (подключаются через \\input).

Запуск:
    python src/make_tables.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import RESULTS, hand_scale_px, load_records

TABLES = RESULTS.parent / "report" / "tables"

# человекочитаемые названия конфигураций
NAMES = {
    "full": "полный кадр",
    "full720": "полный кадр, 720p",
    "full480": "полный кадр, 480p",
    "oracle": "кроп по GT-bbox",
    "tiles2": "2×2 тайла",
    "pose": "поза → ROI (наш)",
    "pose_full": "поза → ROI + кадр",
}
ORDER = ["full", "full720", "full480", "tiles2", "pose", "pose_full", "oracle"]


def tex_name(cfg: str) -> str:
    return NAMES.get(cfg, cfg.replace("_", r"\_"))


def read_errors(cfg: str) -> np.ndarray | None:
    path = RESULTS / "preds" / f"{cfg}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return np.array([[np.inf if v is None else v for v in r["errors_norm"]] for r in rows],
                    dtype=np.float64)


def pck(e: np.ndarray, alpha: float) -> float:
    ok = np.isfinite(e) | np.isinf(e)
    return float(np.mean(e[ok] <= alpha)) if ok.any() else 0.0


def write_main(summaries: list[dict]) -> Path:
    by_cfg = {s["config"]: s for s in summaries}
    cfgs = [c for c in ORDER if c in by_cfg] + sorted(set(by_cfg) - set(ORDER))
    lines = [
        "% автогенерация: python src/make_tables.py",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"конфигурация & PCK@0.1 & PCK@0.2 & AUC & детекция & p50, мс & лишних кистей \\",
        r"\midrule",
    ]
    for c in cfgs:
        s = by_cfg[c]
        lines.append(
            f"{tex_name(c)} & {s['pck@0.1']:.3f} & \\textbf{{{s['pck@0.2']:.3f}}} & "
            f"{s['auc']:.3f} & {s['detection_rate'] * 100:.1f}\\,\\% & "
            f"{s['latency_p50_ms']:.0f} & {s['extra_detections']:.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TABLES / "pck_main.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_size(summaries: list[dict]) -> Path:
    by_cfg = {s["config"]: s for s in summaries}
    cfgs = [c for c in ORDER if c in by_cfg]
    labels = ["small<5%", "mid 5-10%", "large>=10%"]
    head = {"small<5%": "< 5\\,\\%", "mid 5-10%": "5–10\\,\\%", "large>=10%": "≥ 10\\,\\%"}
    lines = [
        "% автогенерация: python src/make_tables.py",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"конфигурация & " + " & ".join(head[k] for k in labels) + r" \\",
        r"\midrule",
    ]
    for c in cfgs:
        d = by_cfg[c].get("pck@0.2_by_hand_size", {})
        cells = [f"{d[k]:.3f}" if k in d else "—" for k in labels]
        lines.append(f"{tex_name(c)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TABLES / "pck_by_size.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_dataset() -> Path:
    rows = []
    for split in ("test", "train", "val"):
        recs = load_records(split)
        sizes = []
        heads = []
        groups = set()
        left = 0
        for r in recs:
            kp = np.asarray(r["kp_px"], dtype=np.float32)
            sizes.append(max(np.ptp(kp[:, 0]), np.ptp(kp[:, 1])) /
                          max(r["width"], r["height"]))
            heads.append(hand_scale_px(r))
            groups.add(r["group"])
            left += r["side"] == "left"
        sizes = np.array(sizes) * 100
        heads = np.array(heads)
        rows.append((split, len(recs), len(groups), left, len(recs) - left,
                     np.median(sizes), np.mean(sizes < 10), np.median(heads)))
    names = {"test": "тест", "train": "train", "val": "val"}
    lines = [
        "% автогенерация: python src/make_tables.py",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"выборка & изображений & клипов & левых & правых & кисть, \% & <10\,\% & $0.7h$, px \\",
        r"\midrule",
    ]
    for name, n, g, l, r_, med, frac, head in rows:
        lines.append(f"{names[name]} & {n} & {g} & {l} & {r_} & {med:.1f} & "
                     f"{frac * 100:.1f}\\,\\% & {head:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TABLES / "dataset.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_per_joint(summaries: list[dict]) -> Path | None:
    from common import HAND_LABELS

    cfgs = [s["config"] for s in summaries if (RESULTS / "preds" / f"{s['config']}.jsonl").exists()]
    errs = {c: read_errors(c) for c in cfgs}
    labels = [n.replace("_", r"\_") for n in HAND_LABELS]
    lines = [
        "% автогенерация: python src/make_tables.py",
        r"\begin{tabular}{l" + "r" * len(cfgs) + "}",
        r"\toprule",
        r"точка & " + " & ".join(tex_name(c) for c in cfgs) + r" \\",
        r"\midrule",
    ]
    for j, lab in enumerate(labels):
        cells = [f"{pck(errs[c][:, j], 0.2):.2f}" for c in cfgs]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path = TABLES / "pck_per_joint.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    summaries = json.loads((RESULTS / "pck_eval.json").read_text(encoding="utf-8"))
    paths = [write_main(summaries), write_size(summaries), write_dataset()]
    pj = write_per_joint(summaries)
    if pj:
        paths.append(pj)
    for p in paths:
        print(f"записано: {p.relative_to(TABLES.parent.parent)}")


if __name__ == "__main__":
    main()