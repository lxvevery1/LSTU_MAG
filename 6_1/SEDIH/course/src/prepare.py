"""Сборка манифеста данных: единый JSONL для train / val / test.

Что делает:
  1. Берёт 846 тестовых изображений из Voxel51/hand-keypoints (как в ТЗ).
  2. Дополняет их пиксельными координатами, head_size и is_left из
     официальных аннотаций CMU (manual_test) — в samples.json их нет.
  3. Добавляет official manual_train (1912) и делит его по клипам на train/val.
  4. Пишет data/processed/{train,val,test}.jsonl и manifest.json.

Запуск:
    python src/prepare.py
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import numpy as np

from common import PROCESSED, RAW, write_jsonl

HF_DIR = RAW / "hf"
CMU_DIR = RAW / "cmu" / "hand_labels"
NAME_RE = re.compile(r"^(?P<group>.+)_(?P<frame>\d+)_(?P<side>[lr])\.jpg$")


def parse_name(name: str) -> dict:
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"не удалось разобрать имя: {name}")
    return {
        "group": m.group("group"),
        "frame": m.group("frame"),
        "side": "left" if m.group("side") == "l" else "right",
    }


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def cmu_record(json_path: Path, image_path: Path, split: str, source: str) -> dict:
    """Одна запись манифеста по json официального датасета CMU."""
    ann = json.loads(json_path.read_text(encoding="utf-8"))
    kp = [[float(x), float(y)] for x, y, _ in ann["hand_pts"]]
    vis = [float(v) for _, _, v in ann["hand_pts"]]
    w, h = image_size(image_path)
    rec = {
        "name": image_path.name,
        **parse_name(image_path.name),
        "split": split,
        "image": str(image_path.relative_to(PROCESSED.parent.parent)),
        "width": w,
        "height": h,
        "kp_px": kp,
        "kp_norm": [[x / w, y / h] for x, y in kp],
        "vis": vis,
        "head_size": float(ann.get("head_size") or 0.0),
        "is_mpii": int(ann.get("is_mpii") or 0),
        "source": source,
    }
    assert len(kp) == 21, f"{json_path.name}: точек {len(kp)}"
    return rec


def hf_records() -> dict[str, dict]:
    """846 тестовых записей из samples.json (нормированные координаты из ТЗ)."""
    samples = json.loads((HF_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]
    out: dict[str, dict] = {}
    for s in samples:
        name = s["filepath"].split("/")[-1]
        side = "right_hand" if "right_hand" in s else "left_hand"
        kp = s[side]["keypoints"][0]["points"]
        out[name] = {"kp_norm": [[float(x), float(y)] for x, y in kp], "side_key": side}
    return out


def build_test() -> list[dict]:
    """Тест: изображение и нормированные точки из HF, метаданные — из CMU."""
    hf = hf_records()
    test_json = sorted((CMU_DIR / "manual_test").glob("*.json"))
    cmu_names = {p.stem + ".jpg" for p in test_json}
    missing_img = sorted(cmu_names - set(hf))
    missing_ann = sorted(set(hf) - cmu_names)
    if missing_img or missing_ann:
        raise RuntimeError(
            f"расхождение HF и CMU: только в CMU {len(missing_img)}, "
            f"только в HF {len(missing_ann)}: {missing_img[:3]} {missing_ann[:3]}"
        )

    records = []
    for jp in test_json:
        img = HF_DIR / "data" / (jp.stem + ".jpg")  # картинки берём из HF (ТЗ)
        rec = cmu_record(jp, img, split="test", source="hf+cmu")
        hf_side = "right_hand" if hf[img.name]["side_key"] == "right_hand" else "left_hand"
        assert hf_side == rec["side"] + "_hand", img.name
        rec["kp_norm_hf"] = hf[img.name]["kp_norm"]
        records.append(rec)
    return records


def build_train(val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Official manual_train → train/val, сплит по клипам (group), без утечки.

    Группы разного размера (от 1 кадра до сотен), поэтому GroupShuffleSplit даёт
    val в 40 % изображений при test_size=0.2. Раскладываем жадно по размеру:
    группы перемешиваются и идут в val, пока не набрана целевая доля.
    """
    records = [
        cmu_record(jp, (CMU_DIR / "manual_train") / (jp.stem + ".jpg"), "train", "cmu")
        for jp in sorted((CMU_DIR / "manual_train").glob("*.json"))
    ]
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_group.setdefault(r["group"], []).append(i)

    rng = np.random.default_rng(seed)
    order = list(by_group)
    rng.shuffle(order)
    target = int(round(val_frac * len(records)))

    val_idx: list[int] = []
    for g in order:
        if len(val_idx) + len(by_group[g]) // 2 <= target:
            val_idx += by_group[g]
    val_set = set(val_idx)
    train = [r for i, r in enumerate(records) if i not in val_set]
    val = [r for i, r in enumerate(records) if i in val_set]
    for r in val:
        r["split"] = "val"
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    test = build_test()
    train, val = build_train(args.val_frac, args.seed)

    write_jsonl(PROCESSED / "test.jsonl", test)
    write_jsonl(PROCESSED / "train.jsonl", train)
    write_jsonl(PROCESSED / "val.jsonl", val)

    def groups(rs):
        return {r["group"] for r in rs}

    leak_tv = groups(train) & groups(test)
    leak_vv = groups(val) & groups(test)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "test_images": "Voxel51/hand-keypoints (846, официальный test-split CMU)",
            "annotations_and_train": "CMU Panoptic Hand DB (manual_train 1912)",
            "license": "CMU: noncommercial research use only; данные не распространяются",
        },
        "protocol": {
            "pck_norm": "0.7 * head_size (px)",
            "pck_primary_alpha": 0.2,
            "miss_policy": "ошибка = +inf, все 21 точка считается промахом",
            "hand_assignment": "предсказание с ближайшим запястьем к GT",
        },
        "counts": {
            "test": len(test),
            "train": len(train),
            "val": len(val),
            "test_groups": len(groups(test)),
            "train_groups": len(groups(train)),
        },
        "group_leak": {"train_test": sorted(leak_tv), "val_test": sorted(leak_vv)},
        "files": {
            "test": "data/processed/test.jsonl",
            "train": "data/processed/train.jsonl",
            "val": "data/processed/val.jsonl",
        },
    }
    (PROCESSED / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"клипов, пересекающихся с тестом: train={len(leak_tv)}, val={len(leak_vv)}")
    print(f"манифест: {(PROCESSED / 'manifest.json').relative_to(PROCESSED.parent.parent)}")


if __name__ == "__main__":
    main()