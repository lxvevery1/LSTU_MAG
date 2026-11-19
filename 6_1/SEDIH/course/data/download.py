"""Скачивание данных и моделей для проекта «ключевые точки руки».

Источники:
  * Voxel51/hand-keypoints  — 846 тестовых изображений + samples.json (ТЗ).
  * CMU Panoptic Hand DB    — официальные manual_train (1912) и manual_test (846)
                              с пиксельными координатами, head_size и is_left.
  * MediaPipe model bundles — hand_landmarker.task и pose_landmarker_lite.task.

Запуск:
    python data/download.py              # всё
    python data/download.py --part hf    # только HF-датасет
    python data/download.py --part cmu   # только CMU-архив
    python data/download.py --part models

Лицензия CMU: noncommercial research use only — файлы не коммитятся,
скрипт лишь скачивает их локально (data/raw/ в .gitignore).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
HF_DIR = RAW / "hf"
CMU_DIR = RAW / "cmu"
MODELS_DIR = ROOT / "models"

HF_REPO = "Voxel51/hand-keypoints"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
CMU_ZIP = "http://domedb.perception.cs.cmu.edu/panopticDB/hands/hand_labels.zip"
CMU_ZIP_SIZE = 589_711_996

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    ),
}

USER_AGENT = "Mozilla/5.0 (compatible; hand-keypoints-course/1.0)"


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def download(
    url: str, dest: Path, expected_size: int | None = None, retries: int = 3
) -> Path:
    """Скачивает файл с временным именем .part (с докачкой) и атомарно переименовывает."""
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session().get(url, headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code != 206:  # сервер не умеет resume
                    have = 0
                r.raise_for_status()
                total = expected_size or (int(r.headers.get("Content-Length") or 0) + have) or 0
                mode = "ab" if have else "wb"
                with open(tmp, mode) as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True,
                    desc=f"  {dest.name} (попытка {attempt})",
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            print(f"  сбой: {exc}; продолжаю через 3 c")
            time.sleep(3)
            continue
        size = tmp.stat().st_size
        if expected_size and size != expected_size:
            if attempt == retries:
                raise RuntimeError(f"{dest.name}: ожидалось {expected_size} байт, получено {size}")
            print(f"  неполный файл ({size}/{expected_size}), докачиваю")
            continue
        tmp.rename(dest)
        return dest
    raise RuntimeError(f"не удалось скачать {url}")


# --- Voxel51/hand-keypoints -------------------------------------------------
def download_hf(force: bool = False) -> None:
    import json

    print("HF: Voxel51/hand-keypoints")
    samples_path = HF_DIR / "samples.json"
    if force and samples_path.exists():
        samples_path.unlink()
    download(f"{HF_BASE}/samples.json", samples_path)

    samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
    names = [s["filepath"].split("/")[-1] for s in samples]
    print(f"  в samples.json записей: {len(samples)}")

    img_dir = HF_DIR / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if force or not (img_dir / n).exists()]
    print(f"  изображений к скачиванию: {len(todo)} (уже есть: {len(names) - len(todo)})")
    if not todo:
        return

    def fetch(name: str) -> str:
        download(f"{HF_BASE}/data/{name}", img_dir / name)
        return name

    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, n): n for n in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="  images"):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{futures[fut]}: {exc}")
    if errors:
        raise RuntimeError("не скачано:\n" + "\n".join(errors[:10]))
    print(f"  готово: {len(list(img_dir.glob('*.jpg')))} jpg")


# --- CMU Panoptic Hand DB ---------------------------------------------------
def download_cmu(keep_zip: bool = False) -> None:
    print("CMU: hand_labels.zip (manual_train + manual_test)")
    zip_path = RAW / "hand_labels.zip"
    download(CMU_ZIP, zip_path, expected_size=CMU_ZIP_SIZE)

    target = CMU_DIR / "hand_labels"
    if target.exists() and any(target.iterdir()):
        print(f"  уже распаковано: {target.relative_to(ROOT)}")
    else:
        print("  распаковка...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(CMU_DIR)
    for split in ("manual_train", "manual_test"):
        d = target / split
        print(f"  {split}: {len(list(d.glob('*.jpg')))} jpg, {len(list(d.glob('*.json')))} json")

    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print("  архив удалён (--keep-zip чтобы оставить)")


# --- модели MediaPipe -------------------------------------------------------
def download_models(force: bool = False) -> None:
    print("MediaPipe model bundles")
    for name, url in MODELS.items():
        dest = MODELS_DIR / name
        if force and dest.exists():
            dest.unlink()
        download(url, dest)
    print(f"  модели в {MODELS_DIR.relative_to(ROOT)}/")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--part", choices=["all", "hf", "cmu", "models"], default="all")
    ap.add_argument("--force", action="store_true", help="перекачать HF/модели")
    ap.add_argument("--keep-zip", action="store_true", help="не удалять CMU-архив")
    args = ap.parse_args()

    t0 = time.time()
    if args.part in ("all", "hf"):
        download_hf(force=args.force)
    if args.part in ("all", "models"):
        download_models(force=args.force)
    if args.part in ("all", "cmu"):
        download_cmu(keep_zip=args.keep_zip)

    if not shutil.which("ffmpeg"):
        print("\n[!] ffmpeg не найден — для H.264-рендера видео понадобится")
    print(f"\nГотово за {time.time() - t0:.1f} c")
    return 0


if __name__ == "__main__":
    sys.exit(main())