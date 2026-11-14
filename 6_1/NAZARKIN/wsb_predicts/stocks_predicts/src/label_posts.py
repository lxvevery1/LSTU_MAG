#!/usr/bin/env python3
"""Apply the frozen fine-tuned FinBERT classifier to the 16k corpus.

Existing 8,000 posts keep their hand labels.  New posts receive the
classifier's argmax label and the full probability vector; low-confidence
posts, all ANALYSIS_DD candidates and a random 20% of the rest are written to
a review queue.  The label source stays ``classifier_prelabel_unverified``
until a human review is recorded, so downstream reports never pretend that a
model prediction is a manual label.

Usage:
    python src/label_posts.py
    python src/label_posts.py --device cuda --batch-size 64
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import (
    CLASSES,
    INFLUENCE_WEIGHTS,
    load_config,
    project_path,
    read_json,
    setup_logging,
    stable_hash,
    write_json,
)

LOG = logging.getLogger("label_posts")


def choose_device(requested: str = "auto") -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(texts: list[str], tokenizer, model, device: torch.device,
            batch_size: int, max_len: int) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**enc).logits.float()
            else:
                logits = model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out.append(probs)
        if (start // batch_size) % 20 == 0:
            LOG.info("classified %s / %s", min(start + batch_size, len(texts)), len(texts))
    return np.vstack(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    lab_cfg = cfg["labels"]
    device = choose_device(args.device)
    ckpt = args.checkpoint or str(project_path(lab_cfg["classifier_checkpoint"]))
    batch_size = args.batch_size or int(lab_cfg.get("batch_size", 64))
    max_len = int(lab_cfg.get("max_len", 256))
    LOG.info("device=%s checkpoint=%s", device, ckpt)

    raw_path = project_path(cfg["paths"]["posts_raw"])
    obj = read_json(raw_path)
    years = obj["years"]
    posts = [p for arr in years.values() for p in arr]
    LOG.info("posts: %s", len(posts))

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.to(device)

    # Keep manual labels; classifier probabilities are stored for all posts
    # (they are a feature for B6 and an audit signal).
    texts = [
        ((p.get("title") or "") + "\n\n" + (p.get("body") or "")).strip()
        for p in posts
    ]
    probs = predict(texts, tokenizer, model, device, batch_size, max_len)

    # Review queue: low confidence, every ANALYSIS_DD candidate, and 20% of
    # the remaining posts (deterministic hash-based sample).
    review_fraction = float(lab_cfg.get("review_fraction", 0.20))
    conf_threshold = float(lab_cfg.get("confidence_review_threshold", 0.55))
    label_dist = collections.Counter()
    by_year = collections.defaultdict(collections.Counter)
    review = collections.Counter()
    gold = []

    for p, prob in zip(posts, probs):
        prob = np.asarray(prob, dtype=float)
        pred_id = int(np.argmax(prob))
        pred_label = CLASSES[pred_id]
        confidence = float(prob[pred_id])
        p["class_probs"] = {c: round(float(prob[i]), 6) for i, c in enumerate(CLASSES)}
        p["classifier_label"] = pred_label
        p["confidence"] = round(confidence, 6)
        p["influence_weight"] = float(INFLUENCE_WEIGHTS.get(
            p.get("class_label") or pred_label, 0.0))

        if p.get("class_label") in CLASSES and p.get("label_source") == "manual":
            p["influence_weight"] = float(INFLUENCE_WEIGHTS[p["class_label"]])
            p["review_status"] = "manual"
            p["review_required"] = False
        else:
            p["class_label"] = pred_label
            p["label_source"] = "classifier_prelabel_unverified"
            need = (
                confidence < conf_threshold
                or pred_label == "ANALYSIS_DD"
                or (stable_hash(p["id"], int(cfg["project"]["seed"]) + 7) % 1000)
                < int(round(review_fraction * 1000))
            )
            p["review_status"] = "pending" if need else "not_required"
            p["review_required"] = bool(need)
            if need:
                review["needs_review"] += 1
            else:
                review["auto_accepted"] += 1

        label_dist[p["class_label"]] += 1
        y = str(datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year)
        by_year[y][p["class_label"]] += 1
        if p.get("label_source") != "manual":
            gold.append(p)

    # A stratified 500-post gold-audit sample.  The sample file intentionally
    # contains no labels: a human annotator fills ``gold_label`` later.
    gold_size = int(lab_cfg.get("gold_audit_size", 500))
    by_class = collections.defaultdict(list)
    for p in gold:
        by_class[p["class_label"]].append(p)
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    sample = []
    if gold:
        per_class = max(1, gold_size // max(1, len(by_class)))
        for label, arr in by_class.items():
            idx = rng.choice(len(arr), size=min(per_class, len(arr)), replace=False)
            sample.extend(arr[int(i)] for i in idx)
        if len(sample) < gold_size:
            remaining = [p for p in gold if p not in sample]
            rng.shuffle(remaining)
            sample.extend(remaining[: gold_size - len(sample)])
    sample = sample[:gold_size]
    rng.shuffle(sample)

    sample_rows = []
    for p in sample:
        sample_rows.append({
            "post_id": p["id"],
            "year": str(datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).year),
            "title": p["title"],
            "body": p["body"],
            "flair": p.get("flair", ""),
            "classifier_label": p.get("classifier_label"),
            "classifier_confidence": p.get("confidence"),
            "gold_label": None,
        })
    write_json(project_path(cfg["paths"]["gold_audit"]), {
        "instructions": "Fill gold_label with one of: " + ", ".join(CLASSES),
        "n": len(sample_rows),
        "sample": sample_rows,
    })

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": ckpt,
        "max_len": max_len,
        "n_posts": len(posts),
        "class_distribution": dict(label_dist),
        "class_distribution_pct": {
            k: round(100 * v / max(1, len(posts)), 2) for k, v in label_dist.items()
        },
        "per_year": {y: dict(c) for y, c in sorted(by_year.items())},
        "review": dict(review),
        "manual_posts": sum(1 for p in posts if p.get("label_source") == "manual"),
        "new_posts": sum(1 for p in posts if p.get("label_source") != "manual"),
        "gold_audit_sample": len(sample_rows),
        "confidence_threshold": conf_threshold,
        "review_fraction": review_fraction,
    }
    obj["meta"] = {**obj.get("meta", {}), **report}
    obj["years"] = years
    write_json(project_path(cfg["paths"]["posts_labeled"]), obj)
    write_json(project_path(cfg["paths"]["reports"]) / "prelabel_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
