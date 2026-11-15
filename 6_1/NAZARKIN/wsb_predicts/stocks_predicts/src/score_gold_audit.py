#!/usr/bin/env python3
"""Score the 500-post class-label gold audit (TZ 8.1, kappa >= 0.65).

Human annotators fill ``gold_label`` in ``data/raw/gold_audit_sample.json``
with one of the four classes.  This script compares those labels with the
frozen classifier's ``classifier_label`` and writes ``reports/gold_audit.json``.

When no labels have been filled yet it exits successfully with status
``blocked`` and an explicit count, rather than pretending the audit passed or
silently skipping it.

Usage:
    python src/score_gold_audit.py
    python src/score_gold_audit.py --gold data/raw/gold_audit_sample.json
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from common import CLASSES, load_config, project_path, read_json, write_json

TARGET_KAPPA = 0.65


def score(y: list[str], pred: list[str], bootstrap: int, seed: int = 42) -> dict:
    kappa = float(cohen_kappa_score(y, pred))
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        yy = [y[i] for i in idx]
        pp = [pred[i] for i in idx]
        if len(set(yy)) > 1 and len(set(pp)) > 1:
            ks.append(cohen_kappa_score(yy, pp))
    ks = np.asarray(ks, dtype=float)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    return {
        "n": len(y),
        "kappa": kappa,
        "kappa_95ci": {
            "lo": float(np.quantile(ks, 0.025)) if len(ks) else None,
            "hi": float(np.quantile(ks, 0.975)) if len(ks) else None,
            "n_bootstrap": int(len(ks)),
        },
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix_rows_gold_cols_classifier": confusion_matrix(
            y, pred, labels=CLASSES).tolist(),
        "gold_distribution": dict(collections.Counter(y)),
        "classifier_distribution": dict(collections.Counter(pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--gold", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gold_path = project_path(args.gold or cfg["paths"]["gold_audit"])
    obj = read_json(gold_path)
    sample = obj.get("sample", [])
    y, pred, skipped = [], [], collections.Counter()
    for row in sample:
        gold = row.get("gold_label")
        classifier = row.get("classifier_label")
        if gold is None or str(gold).strip() == "":
            skipped["unlabelled"] += 1
            continue
        if gold not in CLASSES:
            skipped[f"invalid:{gold}"] += 1
            continue
        if classifier not in CLASSES:
            skipped["missing_classifier_label"] += 1
            continue
        y.append(str(gold))
        pred.append(str(classifier))

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "gold_audit_class_labels",
        "gold_file": str(gold_path),
        "sample_size": len(sample),
        "labelled": len(y),
        "coverage": round(len(y) / max(1, len(sample)), 4),
        "skipped": dict(skipped),
        "target_kappa": TARGET_KAPPA,
    }
    if len(y) < 20:
        report.update({
            "status": "blocked",
            "reason": ("Fewer than 20 gold labels are available. Fill gold_label in "
                       "the gold_audit_sample.json file (or pass --gold) and rerun."),
            "passes_target": None,
        })
    else:
        result = score(y, pred, args.bootstrap)
        report.update(result)
        report["passes_target"] = bool(result["kappa"] >= TARGET_KAPPA)
        report["status"] = "scored"
        if len(y) < len(sample):
            report["warning"] = (f"Only {len(y)}/{len(sample)} rows scored; kappa is "
                                 "computed on the labelled subset.")

    out = project_path(cfg["paths"]["reports"]) / "gold_audit.json"
    write_json(out, report)
    print(json.dumps({k: report[k] for k in
                      ("status", "sample_size", "labelled", "coverage", "passes_target",
                       "kappa", "kappa_95ci") if k in report}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
