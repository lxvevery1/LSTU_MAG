#!/usr/bin/env python3
"""Audit the 4-class classifier against manual labels.

The 500-post gold sample is intentionally unlabelled, so a true Cohen's kappa
for the new semi-supervised labels is not available without human annotation.
This script produces the best reproducible proxy: agreement between the frozen
classifier and the 8,000 hand-labelled posts, restricted (when the split file is
available) to the classifier's 20% held-out split.

The result is a **proxy**, not a gold audit of the new 8,000 prelabels.  It is
reported with that caveat in `reports/data_card.md` and `reports/model_card.md`.

Usage:
    python src/audit_classifier.py
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--holdout", default="../../wsb_classification/logreg_hand/split_test_ids.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)

    manual = read_json(cfg["corpus"]["existing_posts"])
    labeled = read_json(cfg["paths"]["posts_labeled"])
    manual_map = {}
    for arr in manual["years"].values():
        for p in arr:
            manual_map[p["id"]] = p.get("label") or p.get("class_label")
    labeled_map = {p["id"]: p for arr in labeled["years"].values() for p in arr}

    holdout_path = project_path(args.holdout)
    holdout = None
    if holdout_path.exists():
        obj = json.loads(holdout_path.read_text(encoding="utf-8"))
        holdout = set(obj.get("test_ids", []))

    rows = []
    for pid, gold in manual_map.items():
        p = labeled_map.get(pid)
        if p is None or not gold:
            continue
        if holdout is not None and pid not in holdout:
            continue
        probs = p.get("class_probs") or {}
        if not probs:
            continue
        pred = max(probs, key=probs.get)
        rows.append((pid, gold, pred, float(max(probs.values()))))

    if not rows:
        raise SystemExit("no manual/classifier pairs found; check the config and holdout path")
    y = [r[1] for r in rows]
    pred = [r[2] for r in rows]
    kappa = cohen_kappa_score(y, pred)
    accuracy = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro")
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)

    rng = np.random.default_rng(42)
    kappas = []
    for _ in range(args.bootstrap):
        idx = rng.integers(0, len(rows), size=len(rows))
        yy = [y[i] for i in idx]
        pp = [pred[i] for i in idx]
        if len(set(yy)) > 1 and len(set(pp)) > 1:
            kappas.append(cohen_kappa_score(yy, pp))
    kappas = np.asarray(kappas, dtype=float)

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "classifier_vs_manual_proxy",
        "caveat": (
            "This is agreement between the frozen 4-class classifier and hand labels, "
            "not a human audit of the 8,000 new classifier prelabels. The 500-post "
            "gold sample remains unlabelled, so a true kappa for the new labels is "
            "not claimed."
        ),
        "holdout_restricted": holdout is not None,
        "n": len(rows),
        "kappa": float(kappa),
        "kappa_95ci": {
            "lo": float(np.quantile(kappas, 0.025)) if len(kappas) else None,
            "hi": float(np.quantile(kappas, 0.975)) if len(kappas) else None,
            "n_bootstrap": int(len(kappas)),
        },
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(
            y, pred, labels=CLASSES).tolist(),
        "label_distribution_manual": dict(collections.Counter(y)),
        "label_distribution_pred": dict(collections.Counter(pred)),
    }
    out = project_path(cfg["paths"]["reports"]) / "classifier_audit_proxy.json"
    write_json(out, report)
    print(json.dumps({k: report[k] for k in
                      ("kind", "holdout_restricted", "n", "kappa", "kappa_95ci",
                       "accuracy", "macro_f1")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
