#!/usr/bin/env python3
"""Score ticker extraction against the 300-post gold sample (TZ 8.1).

Targets: precision >= 0.95, recall >= 0.90.  Human annotators fill
``gold_tickers`` in ``data/raw/ticker_gold_sample.json`` (created by
``src/sample_ticker_gold.py``).  Until then this script reports status
``blocked`` rather than inventing a number.

Usage:
    python src/score_ticker_gold.py
    python src/score_ticker_gold.py --gold data/raw/ticker_gold_sample.json
"""
from __future__ import annotations

import argparse
import collections
import json
from datetime import datetime, timezone

import numpy as np

from common import load_config, project_path, read_json, write_json

TARGET_PRECISION = 0.95
TARGET_RECALL = 0.90


def prf(pred_sets: list[set], gold_sets: list[set]) -> dict:
    tp = fp = fn = 0
    for pred, gold in zip(pred_sets, gold_sets):
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(1 for p, g in zip(pred_sets, gold_sets) if p == g)
    return {
        "n_posts": len(pred_sets),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_match": float(exact / max(1, len(pred_sets))),
        "n_pred_links": int(sum(len(p) for p in pred_sets)),
        "n_gold_links": int(sum(len(g) for g in gold_sets)),
    }


def bootstrap_prf(pred_sets: list[set], gold_sets: list[set],
                  n_iter: int, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    out = collections.defaultdict(list)
    n = len(pred_sets)
    if n == 0:
        return {}
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        r = prf([pred_sets[i] for i in idx], [gold_sets[i] for i in idx])
        for k in ("precision", "recall", "f1", "exact_match"):
            out[k].append(r[k])
    return {
        k: {"lo": float(np.quantile(v, 0.025)), "hi": float(np.quantile(v, 0.975))}
        for k, v in out.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--gold", default="data/raw/ticker_gold_sample.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    universe = set(load_config(cfg["tickers"]["config"])["universe"])
    path = project_path(args.gold)
    obj = read_json(path)
    sample = obj.get("sample", [])

    pred_sets, gold_sets, skipped = [], [], collections.Counter()
    for row in sample:
        gold = row.get("gold_tickers")
        if gold is None:
            skipped["unlabelled"] += 1
            continue
        if not isinstance(gold, list):
            skipped["invalid_type"] += 1
            continue
        gold_clean = {str(t).upper().strip() for t in gold if str(t).strip()}
        invalid = gold_clean - universe
        if invalid:
            skipped[f"outside_universe:{','.join(sorted(invalid))}"] += 1
            continue
        pred = {str(t).upper().strip() for t in (row.get("auto_tickers") or [])
                if str(t).strip()}
        pred_sets.append(pred & universe)
        gold_sets.append(gold_clean)

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "ticker_extraction_gold_audit",
        "gold_file": str(path),
        "sample_size": len(sample),
        "labelled": len(gold_sets),
        "coverage": round(len(gold_sets) / max(1, len(sample)), 4),
        "skipped": dict(skipped),
        "targets": {"precision": TARGET_PRECISION, "recall": TARGET_RECALL},
    }
    if len(gold_sets) < 20:
        report.update({
            "status": "blocked",
            "reason": ("Fewer than 20 gold-labelled posts are available. Fill "
                       "gold_tickers in ticker_gold_sample.json and rerun."),
            "passes_targets": None,
        })
    else:
        result = prf(pred_sets, gold_sets)
        report.update(result)
        report["bootstrap_95ci"] = bootstrap_prf(pred_sets, gold_sets, args.bootstrap)
        report["passes_targets"] = bool(
            result["precision"] >= TARGET_PRECISION and result["recall"] >= TARGET_RECALL)
        report["status"] = "scored"
        report["n_negative_posts"] = int(sum(1 for g in gold_sets if not g))
        if len(gold_sets) < len(sample):
            report["warning"] = (f"Only {len(gold_sets)}/{len(sample)} posts scored; "
                                 "precision/recall use the labelled subset.")

    out = project_path(cfg["paths"]["reports"]) / "ticker_gold_audit.json"
    write_json(out, report)
    print(json.dumps({k: report[k] for k in
                      ("status", "sample_size", "labelled", "coverage",
                       "precision", "recall", "passes_targets") if k in report},
                     indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
