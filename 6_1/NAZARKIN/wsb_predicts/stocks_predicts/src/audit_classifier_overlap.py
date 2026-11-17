#!/usr/bin/env python3
"""Audit overlap between the frozen stage-1 classifier and the direction test split.

The stage-1 FinBERT classifier was trained on 6,400 of the 8,000 manual posts,
stratified by year and class, using a random 80/20 split.  Its four class
probabilities are model inputs (and B6 is built from them).  That split is not
temporal, so some 2025-2026 direction-test posts were seen by the classifier.

This script quantifies the overlap and evaluates the direction model on the
"clean" subset whose class probabilities do not come from a classifier that saw
their manual label:

  clean = manual posts held out by the classifier (1,600 total) + all
          classifier-prelabelled posts (the classifier generated those labels
          but was not trained on them as manual ground truth).

It also reports the leaked subset separately and the paired delta versus B2 on
the clean rows, so the negative headline can be checked for dependence on this
residual leakage path.

Usage:
    python src/audit_classifier_overlap.py
    python src/audit_classifier_overlap.py --bootstrap 1000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from common import load_config, project_path, read_json, write_json
from metrics import bootstrap_delta, compute_metrics


def _probs(df: pd.DataFrame) -> np.ndarray:
    return df[["p_DOWN", "p_FLAT", "p_UP"]].to_numpy(dtype=float)


def _metrics(df: pd.DataFrame) -> dict:
    return compute_metrics(df["y"].to_numpy(dtype=int), _probs(df))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--run-id", default="finbert_h5_seed42")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    cfg = load_config(args.config)

    pairs = pd.read_parquet(project_path(cfg["paths"]["pairs_h5"]))
    model = pd.read_parquet(
        project_path(cfg["paths"]["reports"]) / "runs" / args.run_id /
        "model_predictions.parquet")
    model = model.merge(
        pairs[["post_id", "ticker", "t0_date", "label_source"]],
        on=["post_id", "ticker", "t0_date"], how="left", validate="one_to_one")

    split_path = project_path(cfg.get("classifier", {}).get(
        "split_test_ids", "../../chromium_extension/api_model/finbert/split_test_ids.json"))
    if not split_path.exists():
        report = {
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "kind": "classifier_training_overlap",
            "status": "unavailable",
            "reason": f"classifier split file not found: {split_path}",
        }
        write_json(project_path("reports/classifier_overlap_audit.json"), report)
        print(json.dumps(report, indent=2))
        return

    clf_test = {str(x) for x in read_json(split_path)["test_ids"]}
    model["clf_seen"] = np.where(
        model["label_source"].eq("manual"),
        ~model["post_id"].astype(str).isin(clf_test),
        False,
    )
    model["clean_subset"] = ~model["clf_seen"]
    model["y"] = model["direction_label"].map({"DOWN": 0, "FLAT": 1, "UP": 2})
    model = model.dropna(subset=["y"]).reset_index(drop=True)
    model["y"] = model["y"].astype(int)

    all_metrics = _metrics(model)
    clean = model[model["clean_subset"]].reset_index(drop=True)
    leaked = model[model["clf_seen"]].reset_index(drop=True)
    # Strictest subset: manual ground truth that the stage-1 classifier never
    # saw.  It is small, so its bootstrap interval is wide, but it avoids both
    # the classifier-trained manual posts and the classifier-prelabelled posts.
    manual_heldout = model[
        model["label_source"].eq("manual") &
        model["post_id"].astype(str).isin(clf_test)
    ].reset_index(drop=True)
    clean_metrics = _metrics(clean)
    leaked_metrics = _metrics(leaked)
    manual_heldout_metrics = _metrics(manual_heldout)

    overlap = {}
    for split in sorted(model["split"].dropna().unique()):
        sub = pairs[(pairs["split"] == split)].drop_duplicates("post_id")
        manual = sub[sub["label_source"] == "manual"]
        prelabel = sub[sub["label_source"] != "manual"]
        seen = ~manual["post_id"].astype(str).isin(clf_test)
        overlap[str(split)] = {
            "unique_posts": int(len(sub)),
            "manual_unique_posts": int(len(manual)),
            "manual_seen_by_classifier": int(seen.sum()),
            "manual_seen_share": round(float(seen.mean()), 4) if len(manual) else None,
            "prelabel_unique_posts": int(len(prelabel)),
        }

    baseline = pd.read_parquet(
        project_path(cfg["paths"]["reports"]) / "baselines_predictions.parquet")
    b2 = baseline[baseline["model"] == "B2"][["post_id", "ticker", "t0_date",
                                               "p_DOWN", "p_FLAT", "p_UP"]] \
        .rename(columns={"p_DOWN": "b_DOWN", "p_FLAT": "b_FLAT", "p_UP": "b_UP"})

    def delta_vs_b2(subset: pd.DataFrame) -> dict:
        merged = subset.merge(b2, on=["post_id", "ticker", "t0_date"], how="inner",
                              validate="one_to_one")
        out = {}
        if not len(merged):
            return out
        dates = merged["t0_date"].astype(str).to_numpy()
        probs_model = merged[["p_DOWN", "p_FLAT", "p_UP"]].to_numpy(dtype=float)
        probs_b2 = merged[["b_DOWN", "b_FLAT", "b_UP"]].to_numpy(dtype=float)
        y = merged["y"].to_numpy(dtype=int)
        for metric in ("macro_f1", "mcc"):
            out[metric] = bootstrap_delta(
                y, probs_model, probs_b2, dates, metric=metric,
                n_iter=args.bootstrap)
        return out

    deltas = delta_vs_b2(clean)
    manual_heldout_deltas = delta_vs_b2(manual_heldout)

    report = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "classifier_training_overlap",
        "status": "ok",
        "run_id": args.run_id,
        "classifier_split": str(split_path),
        "classifier_split_is_temporal": False,
        "note": ("The stage-1 classifier used a random 80/20 split stratified by "
                 "(year, class), so 80% of manual direction-test posts were seen "
                 "during classifier training. Class probabilities are inputs, so "
                 "this is residual auxiliary-label leakage, not direction-label "
                 "leakage."),
        "overlap": overlap,
        "rows": {
            "all": int(len(model)),
            "clean": int(len(clean)),
            "leaked": int(len(leaked)),
            "manual_heldout": int(len(manual_heldout)),
            "clean_share": round(len(clean) / max(1, len(model)), 4),
            "clean_unique_posts": int(clean["post_id"].nunique()),
            "leaked_unique_posts": int(leaked["post_id"].nunique()),
            "manual_heldout_unique_posts": int(manual_heldout["post_id"].nunique()),
        },
        "metrics": {
            "all": all_metrics,
            "clean": clean_metrics,
            "leaked": leaked_metrics,
            "manual_heldout": manual_heldout_metrics,
        },
        "delta_vs_B2_clean": deltas,
        "delta_vs_B2_manual_heldout": manual_heldout_deltas,
    }
    out = project_path(cfg["paths"]["reports"]) / "classifier_overlap_audit.json"
    write_json(out, report)
    print(json.dumps({
        "rows": report["rows"],
        "all": {k: round(all_metrics[k], 4) for k in ("macro_f1", "mcc")},
        "clean": {k: round(clean_metrics[k], 4) for k in ("macro_f1", "mcc")},
        "leaked": {k: round(leaked_metrics[k], 4) for k in ("macro_f1", "mcc")},
        "manual_heldout": {k: round(manual_heldout_metrics[k], 4)
                           for k in ("macro_f1", "mcc")},
        "delta_vs_B2_clean": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                  for kk, vv in v.items()} for k, v in deltas.items()},
        "delta_vs_B2_manual_heldout": {
            k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items()}
            for k, v in manual_heldout_deltas.items()},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
