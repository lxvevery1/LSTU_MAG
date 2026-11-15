#!/usr/bin/env python3
"""Aggregate per-run evaluation outputs into seed and ablation reports.

Reads ``reports/runs/<run_id>/model_metrics.json`` and writes:
  reports/seed_summary.json
  reports/ablations.md

Usage:
    python src/aggregate_results.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from common import project_path

SEED_RE = re.compile(r"^finbert_h5_seed(\d+)$")
ABLATION_RE = re.compile(r"^ablate_(.+)_seed(\d+)$")


def _metrics(report: dict) -> dict:
    cm = report.get("calibrated_metrics", {}) or {}
    td = report.get("ticker_day_metrics", {}) or {}
    return {
        "n": report.get("test_rows"),
        "macro_f1": cm.get("macro_f1"),
        "mcc": cm.get("mcc"),
        "accuracy": cm.get("accuracy"),
        "auc": cm.get("roc_auc_ovr"),
        "auc_pr_up": cm.get("auc_pr_up"),
        "brier": cm.get("brier"),
        "ece": cm.get("ece"),
        "ticker_day_macro_f1": td.get("macro_f1"),
        "ticker_day_mcc": td.get("mcc"),
        "ticker_day_auc": td.get("roc_auc_ovr"),
        "ece_nll_temperature": report.get("temperature"),
    }


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(arr):
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "values": [float(v) for v in arr],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="reports")
    args = parser.parse_args()
    reports_dir = project_path(args.reports)
    runs_dir = reports_dir / "runs"
    reports = {}
    for path in sorted(runs_dir.glob("*/model_metrics.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reports[obj.get("run_id", path.parent.name)] = obj

    seeds, ablations, extra = {}, {}, {}
    for run_id, report in reports.items():
        m = _metrics(report)
        m["run_id"] = run_id
        if SEED_RE.match(run_id):
            seeds[run_id] = m
        elif ABLATION_RE.match(run_id):
            ablations[run_id] = m
        else:
            extra[run_id] = m

    base = seeds.get("finbert_h5_seed42")
    metric_keys = ["macro_f1", "mcc", "accuracy", "auc", "auc_pr_up", "brier", "ece",
                   "ticker_day_macro_f1", "ticker_day_mcc", "ticker_day_auc"]
    summary = {
        "n_runs": len(reports),
        "seeds": {
            "runs": sorted(seeds),
            "aggregate": {k: _mean_std([seeds[r].get(k) for r in seeds]) for k in metric_keys},
        },
        "ablations": {},
        "extra": extra,
    }
    for run_id, m in ablations.items():
        entry = {"metrics": m}
        if base:
            entry["delta_vs_seed42"] = {
                k: (m.get(k) - base.get(k)) if m.get(k) is not None and base.get(k) is not None else None
                for k in ("macro_f1", "mcc", "auc", "ece", "ticker_day_macro_f1", "ticker_day_mcc")
            }
        summary["ablations"][run_id] = entry

    out = reports_dir / "seed_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Seeds and ablations", "",
             f"Generated from **{len(reports)}** evaluated runs.", ""]
    lines += ["## Seed runs (full model)", "",
              "| run | n | macro-F1 | MCC | AUC | ECE | ticker-day F1 | ticker-day MCC |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for run_id in sorted(seeds):
        m = seeds[run_id]
        lines.append(
            f"| {run_id} | {m['n']} | {m['macro_f1']:.4f} | {m['mcc']:.4f} | "
            f"{m['auc']:.4f} | {m['ece']:.4f} | {m['ticker_day_macro_f1']:.4f} | "
            f"{m['ticker_day_mcc']:.4f} |"
        )
    agg = summary["seeds"]["aggregate"]
    lines += ["", "Mean ± std over available seeds:", "",
              "| metric | mean | std | seeds |", "|---|---:|---:|---:|"]
    for k in metric_keys:
        a = agg[k]
        if a["n"]:
            lines.append(f"| {k} | {a['mean']:.4f} | {a['std']:.4f} | {a['n']} |")
    if len(seeds) < 3:
        lines += ["", f"> Only {len(seeds)} of 3 seeds finished; the mean is not yet the final "
                  "42/43/44 estimate."]

    lines += ["", "## Ablations (seed 42)", ""]
    if ablations:
        lines += ["| ablation | macro-F1 | Δ macro-F1 | MCC | Δ MCC | AUC | ECE | Δ ticker-day F1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for run_id in sorted(ablations):
            entry = summary["ablations"][run_id]
            m = entry["metrics"]
            d = entry.get("delta_vs_seed42", {})
            def fmt(x):
                return "—" if x is None else f"{x:+.4f}"
            def val(v):
                return "—" if v is None else f"{v:.4f}"
            lines.append(
                f"| {run_id} | {val(m.get('macro_f1'))} | {fmt(d.get('macro_f1'))} | "
                f"{val(m.get('mcc'))} | {fmt(d.get('mcc'))} | {val(m.get('auc'))} | "
                f"{val(m.get('ece'))} | {fmt(d.get('ticker_day_macro_f1'))} |"
            )
        if base:
            lines += ["", f"Reference: `finbert_h5_seed42` macro-F1 {base['macro_f1']:.4f}, "
                      f"MCC {base['mcc']:.4f}, AUC {base['auc']:.4f}, ECE {base['ece']:.4f}."]
        lines += ["", "Negative deltas mean the ablated model is worse on the same test set. "
                  "Ablations that change the test population (`no_noise_meme`) are not directly "
                  "paired with the reference and are marked by their row count."]
    else:
        lines += ["No ablation runs have been evaluated yet."]

    lines += ["", "## Encoder / max-length comparisons", ""]
    if extra:
        lines += ["| run | n | macro-F1 | MCC | AUC | ECE |", "|---|---:|---:|---:|---:|---:|"]
        for run_id in sorted(extra):
            m = extra[run_id]
            lines.append(f"| {run_id} | {m['n']} | {m['macro_f1']:.4f} | {m['mcc']:.4f} | "
                         f"{m['auc']:.4f} | {m['ece']:.4f} |")
    else:
        lines += ["Not yet run."]

    (reports_dir / "ablations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} and {reports_dir / 'ablations.md'}")


if __name__ == "__main__":
    main()
