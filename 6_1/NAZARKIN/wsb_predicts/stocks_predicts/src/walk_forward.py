#!/usr/bin/env python3
"""Walk-forward validation (TZ section 6.1, additional scheme).

The TZ asks for folds Y in {2023, 2024, 2025} with ``train <= Y-1`` and
``test = Y``.  Early stopping and temperature scaling need a validation year,
so this driver uses the standard out-of-time variant:

    train <= Y-2, validation = Y-1, test = Y, 10-session embargo.

Fold 2025 is exactly the primary temporal split (train 2019-2023, val 2024,
test 2025), so its metrics are read from the reference run's ``per_year``
block instead of retraining.  Folds 2023 and 2024 are built and trained here.

Outputs:
    configs/walk_forward/wf_YYYY.yaml        generated fold config
    reports/walk_forward/wf_YYYY/            fold data/model/eval artifacts
    reports/walk_forward.json                machine-readable fold summary
    reports/walk_forward.md                  human-readable fold table

Usage:
    python src/walk_forward.py --folds 2023,2024 --bootstrap 1000
    python src/walk_forward.py --folds 2023 --force
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import load_config, project_path, read_json, setup_logging, write_json  # noqa: E402

LOG = logging.getLogger("walk_forward")
METRIC_KEYS = ["macro_f1", "mcc", "balanced_accuracy", "roc_auc_ovr", "auc_pr_up",
               "brier", "ece", "accuracy"]


def make_fold_config(test_year: int) -> Path:
    cfg = load_config("configs/data.yaml")
    train_end = test_year - 2
    val_year = test_year - 1
    cfg["project"]["data_version"] = f"wsb16k-wf{test_year}"
    cfg["splits"] = {
        "train_years": [2019, train_end],
        "val_year": val_year,
        "test_years": [test_year, test_year],
        "embargo_trading_days": int(cfg["splits"].get("embargo_trading_days", 10)),
    }
    fold_dir = f"data/processed/walk_forward"
    cfg["paths"]["dataset"] = f"{fold_dir}/wf_{test_year}.parquet"
    cfg["paths"]["pairs_h5"] = f"{fold_dir}/wf_{test_year}_pairs_h5.parquet"
    cfg["paths"]["reports"] = f"reports/walk_forward/wf_{test_year}"
    out = project_path("configs/walk_forward") / f"wf_{test_year}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return out


def metrics_from_report(report: dict) -> dict:
    m = report.get("calibrated_metrics", {}) or {}
    return {k: m.get(k) for k in METRIC_KEYS} | {
        "n": report.get("test_rows"),
        "n_groups": report.get("test_groups"),
        "temperature": report.get("temperature"),
    }


def metrics_from_per_year(block: dict) -> dict:
    return {k: block.get(k) for k in METRIC_KEYS} | {
        "n": block.get("n"),
        "n_groups": None,
        "temperature": None,
    }


def run(cmd: list[str]) -> None:
    LOG.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2023,2024,2025")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    setup_logging()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    results = {}
    reports_dir = project_path("reports")

    for year in folds:
        run_id = f"wf_{year}_seed{args.seed}"
        fold_reports = project_path(f"reports/walk_forward/wf_{year}")
        metrics_path = fold_reports / "runs" / run_id / "model_metrics.json"

        if year == 2025:
            # Existing temporal split: train 2019-2023 (<=Y-2), val 2024 (Y-1),
            # test 2025.  Reuse the reference run instead of retraining.
            ref_cfg = make_fold_config(year)
            ref_metrics_path = reports_dir / "runs" / "finbert_h5_seed42" / "model_metrics.json"
            if not ref_metrics_path.exists():
                LOG.warning("skip 2025: %s does not exist", ref_metrics_path)
                continue
            ref = read_json(ref_metrics_path)
            block = (ref.get("per_year") or {}).get("2025")
            if not block:
                LOG.warning("skip 2025: reference run has no per_year['2025']")
                continue
            fold_cfg = load_config(str(ref_cfg))
            n_test = int(block.get("n", 0))
            results["2025"] = {
                "run_id": "finbert_h5_seed42",
                "source": "reference run per_year 2025 (same train/val/test boundary)",
                "config": str(ref_cfg.relative_to(ROOT)),
                "train_years": fold_cfg["splits"]["train_years"],
                "val_year": fold_cfg["splits"]["val_year"],
                "test_year": 2025,
                "metrics": metrics_from_per_year(block),
                "n_test": n_test,
            }
            LOG.info("2025: reused %s (n=%s)", ref_metrics_path, n_test)
            continue

        if metrics_path.exists() and not args.force:
            LOG.info("skip %s: metrics already exist", run_id)
            report = read_json(metrics_path)
            fold_cfg = load_config(str(make_fold_config(year)))
            results[str(year)] = {
                "run_id": run_id,
                "source": "fold checkpoint",
                "config": f"configs/walk_forward/wf_{year}.yaml",
                "train_years": fold_cfg["splits"]["train_years"],
                "val_year": fold_cfg["splits"]["val_year"],
                "test_year": year,
                "metrics": metrics_from_report(report),
                "n_test": report.get("test_rows"),
            }
            continue

        fold_cfg_path = make_fold_config(year)
        fold_cfg = load_config(str(fold_cfg_path))
        LOG.info("=== fold %s: train %s, val %s, test %s ===", year,
                 fold_cfg["splits"]["train_years"], fold_cfg["splits"]["val_year"],
                 fold_cfg["splits"]["test_years"])
        start = time.time()

        if args.force or not project_path(fold_cfg["paths"]["dataset"]).exists():
            run([sys.executable, "src/build_dataset.py", "--config", str(fold_cfg_path),
                 "--all-horizons"])
        else:
            LOG.info("reuse dataset %s", fold_cfg["paths"]["dataset"])

        model_dir = project_path("models") / run_id
        checkpoint_ok = (model_dir / "model.pt").exists() and (
            model_dir / "training_summary.json").exists()
        if args.force or not checkpoint_ok:
            run([sys.executable, "src/train.py", "--config", str(fold_cfg_path),
                 "--seed", str(args.seed), "--run-id", run_id])
        else:
            LOG.info("reuse checkpoint %s", model_dir)

        run([sys.executable, "src/evaluate.py", "--config", str(fold_cfg_path),
             "--run-id", run_id, "--bootstrap", str(args.bootstrap)])

        report = read_json(metrics_path)
        results[str(year)] = {
            "run_id": run_id,
            "source": "fold checkpoint",
            "config": f"configs/walk_forward/wf_{year}.yaml",
            "train_years": fold_cfg["splits"]["train_years"],
            "val_year": fold_cfg["splits"]["val_year"],
            "test_year": year,
            "metrics": metrics_from_report(report),
            "n_test": report.get("test_rows"),
            "seconds": round(time.time() - start, 1),
        }
        LOG.info("fold %s done in %.1f min", year, (time.time() - start) / 60)

    if not results:
        raise SystemExit("no walk-forward folds available")

    summary = {
        "scheme": "train <= Y-2, val = Y-1, test = Y, 10-session embargo",
        "note": ("The TZ text says train <= Y-1; one year is held out for early "
                 "stopping and temperature scaling, so training stops at Y-2."),
        "seed": args.seed,
        "folds": results,
        "aggregate": {},
    }
    years = sorted(results)
    for key in METRIC_KEYS:
        vals = [results[y]["metrics"].get(key) for y in years]
        arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
        if len(arr):
            summary["aggregate"][key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n_folds": int(len(arr)),
                "values": [float(v) for v in arr],
            }

    write_json(reports_dir / "walk_forward.json", summary)

    lines = ["# Walk-forward validation", "",
             "Scheme: **train ≤ Y−2, validation = Y−1, test = Y**, 10-session embargo. "
             "The TZ text says train ≤ Y−1; one year is reserved for early stopping "
             "and temperature scaling, so training stops at Y−2.", "",
             "| fold | train | val | test | n | macro-F1 | MCC | balanced acc | AUC | ECE |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for y in years:
        r = results[y]
        m = r["metrics"]
        def fmt(key):
            v = m.get(key)
            return "—" if v is None or not np.isfinite(v) else f"{v:.4f}"
        train = r["train_years"]
        lines.append(
            f"| {y} | {train[0]}–{train[1]} | {r['val_year']} | {r['test_year']} | "
            f"{r['n_test']} | {fmt('macro_f1')} | {fmt('mcc')} | "
            f"{fmt('balanced_accuracy')} | {fmt('roc_auc_ovr')} | {fmt('ece')} |"
        )
    lines += ["", "Mean ± std across folds:", "",
              "| metric | mean | std | folds |", "|---|---:|---:|---:|"]
    for key in METRIC_KEYS:
        a = summary["aggregate"].get(key)
        if a:
            lines.append(f"| {key} | {a['mean']:.4f} | {a['std']:.4f} | {a['n_folds']} |")
    lines += ["", "The 2025 fold is the primary model's 2025 slice: its train/val/test "
              "boundaries already match the fold definition, so it is not retrained. "
              "Fold results are not used for model selection.", ""]
    (reports_dir / "walk_forward.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: summary["aggregate"].get(k) for k in ("macro_f1", "mcc", "roc_auc_ovr")},
                     indent=2))
    print(f"wrote {reports_dir / 'walk_forward.md'} and {reports_dir / 'walk_forward.json'}")


if __name__ == "__main__":
    main()
