#!/usr/bin/env python3
"""Run the seed and ablation matrix, then evaluate each checkpoint.

This is intentionally a thin, resumable driver around ``src/train.py`` and
``src/evaluate.py``.  It does not tune anything on the test set: every run uses
the same config, and the only varying inputs are the pre-registered ablation
flags from ``configs/model.yaml``.

Usage:
    python src/run_experiments.py --preset core --bootstrap 1000
    python src/run_experiments.py --preset extra
    python src/run_experiments.py --preset core --dry-run
    python src/run_experiments.py --preset core --only ablate_no_aux
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common import load_config, project_path, setup_logging  # noqa: E402

LOG = logging.getLogger("experiments")


def core_plan(cfg: dict) -> list[dict]:
    ablations = cfg.get("ablations", {})
    plan = [
        {"run_id": "finbert_h5_seed43", "seed": 43, "kind": "seed", "overrides": {}},
        {"run_id": "finbert_h5_seed44", "seed": 44, "kind": "seed", "overrides": {}},
    ]
    for name, overrides in ablations.items():
        plan.append({
            "run_id": f"ablate_{name}_seed42",
            "seed": 42,
            "kind": "ablation",
            "name": name,
            "overrides": dict(overrides),
        })
    return plan


def extra_plan() -> list[dict]:
    return [
        {
            "run_id": "finbert_h5_maxlen512_seed42",
            "seed": 42,
            "kind": "encoder_or_length",
            "name": "max_len_512",
            "overrides": {"max_len": 512},
        },
        {
            "run_id": "minilm_frozen_h5_seed42",
            "seed": 42,
            "kind": "encoder_or_length",
            "name": "minilm_frozen",
            "overrides": {
                "encoder_path": "sentence-transformers/all-MiniLM-L6-v2",
                "freeze_layers": 12,
            },
        },
    ]


def run(cmd: list[str]) -> None:
    LOG.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["core", "extra", "all"], default="core")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true", help="retrain even if a checkpoint exists")
    parser.add_argument("--eval-only", action="store_true", help="skip training")
    parser.add_argument("--only", default="",
                        help="comma-separated substring filters on the run id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config("configs/model.yaml")
    plan = []
    if args.preset in ("core", "all"):
        plan += core_plan(cfg)
    if args.preset in ("extra", "all"):
        plan += extra_plan()
    if args.only:
        filters = [f.strip() for f in args.only.split(",") if f.strip()]
        plan = [p for p in plan if any(f in p["run_id"] for f in filters)]
    if not plan:
        raise SystemExit("empty experiment plan")

    reports = project_path("reports")
    models = project_path("models")
    summary = []
    for item in plan:
        run_id = item["run_id"]
        model_dir = models / run_id
        metrics_path = reports / "runs" / run_id / "model_metrics.json"
        # training_summary.json is written only after the last epoch; a
        # checkpoint without it may be from an interrupted run.
        checkpoint_ok = (model_dir / "model.pt").exists() and (
            model_dir / "training_summary.json").exists()
        LOG.info("=== %s (%s) ===", run_id, item["kind"])
        if args.dry_run:
            print(json.dumps(item, indent=2))
            continue
        start = time.time()
        if metrics_path.exists() and not args.force:
            LOG.info("skip %s: metrics already exist", run_id)
            summary.append({"run_id": run_id, "status": "skipped"})
            continue

        if not args.eval_only and (args.force or not checkpoint_ok):
            cmd = [sys.executable, "src/train.py", "--seed", str(item["seed"]),
                   "--run-id", run_id]
            for key, value in item["overrides"].items():
                cmd += ["--override", f"{key}={json.dumps(value)}"]
            try:
                run(cmd)
            except subprocess.CalledProcessError as exc:
                LOG.error("training failed for %s: %s", run_id, exc)
                summary.append({"run_id": run_id, "status": "train_failed",
                                "kind": item["kind"], "name": item.get("name")})
                continue
        else:
            LOG.info("reuse checkpoint %s", model_dir / "model.pt")

        try:
            run([sys.executable, "src/evaluate.py", "--run-id", run_id,
                 "--bootstrap", str(args.bootstrap)])
        except subprocess.CalledProcessError as exc:
            LOG.error("evaluation failed for %s: %s", run_id, exc)
            summary.append({"run_id": run_id, "status": "eval_failed",
                            "kind": item["kind"], "name": item.get("name")})
            continue
        summary.append({
            "run_id": run_id,
            "status": "done",
            "kind": item["kind"],
            "name": item.get("name"),
            "seconds": round(time.time() - start, 1),
        })
        LOG.info("done %s in %.1f min", run_id, (time.time() - start) / 60)

    out = reports / "experiment_runs.json"
    if args.dry_run:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    # Several driver processes may run in parallel on the same GPU.  Lock the
    # registry so a read-modify-write from one worker cannot drop another
    # worker's freshly written entries.
    with open(out, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        raw = fh.read()
        try:
            existing = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError:
            existing = []
        by_id = {x["run_id"]: x for x in existing}
        for item in summary:
            by_id[item["run_id"]] = item
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(list(by_id.values()), indent=2))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
