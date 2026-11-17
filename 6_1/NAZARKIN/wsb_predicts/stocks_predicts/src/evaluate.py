#!/usr/bin/env python3
"""Evaluate a trained direction model and run the leakage audit.

Outputs:
  reports/model_metrics.json
  reports/model_predictions.parquet
  reports/ticker_day_predictions.parquet
  reports/leakage_audit.json
  reports/model_card.md
  reports/error_analysis.md
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import (
    DIRECTIONS,
    DIRECTION_TO_ID,
    load_config,
    model_path,
    project_path,
    read_json,
    setup_logging,
    write_json,
)
from features import (
    FeaturePipeline,
    PairDataset,
    load_pipeline,
    make_collate,
    prepare_frame,
)
from metrics import (
    aggregate_ticker_day,
    apply_temperature,
    bootstrap_ci,
    bootstrap_delta,
    compute_metrics,
    economic_backtest,
    expected_calibration_error,
    fit_temperature,
)
from model import DirectionFusionModel
from train import predict_logits

LOG = logging.getLogger("evaluate")


def latest_run() -> Path:
    runs = sorted((project_path("models")).glob("finbert_h5_seed*"))
    if not runs:
        raise SystemExit("no trained run found; run src/train.py first")
    return runs[-1]


def load_run(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    mcfg = ckpt["model_config"]
    pipeline = FeaturePipeline.from_json(ckpt["pipeline"])
    model = DirectionFusionModel(
        encoder_path=model_path(mcfg["encoder_path"]),
        numeric_dim=len(pipeline.numeric_cols),
        n_tickers=len(pipeline.tickers),
        n_flairs=len(pipeline.flairs),
        horizons=ckpt["horizons"],
        hidden_dim=int(mcfg.get("hidden_dim", 128)),
        ticker_dim=int(mcfg.get("ticker_embedding_dim", 16)),
        flair_dim=int(mcfg.get("flair_embedding_dim", 8)),
        dropout=float(mcfg.get("dropout", 0.2)),
        freeze_layers=int(mcfg.get("freeze_layers", 8)),
        use_aux_class=bool(mcfg.get("use_aux_class_loss", True)),
        use_magnitude=bool(mcfg.get("use_magnitude_loss", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(run_dir / "tokenizer"))
    return model, pipeline, mcfg, tokenizer, ckpt


def predict_df(model, df, pipeline, tokenizer, mcfg, horizons, max_len,
               device, batch_size=8) -> dict:
    ds = PairDataset(df, tokenizer, pipeline.numeric_cols, max_len, horizons)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=make_collate(tokenizer),
                        pin_memory=(device.type == "cuda"))
    return predict_logits(model, loader, device, horizons)


def split_frames(cfg, mcfg):
    df = prepare_frame(cfg, mcfg, drop_noise_meme=bool(mcfg.get("drop_noise_meme", False)))
    for split in df["split"].unique():
        if split not in {"train", "val", "test"}:
            df = df[df["split"] != split]
    return df


def per_group_metrics(df, probs, group_col, primary: int, min_n=20) -> dict:
    out = {}
    for key, idx in df.groupby(group_col).groups.items():
        idx = np.asarray(list(idx))
        if len(idx) < min_n:
            continue
        y = df.iloc[idx][f"y_h{primary}"].map(DIRECTION_TO_ID).to_numpy(dtype=int)
        if len(set(y)) < 2:
            continue
        out[str(key)] = compute_metrics(y, probs[idx])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--publish", action="store_true",
                        help="also copy the primary artifacts to reports/ root")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ecfg = load_config(args.eval_config)["evaluation"]
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = project_path("models") / args.run_id if args.run_id else latest_run()
    out_dir = project_path(cfg["paths"]["reports"]) / "runs" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    publish = bool(args.publish)
    LOG.info("run=%s device=%s out=%s publish=%s", run_dir, device, out_dir, publish)
    model, pipeline, mcfg, tokenizer, ckpt = load_run(run_dir, device)
    horizons = ckpt["horizons"]
    primary = int(ckpt["primary_horizon"])
    max_len = int(ckpt["max_len"])

    df = split_frames(cfg, mcfg)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    val = pipeline.transform(val)
    test = pipeline.transform(test)
    LOG.info("val=%s test=%s", len(val), len(test))

    res_val = predict_df(model, val, pipeline, tokenizer, mcfg, horizons, max_len, device)
    res_test = predict_df(model, test, pipeline, tokenizer, mcfg, horizons, max_len, device)
    val_sub = val.iloc[res_val["row_index"]].reset_index(drop=True)
    test_sub = test.iloc[res_test["row_index"]].reset_index(drop=True)

    def direction_ids(series: pd.Series) -> np.ndarray:
        return np.asarray(
            [DIRECTION_TO_ID.get(v, -1) if isinstance(v, str) else -1 for v in series],
            dtype=int,
        )

    # Temperature scaling is fitted once per horizon on validation and applied
    # unchanged to test.  The primary horizon drives the headline metrics.
    temperatures: dict[str, float] = {}
    horizon_metrics: dict[str, dict] = {}
    probs_by_h: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for h in horizons:
        yv = direction_ids(val_sub[f"y_h{h}"]) if f"y_h{h}" in val_sub else np.full(len(val_sub), -1)
        mv = yv >= 0
        T_h = (fit_temperature(res_val[f"logits_h{h}"][mv], yv[mv])
               if int(mv.sum()) else 1.0)
        temperatures[str(h)] = float(T_h)
        yt = direction_ids(test_sub[f"y_h{h}"]) if f"y_h{h}" in test_sub else np.full(len(test_sub), -1)
        mt = yt >= 0
        raw_h = apply_temperature(res_test[f"logits_h{h}"][mt], 1.0)
        cal_h = apply_temperature(res_test[f"logits_h{h}"][mt], T_h)
        # Diagnostic only: the temperature that minimises validation ECE.  It is
        # reported alongside the standard NLL-fitted temperature so the reader
        # can see whether the ECE miss is a fitting-objective artefact.
        grid = np.linspace(0.25, 5.0, 200)
        val_eces = [expected_calibration_error(yv[mv], apply_temperature(res_val[f"logits_h{h}"][mv], t))
                    for t in grid]
        T_ece = float(grid[int(np.argmin(val_eces))]) if int(mv.sum()) else 1.0
        ece_h = apply_temperature(res_test[f"logits_h{h}"][mt], T_ece)
        probs_by_h[h] = (yt[mt], raw_h, cal_h)
        horizon_metrics[str(h)] = {
            "n": int(mt.sum()),
            "temperature": float(T_h),
            "temperature_ece": T_ece,
            "raw": compute_metrics(yt[mt], raw_h),
            "calibrated": compute_metrics(yt[mt], cal_h),
            "calibrated_ece_optimal": compute_metrics(yt[mt], ece_h),
        }
        LOG.info("H=%s n=%s temperature=%.3f", h, int(mt.sum()), T_h)

    y_test, test_probs_raw, test_probs = probs_by_h[primary]
    # Keep only rows with a valid primary-horizon label.
    keep = direction_ids(test_sub[f"y_h{primary}"]) >= 0
    test_sub = test_sub.loc[keep].reset_index(drop=True)
    assert len(test_sub) == len(y_test)
    T = temperatures[str(primary)]
    metrics_raw = horizon_metrics[str(primary)]["raw"]
    metrics = horizon_metrics[str(primary)]["calibrated"]
    dates = test_sub["t0_date"].dt.date.to_numpy()
    ci = {m: bootstrap_ci(y_test, test_probs, dates, metric=m, n_iter=args.bootstrap, seed=42)
          for m in ("macro_f1", "mcc", "accuracy", "auc", "auc_pr_up")}

    # per-year and per-ticker report (primary horizon)
    per_year = {}
    for y, g in test_sub.assign(train_year=test_sub["t0_date"].dt.year).groupby("train_year"):
        idx = g.index.to_numpy()
        yy = y_test[idx]
        if len(set(yy)) > 1:
            per_year[str(int(y))] = compute_metrics(yy, test_probs[idx])
    per_ticker = per_group_metrics(test_sub, test_probs, "ticker", primary, min_n=20)

    # ticker-day aggregate
    td = aggregate_ticker_day(test_sub, test_probs)
    y_td = td["direction_label"].map(DIRECTION_TO_ID).to_numpy(dtype=int)
    td_probs = np.vstack([td["p_DOWN"], td["p_FLAT"], td["p_UP"]]).T
    td_metrics = compute_metrics(y_td, td_probs)
    td_ci = {m: bootstrap_ci(y_td, td_probs, td["t0_date"].dt.date.to_numpy(),
                             metric=m, n_iter=args.bootstrap, seed=42)
             for m in ("macro_f1", "mcc", "accuracy", "auc", "auc_pr_up")}

    econ = {}
    for bps in ecfg["economic"].get("transaction_cost_bps", [5, 10]):
        summary, pnl = economic_backtest(td, transaction_cost_bps=float(bps),
                                         quantile=float(ecfg["economic"].get("quantile", 0.1)),
                                         rebalance=str(ecfg["economic"].get("rebalance", "weekly")))
        econ[str(bps)] = summary
        if bps == 5:
            pnl.to_csv(out_dir / "backtest_pnl.csv", index=False)
    # Random-score portfolio reference
    rng = np.random.default_rng(42)
    rand_rets = []
    for _ in range(int(ecfg["economic"].get("random_portfolios", 100))):
        r = td.copy()
        r["_score"] = rng.permutation(len(r))
        # economic_backtest ranks on p_UP-p_DOWN; encode the random score in p_UP
        r["p_UP"] = r["_score"]
        r["p_DOWN"] = 0.0
        s, _ = economic_backtest(r, transaction_cost_bps=5.0,
                                 quantile=float(ecfg["economic"].get("quantile", 0.1)),
                                 rebalance=str(ecfg["economic"].get("rebalance", "weekly")))
        rand_rets.append(s.get("annualized_return", np.nan))
    rand_arr = np.asarray(rand_rets, dtype=float)
    econ["random_reference"] = {
        "n": int(np.isfinite(rand_arr).sum()),
        "annualized_return_mean": float(np.nanmean(rand_arr)) if np.isfinite(rand_arr).any() else float("nan"),
        "annualized_return_p95": float(np.nanquantile(rand_arr, 0.95)) if np.isfinite(rand_arr).any() else float("nan"),
    }
    # SPY reference over the test ticker-day sample (5-day returns, weekly approximation)
    buy_hold = td.groupby(td["t0_date"].dt.to_period("W-FRI"))["benchmark_return"].mean()
    econ["spy_reference"] = {
        "mean_5d_return": float(td["benchmark_return"].mean()),
        "annualized_approx": float((1 + buy_hold.mean()) ** 52 - 1) if len(buy_hold) else float("nan"),
    }

    # Leakage audit
    shuffled_aucs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y_shuf = rng.permutation(y_test)
        if len(set(y_shuf)) == 3:
            from sklearn.metrics import roc_auc_score
            shuffled_aucs.append(float(roc_auc_score(y_shuf, test_probs,
                                                     multi_class="ovr", average="macro")))
    class_prob_cols = [f"prob_{c}" for c in ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]]
    x_mi = test_sub[class_prob_cols].to_numpy(dtype=float)
    mi = mutual_info_classif(x_mi, y_test, discrete_features=False, random_state=42)
    created = test_sub["created_utc"].to_numpy(dtype=float)
    t0_ts = pd.to_datetime(test_sub["t0_utc"]).astype("int64").to_numpy() / 1e9
    leakage = {
        "created_before_t0_violations": int((created >= t0_ts).sum()),
        "train_test_post_overlap": len(set(df[df.split == "train"]["post_id"]) & set(test_sub["post_id"])),
        "shuffled_label_auc_mean": float(np.mean(shuffled_aucs)) if shuffled_aucs else float("nan"),
        "shuffled_label_auc_std": float(np.std(shuffled_aucs)) if shuffled_aucs else float("nan"),
        "shuffled_label_auc_values": shuffled_aucs,
        "class_probs_mutual_information": {c: float(v) for c, v in zip(
            ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"], mi)},
        "class_probs_mi_sum": float(np.sum(mi)),
        "n_test_posts": int(test_sub["post_id"].nunique()),
        "n_test_ticker_days": int(len(td)),
    }
    # Optional post-hoc audit: the frozen stage-1 classifier used a random
    # (not temporal) 80/20 split, so some direction-test posts were seen while
    # the classifier was trained.  This is auxiliary-label leakage, not
    # direction-label leakage; the clean-subset sensitivity check quantifies it.
    overlap_path = project_path(cfg["paths"]["reports"]) / "classifier_overlap_audit.json"
    if overlap_path.exists():
        overlap = read_json(overlap_path)
        if overlap.get("run_id") == run_dir.name:
            leakage["classifier_training_overlap"] = {
                "status": overlap.get("status"),
                "report": str(overlap_path),
                "rows": overlap.get("rows"),
                "metrics": overlap.get("metrics"),
                "delta_vs_B2_clean": overlap.get("delta_vs_B2_clean"),
                "delta_vs_B2_manual_heldout": overlap.get("delta_vs_B2_manual_heldout"),
                "note": overlap.get("note"),
            }

    # Paired comparison against best baseline (B2/B5) on the same rows.
    baseline_preds = project_path(cfg["paths"]["reports"]) / "baselines_predictions.parquet"
    comparison = {}
    if baseline_preds.exists():
        bp = pd.read_parquet(baseline_preds)
        model_pred = test_sub[["post_id", "ticker", "t0_date"]].copy()
        model_pred["y"] = y_test
        model_pred["p_DOWN"] = test_probs[:, 0]
        model_pred["p_FLAT"] = test_probs[:, 1]
        model_pred["p_UP"] = test_probs[:, 2]
        for name in ["B1", "B2", "B3", "B5", "B6"]:
            b = bp[bp["model"] == name].copy()
            if b.empty:
                continue
            m = model_pred.merge(b[["post_id", "ticker", "t0_date", "p_DOWN", "p_FLAT", "p_UP"]],
                                 on=["post_id", "ticker", "t0_date"], suffixes=("_m", "_b"))
            if len(m) < 20:
                continue
            y = m["y"].to_numpy(dtype=int)
            pa = m[["p_DOWN_m", "p_FLAT_m", "p_UP_m"]].to_numpy()
            pb = m[["p_DOWN_b", "p_FLAT_b", "p_UP_b"]].to_numpy()
            ddates = pd.to_datetime(m["t0_date"]).dt.date.to_numpy()
            comparison[name] = {
                "n_common": int(len(m)),
                "delta_macro_f1": bootstrap_delta(y, pa, pb, ddates, metric="macro_f1",
                                                  n_iter=args.bootstrap, seed=42),
                "delta_mcc": bootstrap_delta(y, pa, pb, ddates, metric="mcc",
                                             n_iter=args.bootstrap, seed=42),
                "delta_auc": bootstrap_delta(y, pa, pb, ddates, metric="auc",
                                             n_iter=args.bootstrap, seed=42),
            }

    # Save predictions
    pred = test_sub[["post_id", "ticker", "t0_date", "created_utc", "title", "flair",
                     "class_label", "influence_weight", "direction_label", "abnormal_return",
                     "raw_return", "benchmark_return", "split"]].copy()
    pred["pred"] = [DIRECTIONS[i] for i in test_probs.argmax(1)]
    pred["p_DOWN"] = test_probs[:, 0]
    pred["p_FLAT"] = test_probs[:, 1]
    pred["p_UP"] = test_probs[:, 2]
    pred["confidence"] = test_probs.max(1)
    pred.to_parquet(out_dir / "model_predictions.parquet", index=False)
    td.to_parquet(out_dir / "ticker_day_predictions.parquet", index=False)
    # The newest run's artifacts are only published to the reports root on
    # request, so an ablation run cannot silently overwrite the reference run.
    if publish:
        pred.to_parquet(project_path(cfg["paths"]["reports"]) / "model_predictions.parquet", index=False)
        td.to_parquet(project_path(cfg["paths"]["reports"]) / "ticker_day_predictions.parquet", index=False)

    report = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature": float(T),
        "temperatures": temperatures,
        "test_rows": int(len(test_sub)),
        "test_groups": int(len(td)),
        "raw_metrics": metrics_raw,
        "calibrated_metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "bootstrap_95ci": ci,
        "ticker_day_metrics": td_metrics,
        "ticker_day_bootstrap_95ci": td_ci,
        "per_year": per_year,
        "per_ticker_top": dict(sorted(per_ticker.items(), key=lambda kv: -kv[1].get("n", 0))[:15]),
        "economic": econ,
        "leakage": leakage,
        "baseline_comparison": comparison,
        "horizons": horizons,
        "max_len": max_len,
        "freeze_layers": int(mcfg.get("freeze_layers", 0)),
    }
    write_json(out_dir / "model_metrics.json", report)
    write_json(out_dir / "leakage_audit.json", leakage)
    if publish:
        write_json(project_path(cfg["paths"]["reports"]) / "model_metrics.json", report)
        write_json(project_path(cfg["paths"]["reports"]) / "leakage_audit.json", leakage)

    # ---------------- model card
    bmetrics_path = project_path(cfg["paths"]["reports"]) / "baselines_metrics.json"
    best_base_name, best_base_metrics = None, {}
    if bmetrics_path.exists():
        bm = json.loads(bmetrics_path.read_text(encoding="utf-8"))
        best_base_name, best_base_metrics = max(
            ((k, v.get("test", {})) for k, v in bm.items() if k != "B0"),
            key=lambda kv: kv[1].get("macro_f1", -1), default=(None, {})
        )
    primary_hm = horizon_metrics[str(primary)]
    ece_nll = primary_hm["calibrated"].get("ece", float("nan"))
    ece_opt = primary_hm["calibrated_ece_optimal"].get("ece", float("nan"))
    cmp_best = comparison.get(best_base_name, {}) if best_base_name else {}
    d_f1 = cmp_best.get("delta_macro_f1", {})
    d_mcc = cmp_best.get("delta_mcc", {})
    auc_target = 0.55
    f1_target_ok = bool(d_f1.get("mean", -1) >= 0.03 and d_f1.get("lo", -1) > 0)
    mcc_target_ok = bool(d_mcc.get("mean", -1) >= 0.03 and d_mcc.get("lo", -1) > 0)

    encoder_cfg = str(mcfg.get("encoder_path", ""))
    encoder_lower = encoder_cfg.lower()
    if "minilm" in encoder_lower:
        encoder_base = "MiniLM-L6-v2 sentence encoder"
    elif "deberta" in encoder_lower:
        encoder_base = "DeBERTa-v3-small encoder"
    else:
        encoder_base = ("FinBERT checkpoint (`ProsusAI/finbert` + 4-class WSB head), "
                        "initialized from the project classifier")
    freeze_layers = int(mcfg.get("freeze_layers", 0))
    if freeze_layers >= model.n_layers:
        encoder_desc = f"frozen {encoder_base}; only the fusion layers and heads are trained"
        freeze_desc = f"all {model.n_layers}/{model.n_layers}"
    elif freeze_layers > 0:
        encoder_desc = f"fine-tuned {encoder_base}; first {freeze_layers} layers frozen"
        freeze_desc = f"{freeze_layers}/{model.n_layers}"
    else:
        encoder_desc = f"fully fine-tuned {encoder_base}"
        freeze_desc = f"0/{model.n_layers}"

    train_cfg = mcfg.get("training", {})
    loss_type = str(train_cfg.get("loss_type", "weighted_ce"))
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    lambda_class = float(train_cfg.get("lambda_class", 0.3))
    if loss_type == "focal":
        loss_desc = f"focal loss (γ={focal_gamma:g}, label smoothing {label_smoothing:g})"
    else:
        loss_desc = ("influence-weighted cross-entropy "
                     f"(label smoothing {label_smoothing:g}, class-balance weights, "
                     f"λ_class={lambda_class:g})")

    card = f"""# Model card — {run_dir.name}

## Overview
- Task: 3-class direction (`DOWN`, `FLAT`, `UP`) for a post x ticker pair and for the ticker-day aggregate.
- Primary horizon: H={primary} trading days; auxiliary heads: H={', '.join(str(h) for h in horizons)}.
- Encoder: {encoder_desc}.
- Fusion: pooled encoder embedding ({model.text_dim}) + numeric/market features + ticker embedding + flair embedding -> 2-layer MLP -> horizon heads.
- Frozen layers: {freeze_desc}; max_len={max_len}.
- Loss: {loss_desc}.
- Influence weights are used **only** as loss weights and ticker-day aggregation weights, never as model inputs.

## Data
- 16,000 WSB posts (2,000/year, 2019–2026); 8,000 manual labels + 8,000 frozen-FinBERT prelabels.
- 4,625 valid post x ticker pairs at H=5; 2,357 ticker-day groups; temporal split with a 10-session embargo.
- Directions are beta-adjusted forward abnormal returns vs SPY; see `data_card.md`.

## Test metrics (H={primary}, {len(test_sub)} rows)
| metric | raw | calibrated | 95% block-bootstrap CI (calibrated) |
|---|---:|---:|---|
| Macro-F1 | {metrics_raw.get('macro_f1', float('nan')):.4f} | {metrics.get('macro_f1', float('nan')):.4f} | [{ci['macro_f1']['lo']:.4f}, {ci['macro_f1']['hi']:.4f}] |
| MCC | {metrics_raw.get('mcc', float('nan')):.4f} | {metrics.get('mcc', float('nan')):.4f} | [{ci['mcc']['lo']:.4f}, {ci['mcc']['hi']:.4f}] |
| AUC OvR | {metrics_raw.get('roc_auc_ovr', float('nan')):.4f} | {metrics.get('roc_auc_ovr', float('nan')):.4f} | [{ci['auc']['lo']:.4f}, {ci['auc']['hi']:.4f}] |
| AUC-PR UP | {metrics_raw.get('auc_pr_up', float('nan')):.4f} | {metrics.get('auc_pr_up', float('nan')):.4f} | [{ci['auc_pr_up']['lo']:.4f}, {ci['auc_pr_up']['hi']:.4f}] |
| Brier | {metrics_raw.get('brier', float('nan')):.4f} | {metrics.get('brier', float('nan')):.4f} | — |
| ECE | {metrics_raw.get('ece', float('nan')):.4f} | {ece_nll:.4f} | — |

Ticker-day aggregate: macro-F1 {td_metrics.get('macro_f1', float('nan')):.4f}, MCC {td_metrics.get('mcc', float('nan')):.4f}, AUC {td_metrics.get('roc_auc_ovr', float('nan')):.4f}.

## Per-horizon metrics (calibrated)
| H | n | macro-F1 | MCC | AUC | ECE |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {h} | {hm['n']} | {hm['calibrated'].get('macro_f1', float('nan')):.4f} | "
        f"{hm['calibrated'].get('mcc', float('nan')):.4f} | "
        f"{hm['calibrated'].get('roc_auc_ovr', float('nan')):.4f} | "
        f"{hm['calibrated'].get('ece', float('nan')):.4f} |"
        for h, hm in horizon_metrics.items()
    ) + f"""

## Baseline comparison
Best baseline by test macro-F1: **{best_base_name or 'n/a'}** (macro-F1 {best_base_metrics.get('macro_f1', float('nan')):.4f}, MCC {best_base_metrics.get('mcc', float('nan')):.4f}, AUC {best_base_metrics.get('roc_auc_ovr', float('nan')):.4f}).

Paired block-bootstrap differences on the {cmp_best.get('n_common', 0)} common test rows (model − baseline):
- Δ macro-F1 = **{d_f1.get('mean', float('nan')):+.4f}** [{d_f1.get('lo', float('nan')):+.4f}, {d_f1.get('hi', float('nan')):+.4f}]
- Δ MCC = **{d_mcc.get('mean', float('nan')):+.4f}** [{d_mcc.get('lo', float('nan')):+.4f}, {d_mcc.get('hi', float('nan')):+.4f}]

## Acceptance checks (TZ §8.2)
| criterion | target | result | status |
|---|---|---:|---|
| Test macro-F1 vs best baseline | ≥ +0.03 and CI > 0 | {d_f1.get('mean', float('nan')):+.4f} | {'PASS' if f1_target_ok else 'FAIL'} |
| Test MCC vs best baseline | ≥ +0.03 and CI > 0 | {d_mcc.get('mean', float('nan')):+.4f} | {'PASS' if mcc_target_ok else 'FAIL'} |
| H=5 AUC-ROC | ≥ {auc_target:.2f} | {metrics.get('roc_auc_ovr', float('nan')):.4f} | {'PASS' if metrics.get('roc_auc_ovr', 0) >= auc_target else 'FAIL'} |
| ECE after temperature scaling | ≤ 0.05 | {ece_nll:.4f} | {'PASS' if ece_nll <= 0.05 else 'FAIL'} |
| Shuffled-label AUC | ≈ 0.5 | {leakage['shuffled_label_auc_mean']:.3f} | {'PASS' if abs(leakage['shuffled_label_auc_mean'] - 0.5) < 0.05 else 'FAIL'} |

**Result: the neural model does not clear both the +0.03 MCC and +0.03 macro-F1 requirements over the best baseline.**
This is reported as a negative result. The test set was not used for model selection or calibration;
the paired bootstrap intervals for the best-baseline comparison are reported above. The ECE-optimal scalar temperature on
validation (T={primary_hm['temperature_ece']:.3f}) gives test ECE {ece_opt:.4f}; the miss is small
but not zero. The weighted-CE objective was compared with focal loss (γ=1.5) under the same
protocol; focal did not improve test macro-F1 or MCC (`reports/ablations.md`). See
`error_analysis.md` and `reports/ablations.md` for the analysis.

## Intended use
Research and educational analysis of social-media sentiment. Not investment advice, not a trading system, not causal evidence that a post moves a stock.

## Limitations
- Direction labels are noisy and associative, not causal.
- New-post class labels are classifier prelabels until the human review queue is completed; the classifier-vs-manual holdout Cohen's κ is 0.573 [0.540, 0.605], below the 0.65 target.
- 8,000 new posts are sampled from the local Arctic Shift pool; 2026 is partial.
- The 30-ticker universe is the TZ §4.2 fixed list (including SPY/QQQ); it suffers from survivorship/selection bias.
- Only 22.9% of posts mention a universe ticker; the rest cannot enter the pair dataset.
- Fine-tuning a 6 GB GPU model for 3 epochs is a compute-budget compromise. Seeds 42/43/44 give test macro-F1 0.4275 ± 0.0140 and MCC 0.1511 ± 0.0109; the negative result is not specific to seed 42.
- The ticker-day aggregate is a fixed influence-weighted average, not a learned attention pool.
- Reddit `score`/`num_comments` and the document text are Arctic Shift snapshot values. We cannot reconstruct the values exactly at `t0`, so votes/edits after publication are a residual leakage risk that this dataset cannot eliminate.
- The stage-1 classifier used a random (not temporal) 80/20 split over all years, so 373/463 manual test posts (80.6%) were seen during its training. Class probabilities are inputs, making this auxiliary-label leakage. The clean-subset check still gives macro-F1 0.4384 / MCC 0.1642 and misses the +0.03 margin (Δ vs B2 +0.0081 / +0.0193); the stricter manual-only held-out subset (121 rows / 90 posts) gives Δ −0.0033 / −0.0022. See `classifier_overlap_audit.json`.
- Test ECE is slightly above 0.05; the calibration shift between 2024 and 2025–2026 is documented.
- Encoder and max-length comparisons (`max_len=512`, frozen MiniLM, frozen DeBERTa-v3-small) are single-seed; they are reported in `reports/ablations.md`, not used for model selection.
- Early stopping on validation MCC can select a class-collapsed checkpoint. The frozen DeBERTa comparison did exactly that (test UP recall 0.038), so macro-F1 and per-class recall are reported alongside MCC.

## Ethical / legal
Reddit data are used for research; author names are hashed. Respect Reddit/Arctic Shift terms of use. Do not use the output as personalized financial advice.
"""
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")
    if publish:
        (project_path(cfg["paths"]["reports"]) / "model_card.md").write_text(card, encoding="utf-8")

    # ---------------- error analysis
    cm = metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    err = test_sub.copy()
    err["y"] = [DIRECTIONS[i] for i in y_test]
    err["pred"] = [DIRECTIONS[i] for i in test_probs.argmax(1)]
    err["confidence"] = test_probs.max(1)
    err["correct"] = err["y"] == err["pred"]
    wrong = err[~err["correct"]].sort_values("confidence", ascending=False).head(25)
    lines = [f"# Error analysis — {run_dir.name}", "",
             "## Confusion matrix (rows=true DOWN/FLAT/UP, cols=pred)", "",
             "| true \\ pred | DOWN | FLAT | UP |", "|---|---:|---:|---:|"]
    for i, name in enumerate(DIRECTIONS):
        lines.append(f"| {name} | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")
    lines += ["", "## Per-year macro-F1", "", "| year | n | macro-F1 | MCC |", "|---|---:|---:|---:|"]
    for y, m in sorted(per_year.items()):
        lines.append(f"| {y} | {m['n']} | {m['macro_f1']:.4f} | {m['mcc']:.4f} |")
    lines += ["", "## Most confident errors", "",
              "| post | ticker | true | pred | confidence | title |", "|---|---|---|---|---:|---|"]
    for _, r in wrong.iterrows():
        title = str(r["title"]).replace("|", " ")[:90]
        lines.append(f"| {r['post_id']} | {r['ticker']} | {r['y']} | {r['pred']} | {r['confidence']:.3f} | {title} |")
    lines += ["", "## Calibration", "",
              f"Temperature fitted on validation: **{T:.3f}**.",
              f"ECE before calibration: **{metrics_raw.get('ece', float('nan')):.4f}**, after: **{metrics.get('ece', float('nan')):.4f}**.", "",
              "## Leakage audit", "",
              f"- Posts with `created_utc >= t0`: **{leakage['created_before_t0_violations']}**",
              f"- Post IDs shared between train and test: **{leakage['train_test_post_overlap']}**",
              f"- Shuffled-label AUC: **{leakage['shuffled_label_auc_mean']:.3f} ± {leakage['shuffled_label_auc_std']:.3f}** (target ≈ 0.5)",
              f"- Sum of mutual information between class probabilities and direction: **{leakage['class_probs_mi_sum']:.4f}**", "",
              "## Economic sanity check (ticker-day, weekly rebalance)", "",
              f"- Long-short annualized return (5 bps): **{econ.get('5', {}).get('annualized_return', float('nan')):.2%}**",
              f"- Sharpe (5 bps): **{econ.get('5', {}).get('sharpe', float('nan')):.2f}**",
              f"- Max drawdown: **{econ.get('5', {}).get('max_drawdown', float('nan')):.2%}**",
              f"- SPY mean 5-day return over the same ticker-days: **{econ['spy_reference']['mean_5d_return']:.2%}**", "",
              "The economic result is a sanity check, not an acceptance criterion. Overlapping forward windows, "
              "small samples and the absence of a full execution model make it easy to overfit."]
    (out_dir / "error_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if publish:
        (project_path(cfg["paths"]["reports"]) / "error_analysis.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "run_id": run_dir.name,
        "temperature": T,
        "macro_f1": metrics.get("macro_f1"),
        "mcc": metrics.get("mcc"),
        "auc": metrics.get("roc_auc_ovr"),
        "ticker_day_macro_f1": td_metrics.get("macro_f1"),
        "ticker_day_mcc": td_metrics.get("mcc"),
        "ece": metrics.get("ece"),
        "bootstrap": ci,
        "baseline_comparison": comparison,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
