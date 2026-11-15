#!/usr/bin/env python3
"""Train the WSB direction fusion model.

The model consumes the frozen/tuned FinBERT encoder, numeric post and market
features, a ticker embedding and a flair embedding.  It has one direction head
per horizon plus an auxiliary post-class head.  The influence weight is used
only as a sample weight in the loss (and later in ticker-day aggregation), never
as an input feature.

Usage:
    python src/train.py --seed 42
    python src/train.py --seed 42 --max-len 512 --run-id finbert512
    python src/train.py --quick --limit 200
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from common import (
    CLASSES,
    DIRECTION_TO_ID,
    ID_TO_DIRECTION,
    ensure_dirs,
    load_config,
    model_path,
    project_path,
    set_seed,
    setup_logging,
    write_json,
)
from features import (
    FeaturePipeline,
    PairDataset,
    load_pipeline,
    make_collate,
    numeric_columns,
    prepare_frame,
    save_pipeline,
)
from model import DirectionFusionModel

LOG = logging.getLogger("train")


def parse_override(values: list[str]) -> dict:
    out = {}
    for item in values:
        if "=" not in item:
            raise SystemExit(f"override must be key=value: {item}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def make_optimizer(model: DirectionFusionModel, cfg: dict):
    head_lr = float(cfg["training"]["head_lr"])
    enc_lr = float(cfg["training"]["encoder_lr"])
    wd = float(cfg["training"]["weight_decay"])
    groups = [{"params": list(model.head_parameters()), "lr": head_lr, "weight_decay": wd}]
    enc_params = list(model.encoder_parameters())
    if enc_params:
        groups.append({"params": enc_params, "lr": enc_lr, "weight_decay": wd})
    return torch.optim.AdamW(groups)


def sample_weights(batch: dict, h: int, pipeline: FeaturePipeline,
                   model_cfg: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    y = batch[f"y_h{h}"].to(device)
    mask = y >= 0
    influence = batch["influence"].to(device)
    if not model_cfg.get("loss_use_influence_weight", True):
        influence = torch.ones_like(influence)
    if model_cfg.get("use_engagement_weight", False):
        score = batch["score"].to(device).clamp(min=0)
        engage = torch.clamp(1.0 + 0.3 * torch.log10(1.0 + score), max=2.0)
        influence = influence * engage
    balance = torch.ones_like(influence)
    weights = pipeline.direction_weights.get(str(h), [1.0, 1.0, 1.0])
    if y.numel():
        balance = torch.tensor([weights[int(v)] if int(v) >= 0 else 1.0 for v in y.cpu()],
                               dtype=torch.float32, device=device)
    w = influence * balance
    return w, mask


def direction_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor, mask: torch.Tensor,
                   label_smoothing: float, focal_gamma: float) -> tuple[torch.Tensor, int]:
    if mask.sum() == 0:
        return logits.sum() * 0.0, 0
    logits = logits[mask]
    y = y[mask]
    w = w[mask]
    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
    if focal_gamma and focal_gamma > 0:
        p = torch.softmax(logits, dim=-1)
        pt = p.gather(1, y.unsqueeze(1)).squeeze(1).clamp(min=1e-6)
        ce = (1.0 - pt).pow(focal_gamma) * ce
    loss = (ce * w).sum() / w.sum().clamp(min=1e-8)
    return loss, int(mask.sum())


def forward_batch(model: DirectionFusionModel, batch: dict, device: torch.device) -> dict:
    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        numeric=batch["numeric"].to(device),
        ticker_id=batch["ticker_id"].to(device),
        flair_id=batch["flair_id"].to(device),
    )


def train_epoch(model, loader, optimizer, scheduler, scaler, pipeline, model_cfg,
                device, grad_accum: int, use_amp: bool) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = forward_batch(model, batch, device)
            loss = torch.zeros((), device=device)
            primary_h = int(model_cfg.get("primary_horizon", 5))
            for h in model.horizons:
                if f"y_h{h}" not in batch:
                    continue
                y = batch[f"y_h{h}"].to(device)
                w, mask = sample_weights(batch, h, pipeline, model_cfg, device)
                w_loss, n = direction_loss(
                    out["logits"][f"h{h}"], y, w, mask,
                    float(model_cfg["training"].get("label_smoothing", 0.0)),
                    float(model_cfg["training"].get("focal_gamma", 0.0))
                    if model_cfg["training"].get("loss_type") == "focal" else 0.0,
                )
                horizon_weight = 1.0 if h == primary_h else 0.5
                loss = loss + horizon_weight * w_loss
            if model.aux_class is not None and model_cfg.get("use_aux_class_loss", True):
                aux = F.cross_entropy(out["class_logits"], batch["class_id"].to(device))
                loss = loss + float(model_cfg["training"]["lambda_class"]) * aux
            if model.magnitude is not None and model_cfg.get("use_magnitude_loss", False):
                mag = F.smooth_l1_loss(out["magnitude"], batch["magnitude"].to(device))
                loss = loss + float(model_cfg["training"]["lambda_magnitude"]) * mag
            loss = loss / grad_accum
        scaler.scale(loss).backward()
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           float(model_cfg["training"]["max_grad_norm"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * grad_accum
        n_batches += 1
    # flush leftover accumulation
    if n_batches and n_batches % grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       float(model_cfg["training"]["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / max(1, n_batches)}


@torch.inference_mode()
def predict_logits(model, loader, device, horizons: list[int]) -> dict[str, np.ndarray]:
    model.eval()
    out_logits = {h: [] for h in horizons}
    out_class = []
    out_mag = []
    out_idx = []
    for batch in loader:
        out = forward_batch(model, batch, device)
        for h in horizons:
            out_logits[h].append(out["logits"][f"h{h}"].float().cpu().numpy())
        if "class_logits" in out:
            out_class.append(out["class_logits"].float().cpu().numpy())
        if "magnitude" in out:
            out_mag.append(out["magnitude"].float().cpu().numpy())
        out_idx.append(batch["row_index"].numpy())
    result = {f"logits_h{h}": np.vstack(v) if v else np.empty((0, 3)) for h, v in out_logits.items()}
    result["row_index"] = np.concatenate(out_idx) if out_idx else np.array([], dtype=int)
    if out_class:
        result["class_logits"] = np.vstack(out_class)
    if out_mag:
        result["magnitude"] = np.concatenate(out_mag)
    return result


def evaluate_loader(model, loader, device, df: pd.DataFrame, horizons: list[int],
                    primary_h: int) -> dict:
    res = predict_logits(model, loader, device, horizons)
    idx = res["row_index"]
    sub = df.iloc[idx].reset_index(drop=True)
    metrics = {}
    y = sub.get(f"y_h{primary_h}")
    probs = torch.softmax(torch.tensor(res[f"logits_h{primary_h}"]), dim=-1).numpy()
    valid = y.notna().to_numpy() if y is not None else np.zeros(len(sub), bool)
    if valid.sum() and len(probs) == len(y):
        yv = y[valid].map(DIRECTION_TO_ID).to_numpy(dtype=int)
        pv = probs[valid]
        pred = pv.argmax(1)
        metrics["n"] = int(valid.sum())
        metrics["macro_f1"] = float(f1_score(yv, pred, average="macro", zero_division=0))
        metrics["mcc"] = float(matthews_corrcoef(yv, pred)) if len(set(yv)) > 1 else 0.0
        metrics["accuracy"] = float((yv == pred).mean())
    return metrics


def train_one(config_path: str, seed: int, run_id: str | None, overrides: dict,
              limit: int | None, quick: bool) -> Path:
    cfg = load_config(config_path)
    mcfg_all = load_config("configs/model.yaml")
    mcfg = copy.deepcopy(mcfg_all)
    mcfg.update(copy.deepcopy(mcfg_all["model"]))
    mcfg["training"] = copy.deepcopy(mcfg_all["training"])
    # flatten the model section so existing code can read model.* keys directly
    for k, v in mcfg_all.get("model", {}).items():
        mcfg[k] = v
    for k, v in overrides.items():
        if k in mcfg:
            mcfg[k] = v
        elif k in mcfg["training"]:
            mcfg["training"][k] = v
        else:
            mcfg[k] = v
    if quick:
        mcfg["training"]["epochs"] = 1
        mcfg["freeze_layers"] = 12
    cfg["model_overrides"] = overrides
    horizons = list(mcfg["training"]["horizons"])
    primary_h = int(mcfg["training"]["primary_horizon"])
    max_len = int(mcfg.get("max_len", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mcfg["training"].get("device") not in (None, "auto"):
        device = torch.device(mcfg["training"]["device"])
    set_seed(seed)
    LOG.info("seed=%s device=%s max_len=%s horizons=%s", seed, device, max_len, horizons)

    df = prepare_frame(cfg, mcfg, drop_noise_meme=bool(mcfg.get("drop_noise_meme", False)))
    if limit:
        # keep the temporal structure: take the first `limit` training rows plus
        # a small validation slice; useful only for smoke tests.
        train = df[df["split"] == "train"].head(limit)
        val = df[df["split"] == "val"].head(max(20, limit // 5))
        df = pd.concat([train, val], ignore_index=True)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty:
        raise SystemExit("empty train/val split")
    LOG.info("rows: train=%s val=%s test=%s", len(train_df), len(val_df), len(test_df))

    tickers_cfg = load_config(cfg["tickers"]["config"])
    pipeline = FeaturePipeline(numeric_columns(mcfg)).fit(train_df, tickers_cfg["universe"])
    train_df = pipeline.transform(train_df)
    val_df = pipeline.transform(val_df)
    test_df = pipeline.transform(test_df)

    tokenizer = AutoTokenizer.from_pretrained(model_path(mcfg["encoder_path"]))
    train_ds = PairDataset(train_df, tokenizer, pipeline.numeric_cols, max_len, horizons,
                           use_engagement=bool(mcfg.get("use_engagement_weight", False)),
                           direction_weights=pipeline.direction_weights)
    val_ds = PairDataset(val_df, tokenizer, pipeline.numeric_cols, max_len, horizons,
                         use_engagement=bool(mcfg.get("use_engagement_weight", False)),
                         direction_weights=pipeline.direction_weights)
    collate = make_collate(tokenizer)
    bs = int(mcfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0,
                              collate_fn=collate, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0,
                            collate_fn=collate, pin_memory=(device.type == "cuda"))

    model = DirectionFusionModel(
        encoder_path=model_path(mcfg["encoder_path"]),
        numeric_dim=len(pipeline.numeric_cols),
        n_tickers=len(pipeline.tickers),
        n_flairs=len(pipeline.flairs),
        horizons=horizons,
        hidden_dim=int(mcfg.get("hidden_dim", 128)),
        ticker_dim=int(mcfg.get("ticker_embedding_dim", 16)),
        flair_dim=int(mcfg.get("flair_embedding_dim", 8)),
        dropout=float(mcfg.get("dropout", 0.2)),
        freeze_layers=int(mcfg.get("freeze_layers", 8)),
        use_aux_class=bool(mcfg.get("use_aux_class_loss", True)),
        use_magnitude=bool(mcfg.get("use_magnitude_loss", False)),
    ).to(device)
    if mcfg["training"].get("gradient_checkpointing", False) and int(mcfg.get("freeze_layers", 0)) < 12:
        model.encoder.gradient_checkpointing_enable()
        model.encoder.config.use_cache = False

    optimizer = make_optimizer(model, mcfg)
    epochs = int(mcfg["training"]["epochs"])
    grad_accum = int(mcfg["training"]["grad_accumulation"])
    total_steps = max(1, (len(train_loader) * epochs) // max(1, grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(mcfg["training"].get("warmup_steps", 80)),
        num_training_steps=total_steps,
    )
    use_amp = bool(mcfg["training"].get("fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"finbert_h{primary_h}_seed{seed}_{stamp}"
    run_dir = project_path("models") / run_id
    ensure_dirs(run_dir)
    save_pipeline(run_dir / "feature_pipeline.json", pipeline)
    write_json(run_dir / "run_config.json", {
        "run_id": run_id,
        "seed": seed,
        "config_path": config_path,
        "data_config": cfg,
        "model_config": mcfg,
        "overrides": overrides,
        "device": str(device),
        "created_utc": stamp,
    })
    # keep the tokenizer files next to the run for a self-contained service
    tokenizer.save_pretrained(run_dir / "tokenizer")

    best_mcc = -2.0
    best_epoch = -1
    history = []
    patience = int(mcfg["training"].get("patience", 2))
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scheduler, scaler, pipeline,
                         mcfg, device, grad_accum, use_amp)
        va = evaluate_loader(model, val_loader, device, val_df, horizons, primary_h)
        entry = {"epoch": epoch, "train": tr, "val": va, "seconds": round(time.time() - t0, 1)}
        history.append(entry)
        LOG.info("epoch %s: train_loss=%.4f val=%s", epoch, tr["loss"], va)
        if va.get("mcc", -2) > best_mcc:
            best_mcc = float(va.get("mcc", -2))
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "pipeline": pipeline.to_json(),
                "model_config": mcfg,
                "horizons": horizons,
                "primary_horizon": primary_h,
                "max_len": max_len,
                "seed": seed,
                "epoch": epoch,
            }, run_dir / "model.pt")
        if epoch - best_epoch >= patience:
            LOG.info("early stopping at epoch %s (best %s)", epoch, best_epoch)
            break

    # Save last state as well, plus a small JSON summary.
    torch.save({
        "model_state_dict": model.state_dict(),
        "pipeline": pipeline.to_json(),
        "model_config": mcfg,
        "horizons": horizons,
        "primary_horizon": primary_h,
        "max_len": max_len,
        "seed": seed,
        "epoch": epochs,
        "is_last": True,
    }, run_dir / "model_last.pt")
    summary = {
        "run_id": run_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_mcc": best_mcc,
        "history": history,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "horizons": horizons,
        "primary_horizon": primary_h,
        "max_len": max_len,
        "freeze_layers": int(mcfg.get("freeze_layers", 0)),
        "n_numeric_features": len(pipeline.numeric_cols),
        "encoders": mcfg.get("encoder_path"),
        "created_utc": stamp,
    }
    write_json(run_dir / "training_summary.json", summary)
    LOG.info("saved %s", run_dir)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--override", action="append", default=[],
                        help="model/training override key=value (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None,
                        help="override the encoder token limit (256 or 512)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    setup_logging()
    overrides = parse_override(args.override)
    if args.max_len:
        overrides["max_len"] = int(args.max_len)
    train_one(args.config, args.seed, args.run_id, overrides,
              args.limit, args.quick)


if __name__ == "__main__":
    main()
