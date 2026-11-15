#!/usr/bin/env python3
"""Feature engineering and torch datasets for the direction model."""
from __future__ import annotations

import collections
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from common import (
    CLASSES,
    CLASS_TO_ID,
    DIRECTION_TO_ID,
    INFLUENCE_WEIGHTS,
    load_config,
    project_path,
)

ET = ZoneInfo("America/New_York")

MARKET_NUMERIC = [
    "bench_ret_5", "bench_ret_20", "bench2_ret_5", "bench2_ret_20",
    "ticker_ret_5", "ticker_ret_20", "ticker_vol_20", "beta", "threshold",
]
BASE_NUMERIC = [
    "log1p_score", "log1p_comments", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_premarket", "is_afterhours", "mention_count", "weight_share", "confidence",
]
AGG_NUMERIC = [
    "agg_n_posts", "agg_mean_score", "agg_max_score",
    "agg_class_ANALYSIS_DD", "agg_class_FINANCIAL_CLAIM",
    "agg_class_GENUINE_OPINION", "agg_class_NOISE_MEME",
]


def load_long_dataset(cfg: dict) -> pd.DataFrame:
    path = project_path(cfg["paths"]["dataset"])
    if not path.exists():
        raise SystemExit(
            f"dataset not found: {path}. Run `python src/build_dataset.py --all-horizons` first."
        )
    df = pd.read_parquet(path)
    df["t0_date"] = pd.to_datetime(df["t0_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    return df


def build_wide_frame(df: pd.DataFrame, primary_horizon: int) -> pd.DataFrame:
    """One row per (post, ticker), with all horizon labels as columns."""
    key = ["post_id", "ticker", "t0_date"]
    base = df[df["horizon_days"] == primary_horizon].copy()
    if base.empty:
        raise SystemExit(f"no rows for primary horizon {primary_horizon}")
    labels = df.pivot_table(
        index=key, columns="horizon_days", values="direction_label", aggfunc="first"
    )
    labels.columns = [f"y_h{int(c)}" for c in labels.columns]
    returns = df.pivot_table(
        index=key, columns="horizon_days", values="abnormal_return", aggfunc="first"
    )
    returns.columns = [f"abn_h{int(c)}" for c in returns.columns]
    base = base.set_index(key).join(labels).join(returns).reset_index()
    return base


def _time_features(created_utc: pd.Series, t0_date: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=created_utc.index)
    et = pd.to_datetime(created_utc, unit="s", utc=True).dt.tz_convert(ET)
    hour = et.dt.hour + et.dt.minute / 60.0
    dow = et.dt.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_premarket"] = ((hour >= 4.0) & (hour < 9.5)).astype(float)
    out["is_afterhours"] = ((hour >= 16.0) & (hour < 20.0)).astype(float)
    return out


def add_aggregate_features(base: pd.DataFrame) -> pd.DataFrame:
    """Features of all linked posts for the same ticker and entry day.

    The group is formed *before* the entry open and contains only posts that are
    already in the dataset.  No direction label or future price is used.
    """
    key = ["ticker", "t0_date"]
    g = base.groupby(key)
    agg = g.agg(
        agg_n_posts=("post_id", "nunique"),
        agg_mean_score=("score", "mean"),
        agg_max_score=("score", "max"),
    ).reset_index()
    for c in CLASSES:
        if f"prob_{c}" in base.columns:
            tmp = (
                base.assign(_p=base["prob_" + c])
                .groupby(key)["_p"]
                .mean()
                .rename(f"agg_class_{c}")
                .reset_index()
            )
            agg = agg.merge(tmp, on=key, how="left")
    return base.merge(agg, on=key, how="left")


def prepare_frame(cfg: dict, model_cfg: dict, drop_noise_meme: bool = False) -> pd.DataFrame:
    primary_h = int(cfg["primary_horizon"])
    df = load_long_dataset(cfg)
    base = build_wide_frame(df, primary_h)
    if drop_noise_meme:
        base = base[base["class_label"] != "NOISE_MEME"].copy()
    base = base[~base["purged"].astype(bool)].copy()
    base = base.sort_values(["t0_date", "ticker", "post_id"]).reset_index(drop=True)

    # Text fields.  The fragment is put before the body so truncation keeps
    # the ticker-specific context.
    base["text_plain"] = (base["title"].fillna("") + "\n\n" + base["body"].fillna("")).str.strip()
    if model_cfg.get("use_ticker_fragment", True):
        base["text_model"] = (
            "$" + base["ticker"].astype(str) + " [SEP] " + base["title"].fillna("")
            + " [SEP] " + base["fragment"].fillna("")
            + " [SEP] " + base["body"].fillna("")
        )
    else:
        base["text_model"] = "$" + base["ticker"].astype(str) + " [SEP] " + base["text_plain"]

    base["log1p_score"] = np.log1p(base["score"].clip(lower=0))
    base["log1p_comments"] = np.log1p(base["num_comments"].clip(lower=0))
    tf = _time_features(base["created_utc"], base["t0_date"])
    for c in tf.columns:
        base[c] = tf[c].values
    if model_cfg.get("use_aggregate_features", True):
        base = add_aggregate_features(base)
    else:
        for c in AGG_NUMERIC:
            base[c] = 0.0

    for c in MARKET_NUMERIC:
        if c not in base.columns:
            base[c] = np.nan
    if not model_cfg.get("use_market_features", True):
        for c in MARKET_NUMERIC:
            base[c] = 0.0

    if not model_cfg.get("use_class_probs", True):
        for c in CLASSES:
            base[f"prob_{c}"] = 0.0
        base["confidence"] = 0.0
    base["class_id"] = base["class_label"].map(CLASS_TO_ID).fillna(0).astype(int)
    # Ticker and flair categorical maps are filled by the caller.
    base["ticker_id"] = 0
    base["flair_id"] = 0
    base["magnitude"] = base[f"abn_h{primary_h}"].abs().fillna(0.0)
    for h in [1, 3, 5, 10]:
        col = f"y_h{h}"
        if col not in base.columns:
            base[col] = None
    base["label_h5"] = base[f"y_h{primary_h}"]
    return base


def numeric_columns(model_cfg: dict) -> list[str]:
    cols = list(BASE_NUMERIC)
    if model_cfg.get("use_aggregate_features", True):
        cols += list(AGG_NUMERIC)
    if model_cfg.get("use_market_features", True):
        cols += list(MARKET_NUMERIC)
    if model_cfg.get("use_class_probs", True):
        cols += [f"prob_{c}" for c in CLASSES]
    return cols


@dataclass
class FeaturePipeline:
    numeric_cols: list[str]
    medians: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    tickers: list[str] = field(default_factory=list)
    flairs: list[str] = field(default_factory=list)
    direction_weights: dict[str, list[float]] = field(default_factory=dict)

    def fit(self, train: pd.DataFrame, ticker_universe: list[str]) -> "FeaturePipeline":
        self.tickers = ["<unk>"] + list(ticker_universe)
        self.flairs = ["<unk>"] + sorted(
            x for x in train["flair"].fillna("").value_counts().head(10).index if x
        )
        for c in self.numeric_cols:
            x = pd.to_numeric(train[c], errors="coerce")
            med = float(x.median()) if x.notna().any() else 0.0
            self.medians[c] = med
            x = x.fillna(med)
            self.means[c] = float(x.mean())
            self.stds[c] = float(x.std(ddof=0)) or 1.0
        # Direction class balancing weights per horizon, used in the loss.
        for h in (1, 3, 5, 10):
            col = f"y_h{h}"
            if col not in train.columns:
                continue
            counts = train[col].value_counts()
            total = counts.sum()
            w = []
            for d in ("DOWN", "FLAT", "UP"):
                n = float(counts.get(d, 0))
                w.append(float(total / (3.0 * n)) if n > 0 else 1.0)
            self.direction_weights[str(h)] = w
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.numeric_cols:
            x = pd.to_numeric(out[c], errors="coerce").fillna(self.medians.get(c, 0.0))
            out[c] = (x - self.means.get(c, 0.0)) / (self.stds.get(c, 1.0) or 1.0)
        ticker_map = {t: i for i, t in enumerate(self.tickers)}
        flair_map = {f: i for i, f in enumerate(self.flairs)}
        out["ticker_id"] = out["ticker"].map(ticker_map).fillna(0).astype(int)
        out["flair_id"] = out["flair"].fillna("").map(flair_map).fillna(0).astype(int)
        return out

    def to_json(self) -> dict:
        return {
            "numeric_cols": self.numeric_cols,
            "medians": self.medians,
            "means": self.means,
            "stds": self.stds,
            "tickers": self.tickers,
            "flairs": self.flairs,
            "direction_weights": self.direction_weights,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "FeaturePipeline":
        return cls(**obj)


class PairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, numeric_cols: list[str],
                 max_len: int, horizons: list[int], use_engagement: bool = False,
                 direction_weights: dict[str, list[float]] | None = None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.numeric_cols = numeric_cols
        self.max_len = int(max_len)
        self.horizons = horizons
        self.use_engagement = use_engagement
        self.direction_weights = direction_weights or {}
        self.texts = self.df["text_model"].fillna("").tolist()
        self.numeric = self.df[numeric_cols].to_numpy(dtype=np.float32)
        self.ticker_ids = self.df["ticker_id"].to_numpy(dtype=np.int64)
        self.flair_ids = self.df["flair_id"].to_numpy(dtype=np.int64)
        self.class_ids = self.df["class_id"].to_numpy(dtype=np.int64)
        self.influence = self.df["influence_weight"].to_numpy(dtype=np.float32)
        self.score = self.df["score"].to_numpy(dtype=np.float32)
        self.magnitude = self.df["magnitude"].to_numpy(dtype=np.float32)
        self.labels = {}
        for h in horizons:
            col = f"y_h{h}"
            vals = self.df[col] if col in self.df.columns else pd.Series([None] * len(self.df))
            self.labels[h] = np.array(
                [DIRECTION_TO_ID.get(v, -1) if isinstance(v, str) else -1 for v in vals],
                dtype=np.int64,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_attention_mask=True,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "numeric": torch.tensor(self.numeric[idx], dtype=torch.float32),
            "ticker_id": torch.tensor(self.ticker_ids[idx], dtype=torch.long),
            "flair_id": torch.tensor(self.flair_ids[idx], dtype=torch.long),
            "class_id": torch.tensor(self.class_ids[idx], dtype=torch.long),
            "influence": torch.tensor(self.influence[idx], dtype=torch.float32),
            "score": torch.tensor(self.score[idx], dtype=torch.float32),
            "magnitude": torch.tensor(self.magnitude[idx], dtype=torch.float32),
            "row_index": torch.tensor(idx, dtype=torch.long),
        }
        for h in self.horizons:
            item[f"y_h{h}"] = torch.tensor(self.labels[h][idx], dtype=torch.long)
        return item


def make_collate(tokenizer):
    def collate(batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attention = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            attention[i, :n] = b["attention_mask"]
        out = {
            "input_ids": input_ids,
            "attention_mask": attention,
            "numeric": torch.stack([b["numeric"] for b in batch]),
            "ticker_id": torch.stack([b["ticker_id"] for b in batch]),
            "flair_id": torch.stack([b["flair_id"] for b in batch]),
            "class_id": torch.stack([b["class_id"] for b in batch]),
            "influence": torch.stack([b["influence"] for b in batch]),
            "score": torch.stack([b["score"] for b in batch]),
            "magnitude": torch.stack([b["magnitude"] for b in batch]),
            "row_index": torch.stack([b["row_index"] for b in batch]),
        }
        for key in batch[0]:
            if key.startswith("y_h"):
                out[key] = torch.stack([b[key] for b in batch])
        return out
    return collate


def save_pipeline(path: Path, pipeline: FeaturePipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pipeline.to_json(), indent=2), encoding="utf-8")


def load_pipeline(path: Path) -> FeaturePipeline:
    return FeaturePipeline.from_json(json.loads(path.read_text(encoding="utf-8")))
