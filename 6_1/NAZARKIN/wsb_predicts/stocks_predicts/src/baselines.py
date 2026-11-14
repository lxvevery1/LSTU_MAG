#!/usr/bin/env python3
"""Baselines B0-B6 from TZ section 5.2.

B0  majority / stratified random
B1  TF-IDF word+char + LogisticRegression
B2  TF-IDF + LightGBM
B3  frozen FinBERT [CLS] embeddings + LogisticRegression
B4  ANALYSIS_DD + bullish/bearish lexicon heuristic
B5  market features only (no text)
B6  four class probabilities from the existing FinBERT classifier
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
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from common import DIRECTIONS, DIRECTION_TO_ID, load_config, project_path, read_json, setup_logging, write_json
from metrics import (
    aggregate_ticker_day,
    bootstrap_ci,
    compute_metrics,
    fit_temperature,
    apply_temperature,
)

LOG = logging.getLogger("baselines")
MARKET_COLS = ["bench_ret_5", "bench_ret_20", "bench2_ret_5", "bench2_ret_20",
               "ticker_ret_5", "ticker_ret_20", "ticker_vol_20", "beta", "threshold"]
BULL_WORDS = {
    "bull", "bullish", "long", "calls", "call", "buy", "buying", "moon", "mooning",
    "rocket", "rally", "breakout", "up", "higher", "growth", "beat", "beats",
    "green", "tendies", "yolo", "all in", "add", "adding", "hold", "holding",
    "undervalued", "upside", "strong", "outperform", "record", "squeeze",
}
BEAR_WORDS = {
    "bear", "bearish", "short", "puts", "put", "sell", "selling", "dump", "dumping",
    "crash", "collapse", "down", "lower", "loss", "losses", "miss", "misses",
    "red", "overvalued", "downside", "weak", "underperform", "bankrupt",
    "bankruptcy", "drill", "tank", "tanking", "rug", "rugpull", "bagholder",
}


def load_pairs(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(project_path(cfg["paths"]["pairs_h5"])).copy()
    df["t0_date"] = pd.to_datetime(df["t0_date"])
    df = df[~df["purged"].astype(bool)].copy()
    df["text"] = (df["title"].fillna("") + "\n\n" + df["body"].fillna("")).str.strip()
    df["y"] = df["direction_label"].map(DIRECTION_TO_ID).astype(int)
    return df.reset_index(drop=True)


def align_proba(model, proba: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    out = np.zeros((len(proba), 3))
    for j, c in enumerate(classes):
        if isinstance(c, str):
            i = DIRECTIONS.index(c)
        else:
            i = int(c)
        out[:, i] = proba[:, j]
    return out


def one_hot(labels: list[str]) -> np.ndarray:
    out = np.zeros((len(labels), 3))
    for i, lab in enumerate(labels):
        out[i, DIRECTIONS.index(lab)] = 1.0
    return out


def b4_heuristic(df: pd.DataFrame) -> np.ndarray:
    labels = []
    for _, r in df.iterrows():
        text = (str(r["text"]) + " " + str(r.get("flair", ""))).lower()
        bull = sum(1 for w in BULL_WORDS if w in text)
        bear = sum(1 for w in BEAR_WORDS if w in text)
        is_dd = str(r.get("class_label")) == "ANALYSIS_DD" or float(r.get("prob_ANALYSIS_DD", 0)) >= 0.5
        if not is_dd:
            labels.append("FLAT")
        elif bull > bear:
            labels.append("UP")
        elif bear > bull:
            labels.append("DOWN")
        else:
            labels.append("FLAT")
    return one_hot(labels)


def make_tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=40000, sublinear_tf=True)),
    ])


def fit_model(name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
              embeddings: dict[str, np.ndarray] | None):
    y_train = train["y"].to_numpy()
    if name == "B0":
        model = DummyClassifier(strategy="most_frequent")
        model.fit(np.zeros((len(train), 1)), y_train)
        return model, np.zeros((len(train), 1)), np.zeros((len(val), 1)), np.zeros((len(test), 1)), "majority"
    if name == "B4":
        return None, None, None, None, "heuristic"
    if name == "B1":
        pipe = Pipeline([("tfidf", make_tfidf()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    n_jobs=-1, C=1.0))])
        pipe.fit(train["text"], y_train)
        return pipe, None, None, None, "text"
    if name == "B2":
        import lightgbm as lgb
        pipe = Pipeline([
            ("tfidf", make_tfidf()),
            ("clf", lgb.LGBMClassifier(n_estimators=350, learning_rate=0.05, num_leaves=31,
                                       class_weight="balanced", n_jobs=-1, verbose=-1,
                                       random_state=42)),
        ])
        pipe.fit(train["text"], y_train)
        return pipe, None, None, None, "text"
    if name == "B3":
        if embeddings is None:
            raise RuntimeError("B3 requires finbert_embeddings.npz; run src/embeddings.py")
        def emb(df):
            return np.vstack([embeddings[p] for p in df["post_id"]])
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    n_jobs=-1, C=1.0))])
        pipe.fit(emb(train), y_train)
        return pipe, emb(train), emb(val), emb(test), "embedding"
    if name == "B5":
        def market(df):
            x = df[MARKET_COLS].apply(pd.to_numeric, errors="coerce")
            return x.fillna(x.median()).to_numpy(dtype=float)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    n_jobs=-1, C=1.0))])
        pipe.fit(market(train), y_train)
        return pipe, market(train), market(val), market(test), "market"
    if name == "B6":
        cols = [f"prob_{c}" for c in ["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"]]
        def probs(df):
            return df[cols].to_numpy(dtype=float)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    n_jobs=-1, C=1.0))])
        pipe.fit(probs(train), y_train)
        return pipe, probs(train), probs(val), probs(test), "probs"
    raise ValueError(name)


def predict(name, model, data, kind, df):
    if name == "B4":
        return b4_heuristic(df)
    if kind == "majority":
        proba = model.predict_proba(np.zeros((len(df), 1)))
    elif kind == "text":
        proba = model.predict_proba(df["text"])
    else:
        proba = model.predict_proba(data)
    return align_proba(model, proba)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--models", default="B0,B1,B2,B3,B4,B5,B6")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = load_pairs(cfg)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    LOG.info("rows train=%s val=%s test=%s", len(train), len(val), len(test))

    emb = None
    emb_path = project_path(cfg["paths"]["embeddings"])
    if emb_path.exists():
        z = np.load(emb_path, allow_pickle=True)
        emb = {str(pid): vec for pid, vec in zip(z["post_ids"], z["embeddings"])}
        LOG.info("embeddings: %s x %s", len(emb), next(iter(emb.values())).shape if emb else None)

    reports = {}
    all_preds = []
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        LOG.info("=== %s ===", name)
        model, dtr, dva, dte, kind = fit_model(name, train, val, test, emb)
        p_val = predict(name, model, dva, kind, val)
        p_test = predict(name, model, dte, kind, test)
        # Temperature scaling on validation predictions (log-probs are not
        # ideal logits, but this is a fair, simple calibration check).
        logits_val = np.log(np.clip(p_val, 1e-8, 1.0))
        logits_test = np.log(np.clip(p_test, 1e-8, 1.0))
        try:
            T = fit_temperature(logits_val, val["y"].to_numpy())
            p_test_cal = apply_temperature(logits_test, T)
        except Exception:
            T = 1.0
            p_test_cal = p_test
        yv = val["y"].to_numpy()
        yt = test["y"].to_numpy()
        val_metrics = compute_metrics(yv, p_val, prefix="")
        test_metrics = compute_metrics(yt, p_test, prefix="")
        test_cal_metrics = compute_metrics(yt, p_test_cal, prefix="")
        ci = {m: bootstrap_ci(yt, p_test, test["t0_date"].dt.date.to_numpy(), metric=m,
                              n_iter=args.bootstrap, seed=42)
              for m in ("macro_f1", "mcc", "auc", "auc_pr_up")}
        # ticker-day aggregate on test
        td = aggregate_ticker_day(test, p_test)
        y_td = td["direction_label"].map(DIRECTION_TO_ID).to_numpy(dtype=int)
        td_probs = np.vstack([td["p_DOWN"], td["p_FLAT"], td["p_UP"]]).T
        td_metrics = compute_metrics(y_td, td_probs)
        reports[name] = {
            "kind": kind,
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "val": val_metrics,
            "test": test_metrics,
            "test_calibrated": test_cal_metrics,
            "temperature": float(T),
            "bootstrap_95ci": ci,
            "ticker_day_test": td_metrics,
            "ticker_day_groups": int(len(td)),
        }
        for i, r in test.iterrows():
            all_preds.append({
                "model": name, "post_id": r["post_id"], "ticker": r["ticker"],
                "t0_date": r["t0_date"], "y": int(r["y"]),
                "p_DOWN": float(p_test[i, 0]), "p_FLAT": float(p_test[i, 1]),
                "p_UP": float(p_test[i, 2]),
            })
        LOG.info("%s test macro_f1=%.4f mcc=%.4f auc=%.4f", name,
                 test_metrics.get("macro_f1", float("nan")),
                 test_metrics.get("mcc", float("nan")),
                 test_metrics.get("roc_auc_ovr", float("nan")))

    out = project_path(cfg["paths"]["reports"]) / "baselines_metrics.json"
    write_json(out, reports)
    preds = pd.DataFrame(all_preds)
    preds.to_parquet(project_path(cfg["paths"]["reports"]) / "baselines_predictions.parquet", index=False)
    # markdown table
    lines = ["# Baselines B0–B6 (test split, H=5)", "",
             f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}", "",
             "| model | macro-F1 | MCC | AUC OvR | AUC-PR UP | Brier | ECE | ticker-day macro-F1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in reports:
        r = reports[name]
        t = r["test"]
        lines.append(f"| {name} | {t.get('macro_f1', 0):.4f} | {t.get('mcc', 0):.4f} | "
                     f"{t.get('roc_auc_ovr', float('nan')):.4f} | {t.get('auc_pr_up', float('nan')):.4f} | "
                     f"{t.get('brier', float('nan')):.4f} | {t.get('ece', float('nan')):.4f} | "
                     f"{r['ticker_day_test'].get('macro_f1', 0):.4f} |")
    lines += ["", "Bootstrap 95% CI is in `baselines_metrics.json` (date-block resampling).", "",
              "B3 uses frozen FinBERT embeddings; B6 uses the four class probabilities.", ""]
    (project_path(cfg["paths"]["reports"]) / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
