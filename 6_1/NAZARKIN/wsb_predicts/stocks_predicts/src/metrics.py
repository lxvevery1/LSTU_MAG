#!/usr/bin/env python3
"""Metrics, calibration and ticker-day aggregation helpers."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)

from common import DIRECTIONS, DIRECTION_TO_ID


def _normalise(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    s = probs.sum(axis=1, keepdims=True)
    s[s <= 0] = 1.0
    return probs / s


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y_true, probs, prefix: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=int)
    probs = _normalise(probs)
    if len(y_true) == 0:
        return {}
    pred = probs.argmax(axis=1)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(set(y_true)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(np.mean(np.sum((probs - np.eye(3)[y_true]) ** 2, axis=1))),
        "ece": expected_calibration_error(y_true, probs),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, probs, labels=[0, 1, 2]))
    except Exception:
        out["log_loss"] = float("nan")
    if len(set(y_true)) == 3:
        try:
            out["roc_auc_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    else:
        out["roc_auc_ovr"] = float("nan")
    up = DIRECTIONS.index("UP")
    try:
        out["auc_pr_up"] = float(average_precision_score((y_true == up).astype(int), probs[:, up]))
    except Exception:
        out["auc_pr_up"] = float("nan")
    p, r, f, s = precision_recall_fscore_support(y_true, pred, labels=[0, 1, 2], zero_division=0)
    out["per_class"] = {
        DIRECTIONS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(s[i])} for i in range(3)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist()
    if prefix:
        out = {f"{prefix}{k}": v for k, v in out.items()}
    return out


def _date_index(dates: np.ndarray) -> tuple[np.ndarray, dict]:
    unique = np.unique(dates)
    index = {d: np.where(dates == d)[0] for d in unique}
    return unique, index


def bootstrap_ci(y_true, probs, dates, metric: str = "macro_f1",
                 n_iter: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict:
    """Block bootstrap by trading date."""
    y_true = np.asarray(y_true, dtype=int)
    probs = _normalise(probs)
    dates = np.asarray(dates)
    unique, date_index = _date_index(dates)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_iter):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([date_index[d] for d in sampled])
        y = y_true[idx]
        p = probs[idx]
        if metric == "macro_f1":
            values.append(f1_score(y, p.argmax(1), average="macro", zero_division=0))
        elif metric == "mcc":
            values.append(matthews_corrcoef(y, p.argmax(1)) if len(set(y)) > 1 else 0.0)
        elif metric == "accuracy":
            values.append(accuracy_score(y, p.argmax(1)))
        elif metric == "auc":
            if len(set(y)) == 3:
                values.append(roc_auc_score(y, p, multi_class="ovr", average="macro"))
            else:
                continue
        elif metric == "auc_pr_up":
            up = DIRECTIONS.index("UP")
            if (y == up).sum() == 0:
                continue
            values.append(average_precision_score((y == up).astype(int), p[:, up]))
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(values, alpha / 2)),
        "hi": float(np.quantile(values, 1 - alpha / 2)),
        "n": int(len(values)),
    }


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature scaling on validation logits (minimise NLL)."""
    logits = np.asarray(logits, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    def nll(log_t: np.ndarray) -> float:
        t = float(np.exp(log_t[0]))
        z = logits / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y_true)), y_true].mean())

    res = minimize(nll, x0=np.array([0.0]), method="Nelder-Mead")
    return float(np.exp(res.x[0]))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / max(float(temperature), 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_ticker_day(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Weighted ticker-day aggregation (TZ stage F).

    Weights are influence weight * link share.  Every post in the aggregate is
    published before the entry open by construction of the dataset.
    """
    out = df.copy().reset_index(drop=True)
    out["p_DOWN"] = probs[:, 0]
    out["p_FLAT"] = probs[:, 1]
    out["p_UP"] = probs[:, 2]
    out["_w"] = out["influence_weight"].astype(float) * out["weight_share"].astype(float)
    keys = ["ticker", "t0_date"]
    rows = []
    for (ticker, t0), g in out.groupby(keys, sort=False):
        w = g["_w"].to_numpy(dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(g))
        w = w / w.sum()
        p = np.vstack([g["p_DOWN"], g["p_FLAT"], g["p_UP"]]).T
        pm = (p * w[:, None]).sum(axis=0)
        rows.append({
            "ticker": ticker,
            "t0_date": t0,
            "n_posts": len(g),
            "sum_weight": float(g["influence_weight"].sum()),
            "p_DOWN": float(pm[0]),
            "p_FLAT": float(pm[1]),
            "p_UP": float(pm[2]),
            "prob": float(pm.max()),
            "pred": int(pm.argmax()),
            "direction_label": g["direction_label"].iloc[0],
            "abnormal_return": float(g["abnormal_return"].mean()),
            "raw_return": float(g["raw_return"].mean()),
            "benchmark_return": float(g["benchmark_return"].mean()),
            "split": g["split"].iloc[0],
            "year": int(pd.Timestamp(t0).year),
        })
    return pd.DataFrame(rows)


def economic_backtest(td: pd.DataFrame, transaction_cost_bps: float = 5.0,
                      quantile: float = 0.1, rebalance: str = "weekly") -> tuple[dict, pd.DataFrame]:
    """Long-short sanity-check portfolio on ticker-day predictions.

    ``td`` must contain one row per ticker/day with p_UP, p_DOWN and
    abnormal_return.  Returns are abnormal returns (market-neutral by
    construction); transaction costs are charged on turnover.
    """
    if td.empty:
        return {}, pd.DataFrame()
    df = td.copy().sort_values("t0_date").reset_index(drop=True)
    df["score"] = df["p_UP"] - df["p_DOWN"]
    df["t0_date"] = pd.to_datetime(df["t0_date"])
    if rebalance == "weekly":
        df["period"] = df["t0_date"].dt.to_period("W-FRI")
    elif rebalance == "monthly":
        df["period"] = df["t0_date"].dt.to_period("M")
    else:
        df["period"] = df["t0_date"].dt.floor("D")
    daily = []
    prev_long: set[str] = set()
    prev_short: set[str] = set()
    for period, g in df.groupby("period", sort=True):
        n = max(1, int(math.ceil(len(g) * float(quantile))))
        ranked = g.sort_values("score", ascending=False)
        longs = ranked.head(n)
        shorts = ranked.tail(n)
        long_set = set(longs["ticker"])
        short_set = set(shorts["ticker"])
        turnover = (len(long_set ^ prev_long) + len(short_set ^ prev_short)) / max(2 * n, 1)
        cost = turnover * transaction_cost_bps / 10000.0
        long_ret = float(longs["abnormal_return"].mean()) if len(longs) else 0.0
        short_ret = float(shorts["abnormal_return"].mean()) if len(shorts) else 0.0
        ret = long_ret - short_ret - cost
        daily.append({
            "period": str(period),
            "date": g["t0_date"].min(),
            "n_long": len(longs),
            "n_short": len(shorts),
            "long_ret": long_ret,
            "short_ret": short_ret,
            "turnover": turnover,
            "cost": cost,
            "ret": ret,
            "hit": float(ret > 0),
        })
        prev_long, prev_short = long_set, short_set
    pnl = pd.DataFrame(daily)
    r = pnl["ret"].to_numpy()
    periods_per_year = 52 if rebalance == "weekly" else (12 if rebalance == "monthly" else 252)
    ann_ret = float(np.prod(1 + r) ** (periods_per_year / max(1, len(r))) - 1) if len(r) else float("nan")
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 and np.std(r) > 0 else float("nan")
    equity = np.cumprod(1 + r) if len(r) else np.array([1.0])
    dd = equity / np.maximum.accumulate(equity) - 1 if len(equity) else np.array([0.0])
    summary = {
        "n_periods": int(len(pnl)),
        "annualized_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()) if len(dd) else float("nan"),
        "hit_rate": float((r > 0).mean()) if len(r) else float("nan"),
        "mean_period_return": float(r.mean()) if len(r) else float("nan"),
        "avg_turnover": float(pnl["turnover"].mean()) if len(pnl) else float("nan"),
        "transaction_cost_bps": transaction_cost_bps,
    }
    return summary, pnl


def bootstrap_delta(y_true, probs_a, probs_b, dates, metric: str = "macro_f1",
                    n_iter: int = 1000, seed: int = 42) -> dict:
    """Block-bootstrap the paired difference metric(A) - metric(B)."""
    y_true = np.asarray(y_true, dtype=int)
    probs_a = _normalise(probs_a)
    probs_b = _normalise(probs_b)
    dates = np.asarray(dates)
    unique, date_index = _date_index(dates)
    rng = np.random.default_rng(seed)
    diffs = []

    def score(y, p):
        if metric == "macro_f1":
            return f1_score(y, p.argmax(1), average="macro", zero_division=0)
        if metric == "mcc":
            return matthews_corrcoef(y, p.argmax(1)) if len(set(y)) > 1 else 0.0
        if metric == "accuracy":
            return accuracy_score(y, p.argmax(1))
        if metric == "auc":
            return roc_auc_score(y, p, multi_class="ovr", average="macro") if len(set(y)) == 3 else np.nan
        raise ValueError(metric)

    for _ in range(n_iter):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([date_index[d] for d in sampled])
        a = score(y_true[idx], probs_a[idx])
        b = score(y_true[idx], probs_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(a - b)
    d = np.asarray(diffs, dtype=float)
    return {
        "mean": float(d.mean()) if len(d) else float("nan"),
        "lo": float(np.quantile(d, 0.025)) if len(d) else float("nan"),
        "hi": float(np.quantile(d, 0.975)) if len(d) else float("nan"),
        "n": int(len(d)),
        "prob_gt_0": float((d > 0).mean()) if len(d) else float("nan"),
    }
