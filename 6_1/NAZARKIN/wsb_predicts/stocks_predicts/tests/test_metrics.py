"""Contract tests for the metrics and ticker-day aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import (
    aggregate_ticker_day,
    apply_temperature,
    bootstrap_delta,
    compute_metrics,
    economic_backtest,
    expected_calibration_error,
    fit_temperature,
)


def test_compute_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.eye(3)[y] * 0.9 + 0.0333
    m = compute_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["mcc"] == 1.0
    assert m["n"] == 6


def test_ece_is_zero_for_perfect_calibration():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y]
    assert expected_calibration_error(y, p) == 0.0


def test_temperature_fit_reduces_nll_on_overconfident_logits():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, size=500)
    logits = rng.normal(0, 0.3, size=(500, 3))
    logits[np.arange(len(y)), y] += 2.0
    logits *= 4.0  # deliberately overconfident
    T = fit_temperature(logits, y)
    p_raw = apply_temperature(logits, 1.0)
    p_cal = apply_temperature(logits, T)
    nll_raw = -np.log(p_raw[np.arange(len(y)), y]).mean()
    nll_cal = -np.log(p_cal[np.arange(len(y)), y]).mean()
    assert nll_cal <= nll_raw + 1e-12


def test_ticker_day_aggregation_uses_true_label_not_prediction():
    df = pd.DataFrame({
        "ticker": ["TSLA", "TSLA"],
        "t0_date": pd.to_datetime(["2025-01-06", "2025-01-06"]),
        "influence_weight": [1.0, 1.0],
        "weight_share": [1.0, 1.0],
        "direction_label": ["UP", "UP"],
        "abnormal_return": [0.02, 0.02],
        "raw_return": [0.03, 0.03],
        "benchmark_return": [0.01, 0.01],
        "split": ["test", "test"],
    })
    probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    td = aggregate_ticker_day(df, probs)
    assert len(td) == 1
    assert td.iloc[0]["direction_label"] == "UP"
    assert td.iloc[0]["pred"] == 0  # model disagrees; the label stays ground truth
    assert abs(td.iloc[0]["p_DOWN"] - 0.75) < 1e-9


def test_bootstrap_delta_is_zero_for_identical_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=200)
    p = rng.dirichlet([1, 1, 1], size=200)
    dates = np.repeat(pd.date_range("2025-01-01", periods=20).date, 10)
    out = bootstrap_delta(y, p, p, dates, metric="macro_f1", n_iter=50)
    assert out["mean"] == 0.0
    assert out["lo"] == 0.0 and out["hi"] == 0.0


def test_economic_backtest_charges_costs():
    dates = pd.date_range("2025-01-06", periods=10, freq="W-FRI")
    rows = []
    for d in dates:
        for i, t in enumerate(["A", "B", "C", "D"]):
            rows.append({
                "ticker": t, "t0_date": d, "p_UP": 0.6 if i < 2 else 0.2,
                "p_DOWN": 0.2 if i < 2 else 0.6,
                "abnormal_return": 0.01 if i < 2 else -0.01,
                "direction_label": "UP" if i < 2 else "DOWN",
            })
    td = pd.DataFrame(rows)
    s0, _ = economic_backtest(td, transaction_cost_bps=0.0, quantile=0.25)
    s10, _ = economic_backtest(td, transaction_cost_bps=10.0, quantile=0.25)
    assert s0["annualized_return"] > s10["annualized_return"]
    assert s10["transaction_cost_bps"] == 10.0
