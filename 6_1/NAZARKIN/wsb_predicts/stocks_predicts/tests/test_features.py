"""Feature-pipeline tests: fitting, transforms and leakage guards."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import CLASSES
from features import FeaturePipeline, numeric_columns


def _frame(n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {
            "ticker": ["TSLA", "NVDA", "GME"][i % 3],
            "flair": ["YOLO", "DD", ""][i % 3],
            "score": int(rng.integers(0, 1000)),
            "num_comments": int(rng.integers(0, 100)),
            "mention_count": 1,
            "weight_share": 1.0,
            "confidence": 0.6,
            "log1p_score": float(np.log1p(10)),
            "log1p_comments": 1.0,
            "hour_sin": 0.1, "hour_cos": 0.9, "dow_sin": 0.0, "dow_cos": 1.0,
            "is_premarket": 0.0, "is_afterhours": 0.0,
            "agg_n_posts": 1.0, "agg_mean_score": 10.0, "agg_max_score": 10.0,
            "bench_ret_5": 0.01, "bench_ret_20": 0.02,
            "bench2_ret_5": 0.01, "bench2_ret_20": 0.02,
            "ticker_ret_5": 0.03, "ticker_ret_20": 0.05,
            "ticker_vol_20": 0.02, "beta": 1.1, "threshold": 0.01,
        }
        for c in CLASSES:
            row[f"prob_{c}"] = 0.25
            row[f"agg_class_{c}"] = 0.25
        rows.append(row)
    return pd.DataFrame(rows)


def test_pipeline_fit_transform_fills_medians_and_maps_categories():
    cfg = {"use_aggregate_features": True, "use_market_features": True, "use_class_probs": True}
    cols = numeric_columns(cfg)
    df = _frame()
    pipe = FeaturePipeline(cols).fit(df, ["TSLA", "NVDA", "GME"])
    out = pipe.transform(df)
    assert out[cols].isna().sum().sum() == 0
    assert set(out["ticker_id"]).issubset({0, 1, 2, 3})
    # Unknown ticker/flair must map to <unk>, not crash.
    unseen = df.copy()
    unseen.loc[0, "ticker"] = "ZZZZ"
    unseen.loc[0, "flair"] = "unseen-flair"
    out2 = pipe.transform(unseen)
    assert int(out2.loc[0, "ticker_id"]) == 0
    assert int(out2.loc[0, "flair_id"]) == 0


def test_numeric_features_never_include_labels_or_influence_weight():
    for cfg in [
        {"use_aggregate_features": True, "use_market_features": True, "use_class_probs": True},
        {"use_aggregate_features": False, "use_market_features": False, "use_class_probs": False},
    ]:
        cols = numeric_columns(cfg)
        for forbidden in ["y_h5", "direction_label", "abnormal_return", "influence_weight"]:
            assert forbidden not in cols


def test_pipeline_json_round_trip():
    cfg = {"use_aggregate_features": True, "use_market_features": True, "use_class_probs": True}
    df = _frame()
    pipe = FeaturePipeline(numeric_columns(cfg)).fit(df, ["TSLA", "NVDA", "GME"])
    restored = FeaturePipeline.from_json(pipe.to_json())
    assert restored.numeric_cols == pipe.numeric_cols
    assert restored.tickers == pipe.tickers
    assert restored.medians == pipe.medians
