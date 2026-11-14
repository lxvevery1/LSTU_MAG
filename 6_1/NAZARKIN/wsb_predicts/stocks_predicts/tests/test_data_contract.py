"""Data-contract tests: corpus size, labels, splits and leakage guards."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import CLASSES, DIRECTIONS, INFLUENCE_WEIGHTS, load_config, project_path

ROOT = Path(__file__).resolve().parents[1]


def _load_posts():
    path = ROOT / "data/raw/posts_16k_labeled.json"
    if not path.exists():
        pytest.skip("labelled corpus not built")
    return json.loads(path.read_text(encoding="utf-8"))


def test_corpus_is_16k_and_balanced_by_year():
    obj = _load_posts()
    counts = {year: len(items) for year, items in obj["years"].items()}
    assert sum(counts.values()) == 16_000
    assert all(n == 2_000 for n in counts.values()), counts


def test_post_ids_are_unique_and_classes_complete():
    obj = _load_posts()
    posts = [p for items in obj["years"].values() for p in items]
    ids = [p["id"] for p in posts]
    assert len(ids) == len(set(ids))
    assert all(p.get("class_label") in CLASSES for p in posts)
    bad_weights = [
        p["id"] for p in posts
        if abs(float(p.get("influence_weight", -1)) -
               INFLUENCE_WEIGHTS[p["class_label"]]) > 1e-9
    ]
    assert not bad_weights


def test_direction_dataset_contract():
    path = ROOT / "data/processed/wsb_direction_dataset_v1.parquet"
    if not path.exists():
        pytest.skip("direction dataset not built")
    df = pd.read_parquet(path)
    assert len(df) > 10_000
    h5 = df[df["horizon_days"] == 5]
    assert set(h5["direction_label"]).issubset(set(DIRECTIONS))
    assert h5["direction_label"].notna().all()
    assert set(h5["split"]).issubset({"train", "val", "test"})
    assert h5["created_utc"].notna().all()
    assert (h5["influence_weight"] >= 0).all()
    assert h5["ticker"].nunique() >= 30

    # A post may link to several tickers, but no post may be in two splits.
    split_per_post = h5.groupby("post_id")["split"].nunique()
    assert int(split_per_post.max()) == 1

    train_posts = set(h5[h5["split"] == "train"]["post_id"])
    test_posts = set(h5[h5["split"] == "test"]["post_id"])
    val_posts = set(h5[h5["split"] == "val"]["post_id"])
    assert not (train_posts & test_posts)
    assert not (train_posts & val_posts)
    assert not (val_posts & test_posts)


def test_embargo_removes_cross_boundary_windows():
    path = ROOT / "data/processed/wsb_direction_dataset_v1.parquet"
    prices = ROOT / "stocks_dataset/prices_long.csv"
    if not path.exists() or not prices.exists():
        pytest.skip("dataset or prices not built")
    df = pd.read_parquet(path)
    h5 = df[df["horizon_days"] == 5].copy()
    h5["exit_date"] = pd.to_datetime(h5["exit_date"])
    cfg = load_config("configs/data.yaml")
    px = pd.read_csv(prices, parse_dates=["date"])
    if "stale_quote" in px.columns:
        px = px[~px["stale_quote"].fillna(False).astype(bool)]
    calendar = sorted(px["date"].unique())
    years = np.array([pd.Timestamp(d).year for d in calendar])
    val_start = int(np.where(years == int(cfg["splits"]["val_year"]))[0][0])
    test_start = int(np.where(years == min(cfg["splits"]["test_years"]))[0][0])
    embargo = int(cfg["splits"]["embargo_trading_days"])

    def cutoff(start: int):
        return pd.Timestamp(calendar[start - embargo - 1])

    train = h5[(h5["split"] == "train") & (~h5["purged"])]
    val = h5[(h5["split"] == "val") & (~h5["purged"])]
    assert train["exit_date"].max() <= cutoff(val_start)
    assert val["exit_date"].max() <= cutoff(test_start)
    # The test boundary needs no upper purge: it ends with the available data.
    assert h5[h5["split"] == "test"]["exit_date"].max() <= pd.Timestamp(calendar[-1])


def test_pair_coverage_above_85_percent():
    links = ROOT / "data/interim/post_ticker_links.parquet"
    pairs = ROOT / "data/processed/wsb_pairs_h5.parquet"
    if not links.exists() or not pairs.exists():
        pytest.skip("links or pairs not built")
    n_links = len(pd.read_parquet(links))
    n_pairs = len(pd.read_parquet(pairs))
    assert n_pairs / max(1, n_links) >= 0.85


def test_class_distribution_is_recorded():
    path = ROOT / "reports/prelabel_report.json"
    if not path.exists():
        pytest.skip("prelabel report missing")
    obj = json.loads(path.read_text(encoding="utf-8"))
    dist = obj.get("class_distribution") or obj.get("label_distribution")
    if dist:
        assert set(dist).issubset(set(CLASSES))
        assert sum(dist.values()) > 0
