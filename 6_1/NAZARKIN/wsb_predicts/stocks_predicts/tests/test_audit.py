"""Gold-audit scorer tests (no human labels required)."""
from __future__ import annotations

import numpy as np

from score_gold_audit import score
from score_ticker_gold import bootstrap_prf, prf


def test_perfect_agreement_scores_kappa_one():
    y = ["NOISE_MEME", "FINANCIAL_CLAIM", "ANALYSIS_DD", "GENUINE_OPINION"] * 10
    out = score(y, list(y), bootstrap=50)
    assert out["n"] == 40
    assert abs(out["kappa"] - 1.0) < 1e-12
    assert out["kappa_95ci"]["lo"] == 1.0
    assert out["accuracy"] == 1.0


def test_noisy_agreement_has_bounded_ci_and_confusion_matrix():
    rng = np.random.default_rng(0)
    classes = np.array(["NOISE_MEME", "GENUINE_OPINION", "FINANCIAL_CLAIM", "ANALYSIS_DD"])
    y = classes[rng.integers(0, 4, size=200)].tolist()
    pred = classes[rng.integers(0, 4, size=200)].tolist()
    out = score(y, pred, bootstrap=100)
    assert -1.0 <= out["kappa"] <= 1.0
    assert out["kappa_95ci"]["lo"] <= out["kappa"] <= out["kappa_95ci"]["hi"]
    cm = np.asarray(out["confusion_matrix_rows_gold_cols_classifier"])
    assert cm.shape == (4, 4)
    assert int(cm.sum()) == len(y)


def test_ticker_prf_counts_pairs_and_exact_match():
    pred = [{"AAPL", "TSLA"}, {"TSLA"}, set(), {"NVDA"}]
    gold = [{"TSLA"}, {"TSLA"}, set(), {"NVDA", "AMD"}]
    out = prf(pred, gold)
    assert out["tp"] == 3
    assert out["fp"] == 1
    assert out["fn"] == 1
    assert out["precision"] == 0.75
    assert out["recall"] == 0.75
    assert out["exact_match"] == 0.5


def test_ticker_bootstrap_ci_brackets_point_estimate():
    pred = [{"AAPL"}, {"TSLA"}, set(), {"NVDA"}, {"GME"}, {"AMC"}]
    gold = [{"AAPL"}, {"TSLA"}, set(), {"NVDA", "AMD"}, {"GME"}, {"AMC"}]
    ci = bootstrap_prf(pred, gold, n_iter=100)
    assert set(ci) == {"precision", "recall", "f1", "exact_match"}
    for bounds in ci.values():
        assert bounds["lo"] <= bounds["hi"]
