"""Contract tests for the historical prices / predictions API."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from serve import create_app

ROOT = Path(__file__).resolve().parents[1]
PRICES = ROOT / "stocks_dataset/prices_long.csv"
PREDICTIONS = ROOT / "reports/runs/finbert_h5_seed42/model_predictions.parquet"

pytestmark = pytest.mark.skipif(
    not (PRICES.exists() and PREDICTIONS.exists()),
    reason="price/prediction artifacts are not built",
)


@pytest.fixture(scope="module")
def client():
    from history_api import HistoryService

    svc = HistoryService(default_run_id="finbert_h5_seed42")
    return TestClient(create_app(history_service=svc))


def test_tickers_and_runs(client):
    r = client.get("/tickers")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == "finbert_h5_seed42"
    assert body["n"] == len(body["tickers"]) == 30
    aapl = next(item for item in body["tickers"] if item["ticker"] == "AAPL")
    assert aapl["has_prices"] is True
    assert aapl["test_posts"] > 0

    r = client.get("/runs")
    assert r.status_code == 200
    runs = r.json()
    assert runs["default_run_id"] == "finbert_h5_seed42"
    ids = {item["run_id"] for item in runs["runs"]}
    assert "finbert_h5_seed42" in ids
    seed42 = next(item for item in runs["runs"] if item["run_id"] == "finbert_h5_seed42")
    assert seed42["macro_f1"] == pytest.approx(0.43017, abs=1e-4)
    assert seed42["has_predictions"] is True


def test_prices_contract_and_filters(client):
    r = client.get("/prices/AAPL", params={"start": "2026-09-01", "end": "2026-09-18"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["interval"] == "1d"
    assert body["n"] == len(body["candles"]) == 13
    assert body["start"] == "2026-09-01"
    assert body["end"] == "2026-09-18"
    candle = body["candles"][0]
    assert candle["low"] <= candle["open"] <= candle["high"]
    assert candle["low"] <= candle["close"] <= candle["high"]

    r = client.get("/prices/AAPL", params={"limit": 5})
    assert r.status_code == 200
    assert r.json()["n"] == 5


def test_prediction_contract_matches_chart_numbers(client):
    r = client.get("/predictions/AAPL")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["run_id"] == "finbert_h5_seed42"
    assert body["primary_horizon"] == 5
    assert body["summary"]["n_posts"] == len(body["posts"]) == 46
    assert body["summary"]["n_dates"] == len(body["daily"]) == 26
    assert body["summary"]["prediction_counts"] == {"DOWN": 7, "FLAT": 0, "UP": 19}
    assert body["summary"]["date_hit_rate"] == pytest.approx(0.4230769, abs=1e-6)
    assert body["summary"]["post_hit_rate"] == pytest.approx(0.3913043, abs=1e-6)
    assert body["summary"]["always_up_baseline"] == pytest.approx(0.4615385, abs=1e-6)

    for item in body["daily"]:
        assert abs(sum(item["probabilities"].values()) - 1.0) < 1e-6
        assert item["prediction"] == max(item["probabilities"], key=item["probabilities"].get)

    day = body["daily"][0]
    assert day["date"] == "2025-01-02"
    assert day["n_posts"] == 1
    assert day["prediction"] == "UP"
    assert day["score"] == pytest.approx(
        day["probabilities"]["UP"] - day["probabilities"]["DOWN"]
    )


def test_chart_combines_prices_and_predictions(client):
    r = client.get("/chart/AAPL")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["run_id"] == "finbert_h5_seed42"
    assert len(body["candles"]) > 300
    assert len(body["daily"]) == 26
    assert body["summary"]["n_posts"] == 46
    assert body["start"] <= "2025-01-02"
    assert body["end"] >= "2026-09-18"
    assert "Research use only" in body["disclaimer"]


def test_invalid_inputs_are_rejected(client):
    assert client.get("/prices/FAKE").status_code == 422
    assert client.get("/predictions/FAKE").status_code == 422
    assert client.get(
        "/prices/AAPL", params={"start": "2026-09-18", "end": "2026-09-01"}
    ).status_code == 422
    assert client.get(
        "/predictions/AAPL", params={"run_id": "../etc"}
    ).status_code == 422
    assert client.get(
        "/predictions/AAPL", params={"run_id": "does_not_exist"}
    ).status_code == 404


def test_health_does_not_load_a_model():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok", "model_loaded": False, "run_id": None, "device": None
    }


def test_cors_and_index():
    client = TestClient(create_app())
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "*"
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["docs"] == "/docs"
