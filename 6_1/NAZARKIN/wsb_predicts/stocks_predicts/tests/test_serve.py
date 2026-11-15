"""HTTP API smoke tests.

The fast tests use a stub service so they do not load a 440 MB transformer.
Set ``WSB_TEST_REAL_MODEL=1`` to additionally run one real forward pass.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from serve import create_app


class FakeService:
    def __init__(self):
        self.calls = []

    def info(self):
        return {"run_id": "fake", "device": "cpu", "horizons": [1, 3, 5, 10],
                "ticker_universe": ["TSLA", "NVDA"], "primary_horizon": 5}

    def predict(self, **kwargs):
        ticker = str(kwargs.get("ticker", "")).upper()
        if ticker not in ("TSLA", "NVDA"):
            raise ValueError(f"ticker {ticker!r} is not in the model universe")
        self.calls.append(kwargs)
        probs = {"DOWN": 0.2, "FLAT": 0.3, "UP": 0.5}
        return {
            "run_id": "fake",
            "ticker": ticker,
            "created_utc": 1_700_000_000,
            "t0_utc": "2023-11-15T14:30:00Z",
            "t0_date": "2023-11-15",
            "primary_horizon": 5,
            "direction": "UP",
            "confidence": 0.5,
            "probabilities": probs,
            "calibrated": False,
            "horizons": {
                str(h): {"direction": "UP", "probabilities": probs, "confidence": 0.5}
                for h in (1, 3, 5, 10)
            },
            "post_class": {"NOISE_MEME": 0.1, "GENUINE_OPINION": 0.2,
                           "FINANCIAL_CLAIM": 0.3, "ANALYSIS_DD": 0.4},
            "post_class_label": "ANALYSIS_DD",
            "influence_weight": 1.0,
            "text_fragment": "TSLA calls",
            "warnings": [],
        }


@pytest.fixture()
def client():
    svc = FakeService()
    return TestClient(create_app(svc)), svc


def test_health_and_model_info(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["run_id"] == "fake"
    r = c.get("/model_info")
    assert r.status_code == 200
    assert r.json()["primary_horizon"] == 5


def test_predict_direction_contract(client):
    c, svc = client
    payload = {
        "title": "TSLA calls printing",
        "body": "Buying 250C for next Friday, gamma squeeze setup.",
        "ticker": "tsla",
        "flair": "YOLO",
        "score": 120,
        "num_comments": 14,
    }
    r = c.post("/predict_direction", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ticker"] == "TSLA"
    assert body["direction"] in {"DOWN", "FLAT", "UP"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert set(body["horizons"]) == {"1", "3", "5", "10"}
    assert svc.calls and svc.calls[0]["title"].startswith("TSLA")


def test_unknown_ticker_is_rejected(client):
    c, _ = client
    r = c.post("/predict_direction", json={
        "title": "hello", "body": "world", "ticker": "FAKE"})
    assert r.status_code == 422
    assert "universe" in r.json()["detail"]


def test_missing_text_and_ticker_validation(client):
    c, _ = client
    assert c.post("/predict_direction", json={"title": "x", "body": "y"}).status_code == 422
    r = c.post("/predict_direction", json={"title": "", "body": "", "ticker": "TSLA"})
    assert r.status_code == 422


def test_batch_endpoint(client):
    c, _ = client
    r = c.post("/predict_direction/batch", json=[
        {"title": "TSLA", "body": "", "ticker": "TSLA"},
        {"title": "FAKE", "body": "", "ticker": "FAKE"},
    ])
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert body["results"][1]["error"]


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("WSB_TEST_REAL_MODEL") != "1",
                    reason="set WSB_TEST_REAL_MODEL=1 to load the real checkpoint")
def test_real_model_forward_pass():
    from serve import DirectionService

    svc = DirectionService(device="auto")
    out = svc.predict(
        title="TSLA calls printing",
        body="Buying 250C for next Friday, gamma squeeze setup.",
        ticker="TSLA",
        created_utc="2025-01-15T14:00:00Z",
        flair="YOLO",
    )
    assert out["direction"] in {"DOWN", "FLAT", "UP"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6
    assert out["t0_date"] is not None
