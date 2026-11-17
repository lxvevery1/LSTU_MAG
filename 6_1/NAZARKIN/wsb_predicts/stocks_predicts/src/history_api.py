"""Historical prices and saved test predictions for the dashboard API.

The inference endpoints in :mod:`serve` answer "what does the model say about
this new post?".  The frontend also needs the historical side: OHLC candles and
the saved test-period predictions used in the reports and figures.  This module
keeps those read-only endpoints separate from the transformer so they stay fast
and work even when no checkpoint is loaded.

All data comes from artifacts produced by the training/evaluation pipeline:

* ``stocks_dataset/prices_long.csv``
* ``reports/runs/<run_id>/model_predictions.parquet``
* ``reports/runs/<run_id>/model_metrics.json``

Nothing here re-runs the model or recomputes labels.
"""
from __future__ import annotations

import json
import os
import re
import threading
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from common import load_config, project_path
from extract_tickers import load_ticker_config

REFERENCE_RUN_ID = "finbert_h5_seed42"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DIRECTIONS = ("DOWN", "FLAT", "UP")
Direction = Literal["DOWN", "FLAT", "UP"]
PRICE_COLUMNS = [
    "ticker", "date", "open", "high", "low", "close", "adj_close", "volume",
]
DISCLAIMER = (
    "Research use only. This is a direction classifier evaluated on historical "
    "data, not a price forecast or investment advice."
)


# --------------------------------------------------------------------- schemas
class Candle(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float | None = None
    volume: float | None = None


class PriceResponse(BaseModel):
    ticker: str
    interval: Literal["1d"] = "1d"
    start: str | None = None
    end: str | None = None
    n: int
    candles: list[Candle]


class DailyPrediction(BaseModel):
    date: str
    n_posts: int
    prediction: Direction
    true_direction: Direction
    probabilities: dict[str, float]
    confidence: float
    score: float
    abnormal_return: float | None = None
    hit: bool


class PostPrediction(BaseModel):
    post_id: str
    t0_date: str
    created_utc: int
    title: str
    flair: str
    class_label: str
    influence_weight: float
    prediction: Direction
    true_direction: Direction
    probabilities: dict[str, float]
    confidence: float
    abnormal_return: float | None = None
    hit: bool


class PredictionSummary(BaseModel):
    n_posts: int
    n_dates: int
    post_hit_rate: float
    date_hit_rate: float
    always_up_baseline: float
    mean_confidence: float
    prediction_counts: dict[str, int]
    true_counts: dict[str, int]
    first_date: str | None = None
    last_date: str | None = None


class PredictionsResponse(BaseModel):
    run_id: str
    ticker: str
    primary_horizon: int
    start: str | None = None
    end: str | None = None
    summary: PredictionSummary
    daily: list[DailyPrediction]
    posts: list[PostPrediction]


class TickerInfo(BaseModel):
    ticker: str
    has_prices: bool
    test_posts: int
    first_prediction_date: str | None = None
    last_prediction_date: str | None = None


class TickersResponse(BaseModel):
    run_id: str
    n: int
    tickers: list[TickerInfo]


class RunInfo(BaseModel):
    run_id: str
    created_utc: str | None = None
    test_rows: int | None = None
    test_groups: int | None = None
    macro_f1: float | None = None
    mcc: float | None = None
    auc: float | None = None
    ece: float | None = None
    temperature: float | None = None
    max_len: int | None = None
    has_predictions: bool = True


class RunsResponse(BaseModel):
    default_run_id: str
    n: int
    runs: list[RunInfo]


class ChartResponse(BaseModel):
    ticker: str
    run_id: str
    primary_horizon: int
    interval: Literal["1d"] = "1d"
    start: str | None = None
    end: str | None = None
    candles: list[Candle]
    daily: list[DailyPrediction]
    summary: PredictionSummary
    disclaimer: str = DISCLAIMER


# --------------------------------------------------------------------- helpers
def _iso(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


def _opt_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _direction(value) -> Direction:
    text = str(value).upper()
    return text if text in DIRECTIONS else "FLAT"


def _probabilities(row) -> dict[str, float]:
    return {d: float(row[f"p_{d}"]) for d in DIRECTIONS}


def _row_prediction(row) -> Direction:
    raw = row.get("pred")
    if isinstance(raw, str) and raw.upper() in DIRECTIONS:
        return raw.upper()
    return max(DIRECTIONS, key=lambda d: float(row[f"p_{d}"]))


def _date_filter(
    frame: pd.DataFrame,
    column: str,
    start: date | None,
    end: date | None,
) -> pd.DataFrame:
    if start is not None:
        frame = frame[frame[column].dt.date >= start]
    if end is not None:
        frame = frame[frame[column].dt.date <= end]
    return frame


class HistoryService:
    """Read-only access to price history and saved run predictions."""

    def __init__(
        self,
        config_path: str = "configs/data.yaml",
        runs_root: str | Path | None = None,
        default_run_id: str | None = None,
    ):
        self.cfg = load_config(config_path)
        self.ticker_cfg = load_ticker_config(self.cfg["tickers"]["config"])
        self.universe = list(dict.fromkeys(self.ticker_cfg.get("universe", [])))
        self.primary_horizon = int(self.cfg.get("primary_horizon", 5))
        self.prices_path = project_path(self.cfg["prices"]["prices_long"])
        self.runs_root = project_path(
            runs_root or (self.cfg["paths"]["reports"] + "/runs")
        )
        if default_run_id:
            self.default_run_id = default_run_id
        else:
            raw = os.environ.get("WSB_RUN_DIR")
            self.default_run_id = (
                Path(raw).name if raw else os.environ.get("WSB_RUN_ID", REFERENCE_RUN_ID)
            )
        self._price_lock = threading.Lock()
        self._price_mtime: int | None = None
        self._prices: pd.DataFrame | None = None
        self._pred_lock = threading.Lock()
        self._pred_cache: dict[str, tuple[int, pd.DataFrame]] = {}

    # ------------------------------------------------------------- validation
    def normalize_ticker(self, ticker: str) -> str:
        symbol = str(ticker).strip().upper().lstrip("$")
        if symbol not in self.universe:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"ticker {symbol!r} is not in the model universe "
                    f"({', '.join(self.universe)})"
                ),
            )
        return symbol

    def _run_dir(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id):
            raise HTTPException(status_code=422, detail=f"invalid run_id {run_id!r}")
        path = (self.runs_root / run_id).resolve()
        if path.parent != self.runs_root:
            raise HTTPException(status_code=422, detail=f"invalid run_id {run_id!r}")
        if not path.is_dir():
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return path

    def resolve_run_id(self, run_id: str | None = None) -> str:
        if run_id:
            path = self._run_dir(run_id)
            if not (path / "model_predictions.parquet").exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"run {run_id!r} has no model_predictions.parquet",
                )
            return run_id
        candidates = [self.default_run_id, REFERENCE_RUN_ID]
        for candidate in candidates:
            try:
                self._run_dir(candidate)
            except HTTPException:
                continue
            if (self.runs_root / candidate / "model_predictions.parquet").exists():
                return candidate
        available = self._prediction_run_ids()
        if available:
            return available[0]
        raise HTTPException(
            status_code=503,
            detail="no evaluated run with model_predictions.parquet is available",
        )

    # ------------------------------------------------------------------ prices
    def _price_frame(self) -> pd.DataFrame:
        if not self.prices_path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"price file not found: {self.prices_path}",
            )
        mtime = self.prices_path.stat().st_mtime_ns
        with self._price_lock:
            if self._prices is None or self._price_mtime != mtime:
                frame = pd.read_csv(
                    self.prices_path, usecols=PRICE_COLUMNS, parse_dates=["date"]
                )
                frame = frame[frame["ticker"].isin(self.universe)].copy()
                frame["ticker"] = frame["ticker"].astype(str)
                self._prices = frame
                self._price_mtime = mtime
            return self._prices

    def _price_payload(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
        limit: int | None = None,
    ) -> PriceResponse:
        symbol = self.normalize_ticker(ticker)
        if start and end and start > end:
            raise HTTPException(status_code=422, detail="start must be <= end")
        frame = self._price_frame()
        frame = frame[frame["ticker"] == symbol].sort_values("date")
        if frame.empty:
            raise HTTPException(
                status_code=404, detail=f"no price history for {symbol}"
            )
        frame = _date_filter(frame, "date", start, end)
        if limit is not None:
            frame = frame.tail(limit)
        candles = [
            Candle(
                date=_iso(row.date),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                adj_close=_opt_float(row.adj_close),
                volume=_opt_float(row.volume),
            )
            for row in frame.itertuples(index=False)
        ]
        return PriceResponse(
            ticker=symbol,
            start=candles[0].date if candles else None,
            end=candles[-1].date if candles else None,
            n=len(candles),
            candles=candles,
        )

    # ------------------------------------------------------------- predictions
    def _prediction_frame(self, run_id: str) -> pd.DataFrame:
        path = self._run_dir(run_id) / "model_predictions.parquet"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"run {run_id!r} has no model_predictions.parquet",
            )
        mtime = path.stat().st_mtime_ns
        with self._pred_lock:
            cached = self._pred_cache.get(run_id)
            if cached is None or cached[0] != mtime:
                frame = pd.read_parquet(path)
                frame["t0_date"] = pd.to_datetime(frame["t0_date"])
                if "split" in frame.columns:
                    test = frame[frame["split"] == "test"]
                    if not test.empty:
                        frame = test
                self._pred_cache[run_id] = (mtime, frame)
            return self._pred_cache[run_id][1]

    def _prediction_run_ids(self) -> list[str]:
        if not self.runs_root.exists():
            return []
        return sorted(
            path.name
            for path in self.runs_root.iterdir()
            if path.is_dir() and (path / "model_predictions.parquet").exists()
        )

    def _build_predictions(
        self, frame: pd.DataFrame
    ) -> tuple[list[PostPrediction], list[DailyPrediction], PredictionSummary]:
        posts: list[PostPrediction] = []
        for row in frame.sort_values(["t0_date", "created_utc"]).itertuples(index=False):
            true = _direction(row.direction_label)
            pred = _row_prediction(row._asdict())
            posts.append(
                PostPrediction(
                    post_id=str(row.post_id),
                    t0_date=_iso(row.t0_date) or "",
                    created_utc=int(row.created_utc),
                    title=str(row.title),
                    flair=str(row.flair),
                    class_label=str(row.class_label),
                    influence_weight=float(row.influence_weight),
                    prediction=pred,
                    true_direction=true,
                    probabilities=_probabilities(row._asdict()),
                    confidence=float(row.confidence),
                    abnormal_return=_opt_float(row.abnormal_return),
                    hit=pred == true,
                )
            )

        daily: list[DailyPrediction] = []
        for day, group in frame.groupby("t0_date", sort=True):
            probs = {
                d: float(group[f"p_{d}"].mean()) for d in DIRECTIONS
            }
            pred = max(DIRECTIONS, key=lambda d: probs[d])
            true = _direction(group["direction_label"].iloc[0])
            daily.append(
                DailyPrediction(
                    date=_iso(day) or "",
                    n_posts=int(len(group)),
                    prediction=pred,
                    true_direction=true,
                    probabilities=probs,
                    confidence=float(max(probs.values())),
                    score=float(probs["UP"] - probs["DOWN"]),
                    abnormal_return=_opt_float(group["abnormal_return"].iloc[0]),
                    hit=pred == true,
                )
            )

        pred_counts = {d: 0 for d in DIRECTIONS}
        true_counts = {d: 0 for d in DIRECTIONS}
        for item in daily:
            pred_counts[item.prediction] += 1
            true_counts[item.true_direction] += 1
        n_posts = len(posts)
        summary = PredictionSummary(
            n_posts=n_posts,
            n_dates=len(daily),
            post_hit_rate=float(np.mean([p.hit for p in posts])) if posts else 0.0,
            date_hit_rate=float(np.mean([d.hit for d in daily])) if daily else 0.0,
            always_up_baseline=(
                float(np.mean([d.true_direction == "UP" for d in daily]))
                if daily
                else 0.0
            ),
            mean_confidence=(
                float(np.mean([p.confidence for p in posts])) if posts else 0.0
            ),
            prediction_counts=pred_counts,
            true_counts=true_counts,
            first_date=daily[0].date if daily else None,
            last_date=daily[-1].date if daily else None,
        )
        return posts, daily, summary

    def _prediction_payload(
        self,
        ticker: str,
        run_id: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> PredictionsResponse:
        symbol = self.normalize_ticker(ticker)
        if start and end and start > end:
            raise HTTPException(status_code=422, detail="start must be <= end")
        resolved = self.resolve_run_id(run_id)
        frame = self._prediction_frame(resolved)
        frame = frame[frame["ticker"] == symbol].copy()
        frame = _date_filter(frame, "t0_date", start, end)
        posts, daily, summary = self._build_predictions(frame)
        return PredictionsResponse(
            run_id=resolved,
            ticker=symbol,
            primary_horizon=self.primary_horizon,
            start=start.isoformat() if start else summary.first_date,
            end=end.isoformat() if end else summary.last_date,
            summary=summary,
            daily=daily,
            posts=posts,
        )

    # -------------------------------------------------------------------- runs
    def _run_info(self, run_id: str, has_predictions: bool = True) -> RunInfo:
        path = self._run_dir(run_id)
        info = RunInfo(run_id=run_id, has_predictions=has_predictions)
        metrics_path = path / "model_metrics.json"
        if not metrics_path.exists():
            return info
        try:
            obj = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return info
        metrics = obj.get("calibrated_metrics") or obj.get("raw_metrics") or {}
        info.created_utc = obj.get("created_utc")
        info.test_rows = obj.get("test_rows")
        info.test_groups = obj.get("test_groups")
        info.macro_f1 = _opt_float(metrics.get("macro_f1"))
        info.mcc = _opt_float(metrics.get("mcc"))
        info.auc = _opt_float(metrics.get("roc_auc_ovr"))
        info.ece = _opt_float(metrics.get("ece"))
        info.temperature = _opt_float(obj.get("temperature"))
        info.max_len = obj.get("max_len")
        return info

    # ------------------------------------------------------------------ public
    def tickers(self) -> TickersResponse:
        run_id = self.resolve_run_id()
        frame = self._prediction_frame(run_id)
        prices = self._price_frame()
        has_price = set(prices["ticker"].unique())
        out: list[TickerInfo] = []
        for symbol in self.universe:
            sub = frame[frame["ticker"] == symbol]
            out.append(
                TickerInfo(
                    ticker=symbol,
                    has_prices=symbol in has_price,
                    test_posts=int(len(sub)),
                    first_prediction_date=(
                        _iso(sub["t0_date"].min()) if not sub.empty else None
                    ),
                    last_prediction_date=(
                        _iso(sub["t0_date"].max()) if not sub.empty else None
                    ),
                )
            )
        return TickersResponse(run_id=run_id, n=len(out), tickers=out)

    def runs(self) -> RunsResponse:
        ids = self._prediction_run_ids()
        runs = [self._run_info(run_id) for run_id in ids]
        runs.sort(key=lambda item: item.created_utc or "", reverse=True)
        return RunsResponse(
            default_run_id=self.resolve_run_id(),
            n=len(runs),
            runs=runs,
        )

    def prices(
        self,
        ticker: str,
        start: date | None = None,
        end: date | None = None,
        limit: int | None = None,
    ) -> PriceResponse:
        return self._price_payload(ticker, start=start, end=end, limit=limit)

    def predictions(
        self,
        ticker: str,
        run_id: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> PredictionsResponse:
        return self._prediction_payload(ticker, run_id=run_id, start=start, end=end)

    def chart(
        self,
        ticker: str,
        run_id: str | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> ChartResponse:
        preds = self._prediction_payload(
            ticker, run_id=run_id, start=start, end=end
        )
        # Default window: enough context before the first prediction and a few
        # sessions after the last one, so the chart shows the full test span.
        if start is None:
            first = (
                date.fromisoformat(preds.summary.first_date)
                if preds.summary.first_date
                else None
            )
            price_start = first - timedelta(days=30) if first else None
        else:
            price_start = start
        if end is None:
            last = (
                date.fromisoformat(preds.summary.last_date)
                if preds.summary.last_date
                else None
            )
            price_end = last + timedelta(days=10) if last else None
        else:
            price_end = end
        prices = self._price_payload(
            preds.ticker, start=price_start, end=price_end
        )
        return ChartResponse(
            ticker=preds.ticker,
            run_id=preds.run_id,
            primary_horizon=preds.primary_horizon,
            start=prices.start,
            end=prices.end,
            candles=prices.candles,
            daily=preds.daily,
            summary=preds.summary,
        )


def register_history_routes(
    app: FastAPI,
    get_service: Callable[[], HistoryService],
) -> None:
    """Attach read-only chart/history endpoints to an existing app."""

    @app.get("/tickers", response_model=TickersResponse, tags=["history"])
    def tickers() -> TickersResponse:
        """Ticker universe with saved-test-prediction coverage."""
        return get_service().tickers()

    @app.get("/runs", response_model=RunsResponse, tags=["history"])
    def runs() -> RunsResponse:
        """Evaluated runs that have saved test predictions."""
        return get_service().runs()

    @app.get("/prices/{ticker}", response_model=PriceResponse, tags=["history"])
    def prices(
        ticker: str,
        start: date | None = Query(None, description="Inclusive ISO date"),
        end: date | None = Query(None, description="Inclusive ISO date"),
        limit: int | None = Query(
            None, ge=1, le=20_000, description="Keep the latest N candles"
        ),
    ) -> PriceResponse:
        """Daily OHLCV history for one universe ticker."""
        return get_service().prices(ticker, start=start, end=end, limit=limit)

    @app.get(
        "/predictions/{ticker}",
        response_model=PredictionsResponse,
        tags=["history"],
    )
    def predictions(
        ticker: str,
        run_id: str | None = Query(None, description="Defaults to the served run"),
        start: date | None = Query(None, description="Inclusive ISO date"),
        end: date | None = Query(None, description="Inclusive ISO date"),
    ) -> PredictionsResponse:
        """Saved H=5 test predictions, post rows plus daily aggregates."""
        return get_service().predictions(
            ticker, run_id=run_id, start=start, end=end
        )

    @app.get("/chart/{ticker}", response_model=ChartResponse, tags=["history"])
    def chart(
        ticker: str,
        run_id: str | None = Query(None, description="Defaults to the served run"),
        start: date | None = Query(None, description="Inclusive ISO date"),
        end: date | None = Query(None, description="Inclusive ISO date"),
    ) -> ChartResponse:
        """One payload for the price + daily-prediction chart."""
        return get_service().chart(ticker, run_id=run_id, start=start, end=end)
