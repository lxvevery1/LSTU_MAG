#!/usr/bin/env python3
"""HTTP inference service for the WSB direction model.

Endpoints
---------
GET  /health
GET  /model_info
POST /predict_direction
POST /predict_direction/batch

The service is deliberately conservative: it only accepts tickers from the
configured 30-symbol WSB universe, builds every feature from information
available at or before the next market open, and never uses the direction label
as an input.  Post-class probabilities come from the frozen 4-class FinBERT
classifier and are used only as features, exactly as in training.

Usage:
    python src/serve.py --run-id finbert_h5_seed42 --port 8080
    curl -s localhost:8080/predict_direction -H 'content-type: application/json' -d '{
      "title": "TSLA calls printing",
      "body": "Buying 250C for next Friday, gamma squeeze setup.",
      "ticker": "tsla",
      "flair": "YOLO"
    }'
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Allow `python src/serve.py` from the project root.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_dataset import (  # noqa: E402
    load_prices,
    market_features,
    next_entry_index,
    open_timestamp,
)
from common import (  # noqa: E402
    CLASSES,
    CLASS_TO_ID,
    DIRECTION_TO_ID,
    INFLUENCE_WEIGHTS,
    load_config,
    model_path,
    project_path,
    setup_logging,
)
from extract_tickers import extract_tickers, load_ticker_config, make_fragment  # noqa: E402
from features import FeaturePipeline, _time_features  # noqa: E402
from history_api import HistoryService, register_history_routes  # noqa: E402
from model import DirectionFusionModel  # noqa: E402

LOG = logging.getLogger("serve")
DIRECTIONS = ["DOWN", "FLAT", "UP"]


# --------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    title: str = Field(default="", max_length=20_000)
    body: str = Field(default="", max_length=100_000)
    ticker: str = Field(min_length=1, max_length=10)
    created_utc: float | str | None = Field(
        default=None,
        description="Unix seconds or ISO-8601. Defaults to now (UTC).",
    )
    score: int = Field(default=0, ge=-1_000_000, le=10_000_000)
    num_comments: int = Field(default=0, ge=0, le=10_000_000)
    flair: str = Field(default="", max_length=200)


class HorizonPrediction(BaseModel):
    direction: Literal["DOWN", "FLAT", "UP"]
    probabilities: dict[str, float]
    confidence: float


class PredictResponse(BaseModel):
    run_id: str
    ticker: str
    created_utc: int
    t0_utc: str | None
    t0_date: str | None
    primary_horizon: int
    direction: Literal["DOWN", "FLAT", "UP"]
    confidence: float
    probabilities: dict[str, float]
    calibrated: bool
    horizons: dict[str, HorizonPrediction]
    post_class: dict[str, float]
    post_class_label: str
    influence_weight: float
    text_fragment: str
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------- helpers
def latest_run() -> Path:
    runs = sorted((project_path("models")).glob("finbert_h5_seed*"))
    if not runs:
        raise SystemExit("no trained run found; run src/train.py first")
    return runs[-1]


def parse_timestamp(value: float | str | None) -> int:
    if value is None:
        return int(datetime.now(timezone.utc).timestamp())
    if isinstance(value, (int, float)):
        ts = float(value)
    else:
        text = str(value).strip()
        if re.fullmatch(r"\d+(\.\d+)?", text):
            ts = float(text)
        else:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
    if not math.isfinite(ts) or ts < 0:
        raise ValueError("created_utc must be a valid non-negative timestamp")
    return int(ts)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class DirectionService:
    """Loads one checkpoint and serves post x ticker direction predictions."""

    def __init__(
        self,
        run_dir: Path | str | None = None,
        device: str = "auto",
        config_path: str = "configs/data.yaml",
        temperature: float | None = None,
    ):
        setup_logging()
        self.cfg = load_config(config_path)
        self.run_dir = Path(run_dir) if run_dir else latest_run()
        self.device = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(self.run_dir / "model.pt", map_location=self.device,
                          weights_only=False)
        self.ckpt = ckpt
        self.mcfg = ckpt["model_config"]
        self.horizons = [int(h) for h in ckpt["horizons"]]
        self.primary = int(ckpt["primary_horizon"])
        self.max_len = int(ckpt["max_len"])
        self.pipeline = FeaturePipeline.from_json(ckpt["pipeline"])
        self.model = DirectionFusionModel(
            encoder_path=model_path(self.mcfg["encoder_path"]),
            numeric_dim=len(self.pipeline.numeric_cols),
            n_tickers=len(self.pipeline.tickers),
            n_flairs=len(self.pipeline.flairs),
            horizons=self.horizons,
            hidden_dim=int(self.mcfg.get("hidden_dim", 128)),
            ticker_dim=int(self.mcfg.get("ticker_embedding_dim", 16)),
            flair_dim=int(self.mcfg.get("flair_embedding_dim", 8)),
            dropout=float(self.mcfg.get("dropout", 0.2)),
            freeze_layers=int(self.mcfg.get("freeze_layers", 8)),
            use_aux_class=bool(self.mcfg.get("use_aux_class_loss", True)),
            use_magnitude=bool(self.mcfg.get("use_magnitude_loss", False)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.run_dir / "tokenizer"))

        # 4-class post classifier.  It is only needed when the checkpoint was
        # trained with class probabilities as numeric features.
        self.classifier = None
        self.classifier_tokenizer = None
        self.class_names = list(CLASSES)
        if "prob_NOISE_MEME" in self.pipeline.numeric_cols:
            clf_path = project_path(self.mcfg.get("classifier_checkpoint",
                                                   self.mcfg["encoder_path"]))
            self.classifier = AutoModelForSequenceClassification.from_pretrained(
                str(clf_path)).to(self.device).eval()
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(str(clf_path))
            id2label = {int(k): str(v) for k, v in self.classifier.config.id2label.items()}
            self.class_names = [id2label[i] for i in sorted(id2label)]

        # Price history for market features.  Failure is non-fatal: medians from
        # the training pipeline are used instead.
        try:
            self.calendar, self.open_wide, self.close_wide, self.ret_wide, _ = load_prices(self.cfg)
        except Exception as exc:  # pragma: no cover - only on a broken install
            LOG.warning("price history unavailable, market features will be medians: %s", exc)
            self.calendar = []
            self.open_wide = self.close_wide = self.ret_wide = None
        self.ticker_cfg = load_ticker_config(self.cfg["tickers"]["config"])
        self.universe = list(dict.fromkeys(self.ticker_cfg.get("universe", [])))
        self.fragment_chars = int(self.cfg["tickers"].get("fragment_chars", 200))

        self.temperature = float(temperature) if temperature is not None else self._load_temperature()
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            self.temperature = 1.0
        self._info = {
            "run_id": self.run_dir.name,
            "device": str(self.device),
            "primary_horizon": self.primary,
            "horizons": self.horizons,
            "max_len": self.max_len,
            "n_numeric_features": len(self.pipeline.numeric_cols),
            "ticker_universe": self.universe,
            "temperature_primary": self.temperature,
            "uses_class_probs": self.classifier is not None,
        }
        LOG.info("service ready: %s", json.dumps(self._info))

    # ------------------------------------------------------------------ setup
    def _load_temperature(self) -> float:
        reports = project_path(self.cfg["paths"]["reports"])
        candidates = [reports / "runs" / self.run_dir.name / "model_metrics.json",
                      reports / "model_metrics.json"]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return 1.0
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 1.0
        if obj.get("run_id") != self.run_dir.name:
            return 1.0
        return float(obj.get("temperature", 1.0) or 1.0)

    def info(self) -> dict:
        return dict(self._info)

    # -------------------------------------------------------------- inference
    def _class_probs(self, title: str, body: str) -> tuple[dict[str, float], str, float]:
        if self.classifier is None or self.classifier_tokenizer is None:
            probs = {c: 0.0 for c in CLASSES}
            return probs, "UNCERTAIN", 0.0
        text = (title + "\n\n" + body).strip()
        enc = self.classifier_tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            logits = self.classifier(**enc).logits.float().cpu().numpy()[0]
        p = _softmax(logits)
        probs = {c: 0.0 for c in CLASSES}
        for name, value in zip(self.class_names, p):
            probs[name] = float(value)
        label = max(probs, key=probs.get)
        return probs, label, float(max(probs.values()))

    def _fragment(self, ticker: str, title: str, body: str) -> tuple[str, int]:
        text = (title + "\n\n" + body).strip()
        matches = extract_tickers(text, self.ticker_cfg)
        for rec in matches:
            if rec["ticker"] == ticker:
                return make_fragment(text, int(rec["first_pos"]), self.fragment_chars), \
                    int(rec["mention_count"])
        return make_fragment(text, 0, self.fragment_chars), 0

    def _market(self, ticker: str, created_utc: int) -> tuple[pd.Timestamp | None, dict]:
        empty = {
            "bench_ret_5": np.nan, "bench_ret_20": np.nan,
            "bench2_ret_5": np.nan, "bench2_ret_20": np.nan,
            "ticker_ret_5": np.nan, "ticker_ret_20": np.nan,
            "ticker_vol_20": np.nan, "beta": 1.0, "threshold": 0.005,
        }
        if not self.calendar:
            return None, empty
        idx = next_entry_index(created_utc, self.calendar)
        if idx is None or idx >= len(self.calendar):
            # The requested entry is beyond the cached price history.  Return
            # no market features (the pipeline fills training medians), but
            # keep a human-readable next business day instead of the calendar
            # date of the post itself.
            next_day = pd.bdate_range(
                start=pd.Timestamp(datetime.fromtimestamp(created_utc, tz=timezone.utc).date())
                + pd.Timedelta(days=1), periods=1
            )[0]
            return next_day, empty
        t0_date = self.calendar[idx]
        feats = market_features(
            ticker, idx, self.ret_wide,
            str(self.cfg["prices"]["benchmark"]),
            str(self.cfg["prices"].get("secondary_benchmark", self.cfg["prices"]["benchmark"])),
        )
        from build_dataset import rolling_beta, rolling_sigma
        beta = rolling_beta(
            ticker, idx, self.ret_wide, str(self.cfg["prices"]["benchmark"]),
            int(self.cfg["prices"]["beta_window"]), int(self.cfg["prices"]["beta_min_obs"]),
        )
        sigma = rolling_sigma(ticker, idx, self.ret_wide,
                              int(self.cfg["prices"]["vol_window"]))
        theta = max(float(self.cfg["prices"]["min_threshold"]),
                    float(self.cfg["prices"]["threshold_vol_multiplier"]) * sigma)
        feats["beta"] = float(beta)
        feats["threshold"] = float(theta)
        return t0_date, feats

    def _build_frame(
        self,
        title: str,
        body: str,
        ticker: str,
        created_utc: int,
        score: int,
        num_comments: int,
        flair: str,
    ) -> tuple[pd.DataFrame, dict]:
        probs, class_label, confidence = self._class_probs(title, body)
        fragment, mention_count = self._fragment(ticker, title, body)
        t0_date, market = self._market(ticker, created_utc)
        if t0_date is None:
            t0_date = pd.Timestamp(datetime.fromtimestamp(created_utc, tz=timezone.utc).date())
        created = pd.Series([int(created_utc)])
        t0s = pd.Series([pd.Timestamp(t0_date)])
        tf = _time_features(created, t0s).iloc[0].to_dict()

        if self.mcfg.get("use_ticker_fragment", True):
            text_model = (
                "$" + ticker + " [SEP] " + (title or "")
                + " [SEP] " + fragment + " [SEP] " + (body or "")
            )
        else:
            text_model = "$" + ticker + " [SEP] " + ((title or "") + "\n\n" + (body or "")).strip()

        row: dict[str, Any] = {
            "post_id": "inference",
            "created_utc": int(created_utc),
            "ticker": ticker,
            "title": title or "",
            "body": body or "",
            "flair": flair or "",
            "score": int(score),
            "num_comments": int(num_comments),
            "class_label": class_label,
            "class_id": CLASS_TO_ID.get(class_label, 0),
            "confidence": confidence,
            "influence_weight": float(INFLUENCE_WEIGHTS.get(class_label, 0.0)),
            "mention_count": int(mention_count),
            "weight_share": 1.0,
            "fragment": fragment,
            "t0_date": pd.Timestamp(t0_date),
            "text_model": text_model,
            "text_plain": ((title or "") + "\n\n" + (body or "")).strip(),
            "log1p_score": float(np.log1p(max(0, score))),
            "log1p_comments": float(np.log1p(max(0, num_comments))),
            "agg_n_posts": 1.0,
            "agg_mean_score": float(score),
            "agg_max_score": float(score),
            "magnitude": 0.0,
            "label_h5": None,
        }
        for c in CLASSES:
            row[f"prob_{c}"] = float(probs.get(c, 0.0))
            row[f"agg_class_{c}"] = float(probs.get(c, 0.0))
        row.update(tf)
        row.update(market)
        for h in [1, 3, 5, 10]:
            row.setdefault(f"y_h{h}", None)
        frame = pd.DataFrame([row])
        frame = self.pipeline.transform(frame)
        return frame, {
            "probs": probs,
            "class_label": class_label,
            "confidence": confidence,
            "fragment": fragment,
            "mention_count": mention_count,
            "t0_date": t0_date,
        }

    def predict(
        self,
        title: str = "",
        body: str = "",
        ticker: str = "",
        created_utc: float | str | None = None,
        score: int = 0,
        num_comments: int = 0,
        flair: str = "",
    ) -> dict:
        ticker = str(ticker).strip().upper().lstrip("$")
        if ticker not in self.universe:
            raise ValueError(
                f"ticker {ticker!r} is not in the model universe "
                f"({', '.join(self.universe)})"
            )
        if not (title or body).strip():
            raise ValueError("title or body must contain text")
        ts = parse_timestamp(created_utc)
        frame, meta = self._build_frame(
            title, body, ticker, ts, score, num_comments, flair
        )
        text = frame["text_model"].iloc[0]
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_len, return_tensors="pt"
        ).to(self.device)
        numeric = torch.tensor(
            frame[self.pipeline.numeric_cols].to_numpy(dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )
        ticker_id = torch.tensor(frame["ticker_id"].to_numpy(dtype=np.int64),
                                 dtype=torch.long, device=self.device)
        flair_id = torch.tensor(frame["flair_id"].to_numpy(dtype=np.int64),
                                dtype=torch.long, device=self.device)
        with torch.inference_mode():
            out = self.model(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                numeric=numeric, ticker_id=ticker_id, flair_id=flair_id,
            )
            logits = {h: out["logits"][f"h{h}"].float().cpu().numpy()[0]
                      for h in self.horizons}

        warnings = []
        horizon_out: dict[str, dict] = {}
        for h in self.horizons:
            p = _softmax(logits[h])
            calibrated = False
            if h == self.primary and abs(self.temperature - 1.0) > 1e-9:
                p = _softmax(logits[h] / self.temperature)
                calibrated = True
            probs = {DIRECTIONS[i]: float(p[i]) for i in range(3)}
            horizon_out[str(h)] = {
                "direction": DIRECTIONS[int(np.argmax(p))],
                "probabilities": probs,
                "confidence": float(np.max(p)),
                "calibrated": calibrated,
                "logits": [float(x) for x in logits[h]],
            }
        if (frame["t0_date"].iloc[0] > pd.Timestamp(self.calendar[-1])) if self.calendar else False:
            warnings.append("created_utc is beyond the cached price history; market features use training medians")
        if meta["mention_count"] == 0:
            warnings.append("the requested ticker is not explicitly mentioned in the text")

        primary = horizon_out[str(self.primary)]
        return {
            "run_id": self.run_dir.name,
            "ticker": ticker,
            "created_utc": ts,
            "t0_utc": open_timestamp(meta["t0_date"]).astimezone(timezone.utc)
            .isoformat().replace("+00:00", "Z") if meta["t0_date"] is not None else None,
            "t0_date": str(pd.Timestamp(meta["t0_date"]).date()) if meta["t0_date"] is not None else None,
            "primary_horizon": self.primary,
            "direction": primary["direction"],
            "confidence": primary["confidence"],
            "probabilities": primary["probabilities"],
            "calibrated": bool(primary.get("calibrated", False)),
            "horizons": horizon_out,
            "post_class": {k: float(v) for k, v in meta["probs"].items()},
            "post_class_label": meta["class_label"],
            "influence_weight": float(INFLUENCE_WEIGHTS.get(meta["class_label"], 0.0)),
            "text_fragment": meta["fragment"],
            "warnings": warnings,
        }


# --------------------------------------------------------------------- FastAPI
def create_app(
    service: Any | None = None,
    history_service: Any | None = None,
) -> FastAPI:
    app = FastAPI(
        title="WSB Direction Model",
        version="1.1.0",
        description=(
            "Predict 5-day (and auxiliary 1/3/10-day) beta-adjusted direction of "
            "a WSB post x ticker pair, and serve historical prices plus saved "
            "test predictions for the dashboard. Research use only; not "
            "investment advice."
        ),
    )
    app.state.service = service
    app.state.history_service = history_service

    origins = [
        item.strip()
        for item in os.environ.get("WSB_CORS_ORIGINS", "*").split(",")
        if item.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_service() -> Any:
        if app.state.service is None:
            app.state.service = DirectionService(
                run_dir=os.environ.get("WSB_RUN_DIR") or None,
                device=os.environ.get("WSB_DEVICE", "auto"),
            )
        return app.state.service

    def get_history() -> Any:
        if app.state.history_service is None:
            app.state.history_service = HistoryService()
        return app.state.history_service

    @app.get("/", include_in_schema=False)
    def root() -> dict:
        return {
            "name": "WSB direction model API",
            "version": app.version,
            "docs": "/docs",
            "live_prediction": "POST /predict_direction",
            "history": ["/tickers", "/runs", "/prices/{ticker}",
                        "/predictions/{ticker}", "/chart/{ticker}"],
        }

    @app.get("/health")
    def health() -> dict:
        svc = app.state.service
        if svc is None:
            return {"status": "ok", "model_loaded": False,
                    "run_id": None, "device": None}
        info = svc.info() if hasattr(svc, "info") else {}
        return {"status": "ok", "model_loaded": True, "run_id": info.get("run_id"),
                "device": info.get("device")}

    @app.get("/model_info")
    def model_info() -> dict:
        svc = get_service()
        return svc.info() if hasattr(svc, "info") else {}

    @app.post("/predict_direction", response_model=PredictResponse)
    def predict_direction(req: PredictRequest) -> dict:
        if not (req.title.strip() or req.body.strip()):
            raise HTTPException(status_code=422, detail="title or body must contain text")
        svc = get_service()
        try:
            return svc.predict(
                title=req.title, body=req.body, ticker=req.ticker,
                created_utc=req.created_utc, score=req.score,
                num_comments=req.num_comments, flair=req.flair,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("prediction failed")
            raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

    @app.post("/predict_direction/batch")
    def predict_batch(items: list[PredictRequest]) -> dict:
        if len(items) > 100:
            raise HTTPException(status_code=422, detail="at most 100 items per batch")
        svc = get_service()
        out = []
        for req in items:
            try:
                out.append(svc.predict(
                    title=req.title, body=req.body, ticker=req.ticker,
                    created_utc=req.created_utc, score=req.score,
                    num_comments=req.num_comments, flair=req.flair,
                ))
            except ValueError as exc:
                out.append({"error": str(exc), "ticker": req.ticker})
        return {"n": len(out), "results": out}

    register_history_routes(app, get_history)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--preload", action="store_true", help="load the model before serving")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    setup_logging()
    run_dir = project_path("models") / args.run_id if args.run_id else latest_run()
    os.environ["WSB_RUN_DIR"] = str(run_dir)
    os.environ["WSB_DEVICE"] = args.device
    service = None
    if args.preload or args.temperature is not None:
        service = DirectionService(run_dir=run_dir, device=args.device,
                                   temperature=args.temperature)
    application = create_app(service)
    import uvicorn

    if args.reload:
        LOG.warning("--reload is ignored because the model is loaded in-process")
    uvicorn.run(application, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
