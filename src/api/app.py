"""
Swiss Snow Quality API — FastAPI serving endpoint.

GET /predict/{station_id}
  → returns tomorrow's snow_quality score + good_ski_day probability

GET /forecast/{station_id}
  → returns 7-day forecast using OpenMeteo + current station conditions

GET /stations
  → returns all station metadata with latest predictions

Run:
    uvicorn src.api.app:app --reload --port 8000
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger(__name__)

app = FastAPI(
    title="Swiss Snow Quality API",
    description="Predict snow quality scores for Swiss alpine stations",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parents[2]
MODELS_DIR = BASE / "models"
FEATURES_PATH = BASE / "data" / "processed" / "features_train.parquet"
RECENT_PATH = BASE / "data" / "raw" / "meteoswiss_daily_recent.parquet"

FEATURE_COLS = [
    "htoautd0", "tre200d0", "ure200d0", "fkl010d0", "fkl010d1", "sre000d0",
    "temp_mean_3d", "temp_mean_7d", "temp_delta_3d", "temp_min_7d", "temp_max_3d",
    "precip_3d", "precip_7d", "precip_14d",
    "snow_delta", "snow_delta_3d", "snow_depth_lag1", "snow_depth_lag3", "snow_depth_lag7",
    "humidity_mean_3d", "wind_mean_3d", "gust_max_3d", "sunshine_3d",
    "days_since_snowfall", "rain_48h",
    "altitude", "doy_sin", "doy_cos", "month",
]


# ── model loading ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_models():
    import os
    if os.environ.get("HOPSWORKS_API_KEY"):
        try:
            import sys
            sys.path.insert(0, str(BASE))
            from hopsworks_utils import load_model_dir, MODEL_REGRESSOR_NAME, MODEL_CLASSIFIER_NAME
            reg_dir = load_model_dir(MODEL_REGRESSOR_NAME)
            clf_dir = load_model_dir(MODEL_CLASSIFIER_NAME)
            reg = xgb.XGBRegressor()
            reg.load_model(next(reg_dir.glob("*.json")))
            clf = xgb.XGBClassifier()
            clf.load_model(next(clf_dir.glob("*.json")))
            log.info("Models loaded from Hopsworks registry.")
            return reg, clf
        except Exception as exc:
            log.warning("Hopsworks model load failed, falling back to local: %s", exc)
    reg = xgb.XGBRegressor()
    reg.load_model(MODELS_DIR / "snow_quality_regressor.json")
    clf = xgb.XGBClassifier()
    clf.load_model(MODELS_DIR / "good_ski_day_classifier.json")
    log.info("Models loaded from local files.")
    return reg, clf


@lru_cache(maxsize=1)
def load_station_features() -> pd.DataFrame:
    """Load the most recent feature row per station from the recent parquet."""
    import sys
    sys.path.insert(0, str(BASE / "src"))
    from features.build_features import compute_targets, compute_features

    df = pd.read_parquet(RECENT_PATH)
    df = df.groupby("station_id", group_keys=False).apply(compute_targets)
    df = df.groupby("station_id", group_keys=False).apply(compute_features)

    # Latest row per station (most recent observation)
    latest = df.sort_values("date").groupby("station_id").last().reset_index()
    return latest


@lru_cache(maxsize=1)
def load_stations_meta() -> pd.DataFrame:
    latest = load_station_features()
    return latest[["station_id", "name", "lat", "lon", "altitude"]].copy()


# ── response models ───────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    altitude: float
    observation_date: str
    snow_depth_cm: float | None
    temperature_c: float | None
    snow_quality_score: float          # 0–100
    good_ski_day_probability: float    # 0–1
    snow_quality_label: str            # "excellent" / "good" / "fair" / "poor"
    days_since_snowfall: int | None


class StationSummary(BaseModel):
    station_id: str
    name: str
    lat: float
    lon: float
    altitude: float
    snow_quality_score: float
    good_ski_day_probability: float
    snow_quality_label: str


def _quality_label(score: float) -> str:
    if score >= 65:
        return "excellent"
    elif score >= 40:
        return "good"
    elif score >= 20:
        return "fair"
    return "poor"


def _predict_row(row: pd.Series) -> tuple[float, float]:
    reg, clf = load_models()
    feats = [c for c in FEATURE_COLS if c in row.index]
    X = pd.DataFrame([row[feats].values], columns=feats)
    quality = float(reg.predict(X).clip(0, 100)[0])
    prob = float(clf.predict_proba(X)[0, 1])
    return round(quality, 1), round(prob, 3)


# ── endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stations", response_model=list[StationSummary])
def get_stations():
    """All stations with their latest predicted snow quality."""
    df = load_station_features()
    results = []
    reg, clf = load_models()
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feats].fillna(0)
    qualities = reg.predict(X).clip(0, 100)
    probs = clf.predict_proba(X)[:, 1]

    for i, row in df.iterrows():
        results.append(StationSummary(
            station_id=row["station_id"],
            name=row.get("name", ""),
            lat=round(float(row["lat"]), 5),
            lon=round(float(row["lon"]), 5),
            altitude=float(row["altitude"]),
            snow_quality_score=round(float(qualities[i]), 1),
            good_ski_day_probability=round(float(probs[i]), 3),
            snow_quality_label=_quality_label(qualities[i]),
        ))
    return sorted(results, key=lambda x: -x.snow_quality_score)


@app.get("/predict/{station_id}", response_model=PredictionResponse)
def predict(station_id: str):
    """Predict tomorrow's snow quality for a single station."""
    sid = station_id.upper()
    df = load_station_features()
    row_df = df[df["station_id"] == sid]
    if row_df.empty:
        raise HTTPException(404, f"Station '{sid}' not found or has no snow depth data")
    row = row_df.iloc[0]
    quality, prob = _predict_row(row)

    return PredictionResponse(
        station_id=sid,
        name=row.get("name", ""),
        lat=round(float(row["lat"]), 5),
        lon=round(float(row["lon"]), 5),
        altitude=float(row["altitude"]),
        observation_date=str(row["date"].date()),
        snow_depth_cm=float(row["htoautd0"]) if pd.notna(row.get("htoautd0")) else None,
        temperature_c=float(row["tre200d0"]) if pd.notna(row.get("tre200d0")) else None,
        snow_quality_score=quality,
        good_ski_day_probability=prob,
        snow_quality_label=_quality_label(quality),
        days_since_snowfall=int(row["days_since_snowfall"]) if pd.notna(row.get("days_since_snowfall")) else None,
    )
