"""
Feature engineering pipeline for the Swiss snow quality model.

Reads: data/raw/meteoswiss_daily_historical.parquet
Writes: data/processed/features_train.parquet

If HOPSWORKS_API_KEY is set, also pushes features to the Hopsworks Feature Store.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Winter months only (snow season)
WINTER_MONTHS = [11, 12, 1, 2, 3, 4]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Keep only rows where snow depth was measured
    df = df[df["htoautd0"].notna()].copy()
    # Winter months only
    df = df[df["date"].dt.month.isin(WINTER_MONTHS)].copy()
    log.info("Loaded %d rows with snow depth (winter months)", len(df))
    return df


def _days_since_fresh_snow(series: pd.Series, threshold_cm: float = 2.0) -> pd.Series:
    """Compute days since last snowfall event (delta >= threshold) per sorted series."""
    is_fresh = series.diff().fillna(0) >= threshold_cm
    count = 999
    result = []
    for f in is_fresh:
        count = 0 if f else count + 1
        result.append(count)
    return pd.Series(result, index=series.index)


def _quality_score(depth, snow_delta, days_since_snow, temp, rain_48h, sunshine_3d=None):
    """
    Additive snow quality score v2 (0–100).

    Additive design means no single bad factor zeroes out deep-snowpack stations.
    Components:
      depth_score      0–65 pts  sqrt-scaled, 250cm → 65 pts  (fixes ceiling: true max = 100)
      temp_score     −10–+12 pts  cold = bonus, warm = penalty (narrower to reduce zero-inflation)
      fresh_score      0–15 pts  exp decay, half-life ~3.5 days
      delta_bonus      0– 5 pts  reward active accumulation (heavy snowfall days)
      rain_score     −15– 0 pts  warm-rain-only penalty (temp > 1°C); cold heavy precip = snow, not rain
      cold_sun_bonus   0– 3 pts  sunny powder days at altitude get a small bonus

    Key improvements vs v1:
    - Warm-rain-only penalty: cold heavy precipitation is snowfall, not rain.
      Cold precip days: v1 scored 31→ v2 scores 53 (correctly high, as those are snowfall days).
    - Fixed ceiling: v1 maxed at 90 (60+15+15), v2 reaches 100 (65+12+15+5+3).
    - Snow delta bonus: heavy accumulation days scored 40 in v1 vs 62 in v2.
    - Sunshine interaction: cold+sunny (powder day) gets a bonus; warm+sunny is neutral.
    """
    # 1. Depth component (0–65 pts): sqrt-scaled with 250cm ceiling
    depth_score = (np.sqrt(depth.clip(0, 250)) / np.sqrt(250) * 65).clip(0, 65)

    # 2. Temperature component (−10 to +12 pts): narrower than v1 to reduce zero-inflation
    temp_score = (-temp * 1.5).clip(-10, 12)

    # 3. Freshness component (0–15 pts): exp decay, half-life ~3.5 days
    fresh_score = (15 * np.exp(-days_since_snow / 5)).clip(0, 15)

    # 4. Accumulation bonus (0–5 pts): reward fresh snowfall events
    delta_bonus = (snow_delta.clip(0, 20) / 20 * 5).clip(0, 5)

    # 5. Warm-rain penalty (−15 to 0 pts): only penalize precipitation when temp > 1°C
    #    At or below 1°C, heavy precipitation is likely snow — no penalty applied
    warm_rain = rain_48h * (temp > 1.0).astype(float)
    rain_score = (-warm_rain * 2.0).clip(-15, 0)

    # 6. Cold+sunny bonus (0–3 pts): bright powder days at altitude
    if sunshine_3d is not None:
        cold_sun_bonus = (sunshine_3d * (-temp).clip(0, 10) * 0.03).clip(0, 3)
    else:
        cold_sun_bonus = 0

    raw = depth_score + temp_score + fresh_score + delta_bonus + rain_score + cold_sun_bonus
    return raw.clip(0, 100).round(1)


def compute_targets(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.sort_values("date").copy()

    g["snow_delta"] = g["htoautd0"].diff()
    g["rain_48h"] = g["rre150d0"].fillna(0).rolling(2).sum()
    g["days_since_snowfall"] = _days_since_fresh_snow(g["htoautd0"])

    # Current-day quality (used only to create the NEXT-DAY target)
    today_quality = _quality_score(
        g["htoautd0"],
        g["snow_delta"],
        g["days_since_snowfall"],
        g["tre200d0"],
        g["rain_48h"],
        sunshine_3d=g.get("sre000d0"),
    )

    # ---- Targets are NEXT-DAY values (predict tomorrow from today's context) ----
    # This avoids leakage: features are t, targets are t+1
    g["snow_quality_next"] = today_quality.shift(-1)        # tomorrow's score (regression)

    # is_good_ski_day: rain penalty conditioned on temperature (warm rain only)
    # At temp <= 1°C, heavy precipitation is snowfall — not a disqualifier
    warm_rain_next = g["rain_48h"].shift(-1) * (g["tre200d0"].shift(-1) > 1.0).astype(float)
    g["is_good_ski_day_next"] = (                            # tomorrow's binary label
        (g["htoautd0"].shift(-1) >= 20) &
        (g["tre200d0"].shift(-1) <= 3) &
        (warm_rain_next < 5)
    ).astype(float)  # float to allow NaN at end-of-series

    return g


def compute_features(grp: pd.DataFrame) -> pd.DataFrame:
    g = grp.sort_values("date").copy()

    # Temperature features
    g["temp_mean_3d"] = g["tre200d0"].rolling(3, min_periods=1).mean()
    g["temp_mean_7d"] = g["tre200d0"].rolling(7, min_periods=2).mean()
    g["temp_delta_3d"] = g["tre200d0"] - g["tre200d0"].shift(3)   # warming trend
    g["temp_min_7d"] = g["tre200d0"].rolling(7, min_periods=1).min()
    g["temp_max_3d"] = g["tre200d0"].rolling(3, min_periods=1).max()

    # Precipitation
    g["precip_3d"] = g["rre150d0"].fillna(0).rolling(3).sum()
    g["precip_7d"] = g["rre150d0"].fillna(0).rolling(7).sum()
    g["precip_14d"] = g["rre150d0"].fillna(0).rolling(14).sum()

    # Snow depth features
    g["snow_delta_3d"] = g["htoautd0"].diff(3)
    g["snow_depth_lag1"] = g["htoautd0"].shift(1)
    g["snow_depth_lag3"] = g["htoautd0"].shift(3)
    g["snow_depth_lag7"] = g["htoautd0"].shift(7)

    # Humidity and wind
    g["humidity_mean_3d"] = g["ure200d0"].rolling(3, min_periods=1).mean()
    g["wind_mean_3d"] = g["fkl010d0"].rolling(3, min_periods=1).mean()
    g["gust_max_3d"] = g["fkl010d1"].rolling(3, min_periods=1).max()

    # Sunshine
    g["sunshine_3d"] = g["sre000d0"].rolling(3, min_periods=1).mean()

    # Seasonal encoding (captures time-of-year non-linearly)
    doy = g["date"].dt.dayofyear
    g["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    g["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    g["month"] = g["date"].dt.month

    return g


FEATURE_COLS = [
    # Current conditions
    "htoautd0",           # snow depth (cm)
    "tre200d0",           # air temp
    "ure200d0",           # humidity
    "fkl010d0",           # wind speed
    "fkl010d1",           # gust max
    "sre000d0",           # sunshine duration
    # Rolling features
    "temp_mean_3d",
    "temp_mean_7d",
    "temp_delta_3d",
    "temp_min_7d",
    "temp_max_3d",
    "precip_3d",
    "precip_7d",
    "precip_14d",
    "snow_delta",
    "snow_delta_3d",
    "snow_depth_lag1",
    "snow_depth_lag3",
    "snow_depth_lag7",
    "humidity_mean_3d",
    "wind_mean_3d",
    "gust_max_3d",
    "sunshine_3d",
    "days_since_snowfall",
    "rain_48h",
    # Static
    "altitude",
    "doy_sin",
    "doy_cos",
    "month",
]

TARGET_COLS = ["snow_quality_next", "is_good_ski_day_next"]
META_COLS = ["station_id", "name", "date", "lat", "lon"]


def build(raw_path: Path, out_path: Path) -> pd.DataFrame:
    df = load_raw(raw_path)

    log.info("Computing targets…")
    df = df.groupby("station_id", group_keys=False).apply(compute_targets)

    log.info("Computing features…")
    df = df.groupby("station_id", group_keys=False).apply(compute_features)

    keep = META_COLS + FEATURE_COLS + TARGET_COLS
    df_out = df[[c for c in keep if c in df.columns]].dropna(
        subset=["snow_quality_next", "htoautd0", "tre200d0"]
    )
    # Convert is_good_ski_day_next to int after dropping NaN rows
    df_out["is_good_ski_day_next"] = df_out["is_good_ski_day_next"].astype(int)

    log.info(
        "Final dataset: %d rows, %.1f%% good-ski-days-next",
        len(df_out),
        100 * df_out["is_good_ski_day_next"].mean(),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    log.info("Saved → %s", out_path)

    if os.environ.get("HOPSWORKS_API_KEY"):
        try:
            sys.path.insert(0, str(Path(__file__).parents[2]))
            from hopsworks_utils import push_training_features
            push_training_features(df_out)
        except Exception as exc:
            log.warning("Hopsworks feature push failed (non-fatal): %s", exc)

    return df_out


def build_recent(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Build features_recent.parquet from the raw recent parquet.

    Mirrors the same compute_targets + compute_features pipeline used by app.py
    and the API so that drift monitoring compares an identical feature schema.

    Differences from build() (training):
    - No winter-month filter (keep all recent dates as-is)
    - No htoautd0 filter (keep all 75 stations; non-sensor rows have NaN features)
    - Targets are included but will be NaN for the last row of each station
      (no next-day observation available) — fine for drift monitoring.
    """
    log.info("Building recent features from %s", raw_path)
    df = pd.read_parquet(raw_path)
    log.info("Loaded %d raw rows, %d stations", len(df), df["station_id"].nunique())

    log.info("Computing targets…")
    df = df.groupby("station_id", group_keys=False).apply(compute_targets)

    log.info("Computing features…")
    df = df.groupby("station_id", group_keys=False).apply(compute_features)

    keep = META_COLS + FEATURE_COLS + TARGET_COLS
    df_out = df[[c for c in keep if c in df.columns]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    log.info(
        "Recent features saved → %s  (%d rows, %d stations, %d/%d feature cols present)",
        out_path,
        len(df_out),
        df_out["station_id"].nunique(),
        sum(c in df_out.columns for c in FEATURE_COLS),
        len(FEATURE_COLS),
    )

    if os.environ.get("HOPSWORKS_API_KEY"):
        try:
            sys.path.insert(0, str(Path(__file__).parents[2]))
            from hopsworks_utils import push_recent_features
            push_recent_features(df_out)
        except Exception as exc:
            log.warning("Hopsworks recent push failed (non-fatal): %s", exc)

    return df_out


if __name__ == "__main__":
    build(
        raw_path=Path("data/raw/meteoswiss_daily_historical.parquet"),
        out_path=Path("data/processed/features_train.parquet"),
    )
    build_recent(
        raw_path=Path("data/raw/meteoswiss_daily_recent.parquet"),
        out_path=Path("data/processed/features_recent.parquet"),
    )
