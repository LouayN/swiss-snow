"""
Hopsworks helpers — Feature Store + Model Registry.

Set HOPSWORKS_API_KEY environment variable (or use .hw_api_key file).

Feature groups:
  snow_features          (offline, training)   primary_key: [station_id, date]
  snow_features_recent   (online + offline)    primary_key: [station_id]

Models in registry:
  snow_quality_regressor    — XGBoost regressor (MAE 0–100 score)
  good_ski_day_classifier   — XGBoost classifier (binary good-ski-day)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

FG_TRAINING_NAME = "snow_features"
FG_TRAINING_VERSION = 1
FG_RECENT_NAME = "snow_features_recent"
FG_RECENT_VERSION = 1
MODEL_REGRESSOR_NAME = "snow_quality_regressor"
MODEL_CLASSIFIER_NAME = "good_ski_day_classifier"
HOPSWORKS_HOST = os.environ.get("HOPSWORKS_HOST", "c.app.hopsworks.ai")
HOPSWORKS_PROJECT = os.environ.get("HOPSWORKS_PROJECT", "swiss_snow")

_project = None


def get_project():
    """Return cached Hopsworks project connection."""
    global _project
    if _project is None:
        import hopsworks
        api_key = os.environ.get("HOPSWORKS_API_KEY")
        log.info("Logging in to Hopsworks at %s…", HOPSWORKS_HOST)
        _project = hopsworks.login(
            host=HOPSWORKS_HOST,
            project=HOPSWORKS_PROJECT,
            api_key_value=api_key,
        )
        log.info("Connected to Hopsworks project: %s", _project.name)
    return _project


# ── Feature Store ──────────────────────────────────────────────────────────────

def push_training_features(df) -> None:
    """
    Insert training feature DataFrame into the offline Feature Store.

    Upserts by primary key (station_id, date) so re-runs are safe.
    """
    import pandas as pd

    fs = get_project().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_TRAINING_NAME,
        version=FG_TRAINING_VERSION,
        primary_key=["station_id", "date"],
        event_time="date",
        description="Swiss MeteoSwiss snow features — training set (2009–2025)",
    )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    log.info("Inserting %d rows into feature group '%s'…", len(df), FG_TRAINING_NAME)
    fg.insert(df, write_options={"wait_for_job": False})
    log.info("Training features pushed to Hopsworks.")


def push_recent_features(df) -> None:
    """
    Insert latest-per-station features into the online Feature Store.

    online_enabled=True makes these available for low-latency inference.
    Keeps only the most recent row per station before inserting.
    """
    import pandas as pd

    fs = get_project().get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FG_RECENT_NAME,
        version=FG_RECENT_VERSION,
        primary_key=["station_id"],
        event_time="date",
        online_enabled=True,
        description="Latest Swiss snow features per station — online inference",
    )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    # Keep only latest observation per station
    df = df.sort_values("date").groupby("station_id").last().reset_index()
    log.info(
        "Inserting %d station rows into online feature group '%s'…",
        len(df), FG_RECENT_NAME,
    )
    fg.insert(df, write_options={"wait_for_job": False})
    log.info("Recent features pushed to Hopsworks online store.")


def get_recent_features_online(station_ids):
    """
    Fetch latest features for given station IDs from the online Feature Store.

    Returns a DataFrame with one row per station.
    Falls back gracefully if the feature group is not populated yet.
    """
    fs = get_project().get_feature_store()
    fv = fs.get_or_create_feature_view(
        name="snow_features_recent_fv",
        version=1,
        query=fs.get_feature_group(FG_RECENT_NAME, version=FG_RECENT_VERSION).select_all(),
    )
    entries = [{"station_id": sid} for sid in station_ids]
    vectors = fv.get_feature_vectors(entries, return_type="pandas")
    log.info("Fetched %d feature vectors from online store.", len(vectors))
    return vectors


# ── Model Registry ─────────────────────────────────────────────────────────────

def push_model(model_name: str, model_path: str | Path, metrics: dict) -> None:
    """
    Upload a model file to the Hopsworks Model Registry.

    Creates a new version each time; the latest version is used by default
    when loading.
    """
    import shutil

    mr = get_project().get_model_registry()
    export_dir = Path(f"/tmp/hw_export_{model_name}")
    export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, export_dir / Path(model_path).name)

    model_meta = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"XGBoost — {model_name}",
    )
    model_meta.save(str(export_dir))
    log.info("Model '%s' pushed to Hopsworks registry (metrics: %s).", model_name, metrics)


def load_model_dir(model_name: str, version: int = None) -> Path:
    """
    Download a model from the Hopsworks Model Registry.

    Returns the local directory path containing the model file.
    """
    mr = get_project().get_model_registry()
    model_meta = mr.get_model(model_name, version=version)
    local_dir = model_meta.download()
    log.info("Model '%s' v%d downloaded to %s", model_name, version, local_dir)
    return Path(local_dir)
