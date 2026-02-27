"""
Prefect flow: Swiss Snow weekly retrain pipeline.

Pipeline steps (run in sequence):
  1. fetch_recent_data  — download latest MeteoSwiss observations
  2. build_features     — recompute feature parquet from raw data
  3. run_drift_report   — Evidently drift check (warning only, does not block retrain)
  4. retrain_models     — retrain XGBoost models, log to MLflow

Schedule: every Monday at 06:00 (cron: "0 6 * * 1")

Running:
  # One-off manual run:
  python pipelines/weekly_retrain.py --run-now

  # Start scheduled loop (runs indefinitely, triggers every Monday 06:00):
  python pipelines/weekly_retrain.py
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger
from prefect.schedules import Cron

# Add project root to path so we can import drift_report
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="fetch-recent-data", retries=2, retry_delay_seconds=60)
def fetch_recent_data() -> None:
    """Download the latest MeteoSwiss recent observations (~54 days)."""
    logger = get_run_logger()
    logger.info("Fetching recent MeteoSwiss data…")
    result = subprocess.run(
        [sys.executable, "src/data/fetch_meteoswiss.py", "recent"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"fetch_meteoswiss.py failed (exit {result.returncode})")
    logger.info("Recent data fetch complete.")


@task(name="build-features", retries=1)
def build_features() -> None:
    """Recompute features_train.parquet and features_recent.parquet."""
    logger = get_run_logger()
    logger.info("Building features…")
    result = subprocess.run(
        [sys.executable, "src/features/build_features.py"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"build_features.py failed (exit {result.returncode})")
    logger.info("Features built.")


@task(name="drift-report")
def run_drift_report() -> dict:
    """Run Evidently drift check. Logs a warning if significant drift is found."""
    logger = get_run_logger()
    logger.info("Running drift report…")

    try:
        from src.monitoring.drift_report import generate_drift_report
        result = generate_drift_report()
        drift_share = result["drift_share"]
        if result["drifted"]:
            logger.warning(
                "DATA DRIFT DETECTED: %.1f%% of features drifted. "
                "Drifted features: %s. "
                "Report: %s",
                drift_share * 100,
                result["drifted_features"],
                result["report_path"],
            )
        else:
            logger.info(
                "No significant drift detected (%.1f%% of features drifted).",
                drift_share * 100,
            )
        return result
    except Exception as exc:
        # Drift report failure must not block retraining
        logger.warning("Drift report failed (non-fatal): %s", exc)
        return {"drift_share": None, "drifted": False, "drifted_features": [], "report_path": None}


@task(name="push-features-hopsworks")
def push_features_hopsworks() -> None:
    """Push training + recent features to Hopsworks Feature Store (skipped if no API key)."""
    import os
    logger = get_run_logger()

    if not os.environ.get("HOPSWORKS_API_KEY"):
        logger.info("HOPSWORKS_API_KEY not set — skipping feature store push.")
        return

    logger.info("Pushing features to Hopsworks…")
    try:
        import pandas as pd
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from hopsworks_utils import push_training_features, push_recent_features

        train_path = Path("data/processed/features_train.parquet")
        recent_path = Path("data/processed/features_recent.parquet")

        if train_path.exists():
            push_training_features(pd.read_parquet(train_path))
            logger.info("Training features pushed.")
        if recent_path.exists():
            push_recent_features(pd.read_parquet(recent_path))
            logger.info("Recent features pushed to online store.")
    except Exception as exc:
        logger.warning("Hopsworks push failed (non-fatal): %s", exc)


@task(name="retrain-models", retries=1)
def retrain_models() -> None:
    """Retrain both XGBoost models and log metrics to MLflow."""
    logger = get_run_logger()
    logger.info("Retraining models…")
    result = subprocess.run(
        [sys.executable, "src/models/train.py"],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"train.py failed (exit {result.returncode})")
    logger.info("Model retraining complete.")


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="swiss-snow-weekly-retrain", log_prints=True)
def weekly_retrain() -> None:
    """
    Full weekly retrain pipeline:
      fetch → features → drift check → retrain
    """
    fetch_recent_data()
    build_features()
    push_features_hopsworks()
    run_drift_report()
    retrain_models()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swiss Snow weekly retrain pipeline")
    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Execute the flow once immediately instead of starting the scheduler",
    )
    args = parser.parse_args()

    if args.run_now:
        weekly_retrain()
    else:
        # Start the Prefect serve loop — triggers on schedule without a Prefect server
        weekly_retrain.serve(
            name="weekly-retrain-schedule",
            schedules=[Cron("0 6 * * 1", timezone="Europe/Zurich")],
        )
