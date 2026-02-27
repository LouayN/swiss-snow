"""
Evidently drift monitoring for the Swiss Snow Quality project.

Compares training feature distribution (reference) against recent inference data (current).
Generates an HTML report and a JSON summary, then logs key metrics to MLflow.

Usage (standalone):
    python src/monitoring/drift_report.py
    python src/monitoring/drift_report.py --reference data/processed/features_train.parquet \
        --current data/processed/features_recent.parquet --output reports
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Add project root to path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.build_features import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DRIFT_THRESHOLD = 0.30   # flag if >30% of features drift
P_VALUE_THRESHOLD = 0.05  # K-S test threshold used by Evidently


def generate_drift_report(
    reference_path: str = "data/processed/features_train.parquet",
    current_path: str = "data/processed/features_recent.parquet",
    output_dir: str = "reports",
) -> dict:
    """
    Generate an Evidently drift report comparing training vs recent data.

    Returns a dict with:
        drift_share      — fraction of features that drifted (0–1)
        report_path      — absolute path to the HTML report
        summary_path     — absolute path to the JSON summary
        drifted_features — list of feature names that drifted
        drifted          — bool, True if drift_share > DRIFT_THRESHOLD
    """
    reference_path = Path(reference_path)
    current_path = Path(current_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference dataset not found: {reference_path}")
    if not current_path.exists():
        raise FileNotFoundError(f"Current dataset not found: {current_path}")

    log.info("Loading reference: %s", reference_path)
    reference = pd.read_parquet(reference_path)

    log.info("Loading current: %s", current_path)
    current = pd.read_parquet(current_path)

    # Keep only the feature columns present in both datasets
    shared_cols = [c for c in FEATURE_COLS if c in reference.columns and c in current.columns]
    missing = set(FEATURE_COLS) - set(shared_cols)
    if missing:
        log.warning("Columns not found in both datasets, skipping: %s", sorted(missing))

    reference = reference[shared_cols]
    current = current[shared_cols]

    log.info(
        "Running Evidently report: reference=%d rows, current=%d rows, features=%d",
        len(reference), len(current), len(shared_cols),
    )

    # Evidently v0.7+ API
    data_def = DataDefinition(numerical_columns=shared_cols)
    ref_ds = Dataset.from_pandas(reference, data_definition=data_def)
    cur_ds = Dataset.from_pandas(current, data_definition=data_def)

    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(reference_data=ref_ds, current_data=cur_ds)

    # Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = output_dir / f"drift_{timestamp}.html"
    snapshot.save_html(str(html_path))
    log.info("HTML report saved: %s", html_path)

    # Extract drift metrics from snapshot
    drift_results = _extract_drift_results(snapshot.dict(), shared_cols)
    drift_share = drift_results["drift_share"]
    drifted_features = drift_results["drifted_features"]

    summary = {
        "timestamp": timestamp,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "n_features": len(shared_cols),
        "drift_share": round(drift_share, 4),
        "drifted_features": drifted_features,
        "drifted": drift_share > DRIFT_THRESHOLD,
        "report_path": str(html_path.resolve()),
    }

    summary_path = output_dir / f"drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("JSON summary saved: %s", summary_path)

    if summary["drifted"]:
        log.warning(
            "DATA DRIFT DETECTED: %.1f%% of features drifted (threshold %.0f%%)",
            drift_share * 100, DRIFT_THRESHOLD * 100,
        )
        log.warning("Drifted features: %s", drifted_features)
    else:
        log.info(
            "No significant drift: %.1f%% of features drifted (threshold %.0f%%)",
            drift_share * 100, DRIFT_THRESHOLD * 100,
        )

    _log_to_mlflow(summary)

    summary["summary_path"] = str(summary_path.resolve())
    return summary


def _extract_drift_results(report_dict: dict, feature_cols: list) -> dict:
    """
    Parse the Evidently v0.7 snapshot dict to extract per-feature drift results.

    The snapshot contains:
      - DriftedColumnsCount metric: value = {"count": N, "share": S}
      - ValueDrift metrics per column: value = p_value (float); drift if p < threshold
    """
    drifted_features = []

    # Fast path: use DriftedColumnsCount if available
    share_from_summary = None
    for m in report_dict.get("metrics", []):
        name = m.get("metric_name", "")
        value = m.get("value")
        if "DriftedColumnsCount" in name and isinstance(value, dict):
            share_from_summary = value.get("share", 0.0)

    # Collect per-column drift using ValueDrift metrics
    col_set = set(feature_cols)
    for m in report_dict.get("metrics", []):
        name = m.get("metric_name", "")
        value = m.get("value")
        config = m.get("config", {})
        if "ValueDrift" in name and isinstance(value, (int, float)):
            col = config.get("column")
            if col and col in col_set:
                # p-value: if below threshold → drift
                if value < P_VALUE_THRESHOLD:
                    drifted_features.append(col)

    n_total = len(feature_cols)
    if drifted_features:
        drift_share = len(drifted_features) / n_total
    elif share_from_summary is not None:
        drift_share = share_from_summary
    else:
        drift_share = 0.0

    return {
        "drift_share": drift_share,
        "drifted_features": sorted(drifted_features),
    }


def _log_to_mlflow(summary: dict) -> None:
    """Log drift metrics to MLflow under the 'drift_monitoring' experiment."""
    try:
        mlflow.set_experiment("drift_monitoring")
        with mlflow.start_run(run_name=f"drift_{summary['timestamp']}"):
            mlflow.log_metric("drift_share", summary["drift_share"])
            mlflow.log_metric("n_drifted_features", len(summary["drifted_features"]))
            mlflow.log_metric("n_features", summary["n_features"])
            mlflow.log_metric("reference_rows", summary["reference_rows"])
            mlflow.log_metric("current_rows", summary["current_rows"])
            mlflow.log_param("drifted", str(summary["drifted"]))
            if summary["drifted_features"]:
                mlflow.log_param("drifted_features", ",".join(summary["drifted_features"]))
            mlflow.log_artifact(summary["report_path"])
        log.info("Drift metrics logged to MLflow (experiment: drift_monitoring)")
    except Exception as exc:
        log.warning("Could not log to MLflow: %s", exc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument(
        "--reference",
        default="data/processed/features_train.parquet",
        help="Path to reference (training) parquet",
    )
    parser.add_argument(
        "--current",
        default="data/processed/features_recent.parquet",
        help="Path to current (recent) parquet",
    )
    parser.add_argument(
        "--output",
        default="reports",
        help="Output directory for reports",
    )
    args = parser.parse_args()

    result = generate_drift_report(
        reference_path=args.reference,
        current_path=args.current,
        output_dir=args.output,
    )
    print(f"\nDrift share: {result['drift_share']:.1%}")
    print(f"Drifted:     {result['drifted']}")
    if result["drifted_features"]:
        print(f"Drifted features: {result['drifted_features']}")
    print(f"Report: {result['report_path']}")
