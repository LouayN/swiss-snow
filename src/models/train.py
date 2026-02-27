"""
Train XGBoost models for Swiss snow quality prediction.

Two models:
  1. snow_quality_regressor  — predicts quality score 0–100
  2. good_ski_day_classifier — predicts P(is_good_ski_day)

Uses time-based train/val split (train on 2009–2022, validate on 2023–2025).
Tracks all experiments with MLflow.

Usage:
    python3 src/models/train.py
"""

import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    r2_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "htoautd0", "tre200d0", "ure200d0", "fkl010d0", "fkl010d1", "sre000d0",
    "temp_mean_3d", "temp_mean_7d", "temp_delta_3d", "temp_min_7d", "temp_max_3d",
    "precip_3d", "precip_7d", "precip_14d",
    "snow_delta", "snow_delta_3d", "snow_depth_lag1", "snow_depth_lag3", "snow_depth_lag7",
    "humidity_mean_3d", "wind_mean_3d", "gust_max_3d", "sunshine_3d",
    "days_since_snowfall", "rain_48h",
    "altitude", "doy_sin", "doy_cos", "month",
]

VAL_START = "2023-01-01"
MODELS_DIR = Path("models")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["date"] < VAL_START].copy()
    val = df[df["date"] >= VAL_START].copy()
    log.info(
        "Train: %d rows (%s–%s) | Val: %d rows (%s–%s)",
        len(train), train["date"].min().date(), train["date"].max().date(),
        len(val), val["date"].min().date(), val["date"].max().date(),
    )
    return train, val


def get_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    return df[feats], df[target]


# ──────────────────────────────────────────────
# 1. Snow quality regressor
# ──────────────────────────────────────────────

def train_regressor(train: pd.DataFrame, val: pd.DataFrame) -> xgb.XGBRegressor:
    X_tr, y_tr = get_xy(train, "snow_quality_next")
    X_val, y_val = get_xy(val, "snow_quality_next")

    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        random_state=42,
        eval_metric="mae",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="snow_quality_regressor"):
        mlflow.log_params(params)
        mlflow.log_param("val_start", VAL_START)
        mlflow.log_param("n_features", len(X_tr.columns))
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_val", len(X_val))

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        preds = model.predict(X_val).clip(0, 100)
        mae = mean_absolute_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        log.info("Regressor — MAE: %.2f  R²: %.3f", mae, r2)
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_r2", r2)
        mlflow.log_metric("best_iteration", model.best_iteration)

        # Feature importance
        imp = pd.Series(model.feature_importances_, index=X_tr.columns)
        top5 = imp.nlargest(5).to_dict()
        mlflow.log_params({f"fi_{k}": round(v, 4) for k, v in top5.items()})

        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "snow_quality_regressor.json"
        model.save_model(model_path)
        mlflow.xgboost.log_model(model, "snow_quality_regressor")
        log.info("Saved → %s", model_path)

        if os.environ.get("HOPSWORKS_API_KEY"):
            try:
                sys.path.insert(0, str(Path(__file__).parents[2]))
                from hopsworks_utils import push_model
                push_model("snow_quality_regressor", model_path, {"mae": round(mae, 4), "r2": round(r2, 4)})
            except Exception as exc:
                log.warning("Hopsworks model push failed (non-fatal): %s", exc)

    return model


# ──────────────────────────────────────────────
# 2. Good ski day classifier
# ──────────────────────────────────────────────

def train_classifier(train: pd.DataFrame, val: pd.DataFrame) -> xgb.XGBClassifier:
    X_tr, y_tr = get_xy(train, "is_good_ski_day_next")
    X_val, y_val = get_xy(val, "is_good_ski_day_next")

    # Handle class imbalance with scale_pos_weight
    pos = y_tr.sum()
    neg = len(y_tr) - pos
    spw = neg / pos
    log.info("Classifier — scale_pos_weight: %.2f  (pos=%d neg=%d)", spw, pos, neg)

    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        reg_lambda=1.0,
        reg_alpha=0.1,
        use_label_encoder=False,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="good_ski_day_classifier"):
        mlflow.log_params({k: v for k, v in params.items() if k != "use_label_encoder"})
        mlflow.log_param("val_start", VAL_START)
        mlflow.log_param("n_train", len(X_tr))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("val_positive_rate", round(y_val.mean(), 3))

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        ap = average_precision_score(y_val, proba)
        brier = brier_score_loss(y_val, proba)

        log.info(
            "Classifier — AUC-ROC: %.3f  Avg-Precision: %.3f  Brier: %.4f",
            auc, ap, brier,
        )
        mlflow.log_metric("val_auc_roc", auc)
        mlflow.log_metric("val_avg_precision", ap)
        mlflow.log_metric("val_brier", brier)
        mlflow.log_metric("best_iteration", model.best_iteration)

        imp = pd.Series(model.feature_importances_, index=X_tr.columns)
        top5 = imp.nlargest(5).to_dict()
        mlflow.log_params({f"fi_{k}": round(v, 4) for k, v in top5.items()})

        model_path = MODELS_DIR / "good_ski_day_classifier.json"
        model.save_model(model_path)
        mlflow.xgboost.log_model(model, "good_ski_day_classifier")
        log.info("Saved → %s", model_path)

        if os.environ.get("HOPSWORKS_API_KEY"):
            try:
                sys.path.insert(0, str(Path(__file__).parents[2]))
                from hopsworks_utils import push_model
                push_model(
                    "good_ski_day_classifier",
                    model_path,
                    {"auc_roc": round(auc, 4), "avg_precision": round(ap, 4), "brier": round(brier, 4)},
                )
            except Exception as exc:
                log.warning("Hopsworks model push failed (non-fatal): %s", exc)

    return model


if __name__ == "__main__":
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("swiss_snow")

    df = load_data(Path("data/processed/features_train.parquet"))
    train, val = split(df)

    log.info("Training snow quality regressor…")
    reg = train_regressor(train, val)

    log.info("Training good ski day classifier…")
    clf = train_classifier(train, val)

    log.info("Done. View results: mlflow ui --port 5000")
