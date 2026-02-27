---
title: Swiss Snow Quality
emoji: ❄️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Swiss Snow Quality Predictor

Predicts **snow quality** and **good ski day probability** for 75 Swiss alpine stations, using MeteoSwiss official open data and XGBoost models — served via an interactive map.

---

## Demo

![Streamlit map showing snow quality per station](.github/assets/demo.png)

> Interactive PyDeck map — each station bubble is colored by predicted snow quality (0–100).

---

## Architecture

```
MeteoSwiss OGD ──┐
                  ├──► Feature Pipeline ──► Hopsworks Feature Store
OpenMeteo ERA5 ──┘          │
                             ▼
                      XGBoost Training ──► Hopsworks Model Registry
                                                    │
                                                    ▼
                                           Streamlit Map (app.py)
```

If `HOPSWORKS_API_KEY` is not set, everything runs locally — features write to `data/processed/`, models save to `models/`. Hopsworks is purely additive.

---

## Data Sources

| Source | Coverage | Key variable |
|---|---|---|
| [MeteoSwiss OGD](https://data.geo.admin.ch) | 75 stations, 1863–2025 | `htoautd0` — snow depth (cm) |
| OpenMeteo ERA5 | Global reanalysis | Temperature, precipitation |

No API keys required.

---

## Models

| Model | Type | MAE | R² / AUC |
|---|---|---|---|
| `snow_quality_regressor` | XGBoost | 4.48 | R²=0.877 |
| `good_ski_day_classifier` | XGBoost | — | AUC=0.986, AP=0.962 |

- **Target**: next-day (t+1) to prevent label leakage
- **Train**: 2009–2022 (49K rows) | **Val**: 2023–2025 (13K rows)
- **Snow quality score** (0–100): composite of snow depth, recent snowfall delta, temperature, cold+sunny bonus, warm-rain penalty
- **Good ski day**: binary label using warm-rain conditioning — `rain_48h × (temp > 1°C) < 5mm` → 36.4% positive rate

---

## MLOps

### Feature Store + Model Registry — Hopsworks

Set `HOPSWORKS_API_KEY` to activate. All pipelines degrade gracefully without it.

| Feature group | Primary key | Store |
|---|---|---|
| `snow_features` v1 | `[station_id, date]` | Offline (training) |
| `snow_features_recent` v1 | `[station_id]` | Online + Offline (inference) |

Models are versioned in the Hopsworks Model Registry. `app.py` loads from the registry on startup and falls back to local JSON files if unavailable.

### Experiment Tracking — MLflow

All training runs logged: hyperparameters, metrics, and model artifacts.

```bash
mlflow ui --port 5000
```

### Automated Retraining — Prefect 3

Weekly pipeline every **Monday 06:00 Zurich** time:

```
fetch fresh data → rebuild features → push to Hopsworks → drift check → retrain → log to MLflow
```

```bash
python pipelines/weekly_retrain.py           # run on schedule
python pipelines/weekly_retrain.py --run-now # trigger immediately
```

### Data & Model Drift Monitoring — Evidently

Generates HTML + JSON drift reports using `DataDriftPreset` + `DataSummaryPreset`. Reports saved to `reports/` and logged to the `drift_monitoring` MLflow experiment.

```bash
python src/monitoring/drift_report.py
```

### CI — GitHub Actions

`ci.yml` runs on every push and PR: lint (ruff) + smoke-test imports.

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch latest data
python src/data/fetch_meteoswiss.py recent

# Rebuild features  (also pushes to Hopsworks if HOPSWORKS_API_KEY is set)
python src/features/build_features.py

# Retrain models    (also pushes to Hopsworks Model Registry if key is set)
python src/models/train.py

# Start map frontend
streamlit run app.py
```

To activate Hopsworks (optional):

```bash
export HOPSWORKS_API_KEY=your_key_here   # from app.hopsworks.ai → Settings → API Keys
```

### With Docker (local only)

```bash
docker compose build
docker compose up        # API on :8000, Streamlit on :8501
```

---

## Deployment (free)

| Service | Provider | Cost |
|---|---|---|
| Streamlit frontend | [Hugging Face Spaces](https://huggingface.co/spaces) | $0 |
| Feature store + model registry | [Hopsworks Serverless](https://app.hopsworks.ai) | $0 |
| Automated weekly retraining | GitHub Actions | $0 |

**One-time setup:**
1. Create a Hugging Face Space (Streamlit SDK)
2. Add `HOPSWORKS_API_KEY` as a Space Secret (Settings tab)
3. `git remote add space https://huggingface.co/spaces/<username>/swiss-snow`
4. `git push space main`

**GitHub Actions secrets required** (for weekly retrain):
- `HOPSWORKS_API_KEY` — pushes retrained models to Hopsworks registry
- `HF_TOKEN` — restarts the Space so it loads the new model
- `HF_SPACE` — Space ID in `username/space-name` format (Actions variable, not secret)

---

## Project Structure

```
swiss_snow/
├── src/
│   ├── data/
│   │   ├── fetch_meteoswiss.py     # MeteoSwiss OGD fetcher
│   │   └── fetch_openmeteo.py      # OpenMeteo forecast + ERA5 fetcher
│   ├── features/
│   │   └── build_features.py       # Feature pipeline + target computation
│   ├── models/
│   │   └── train.py                # XGBoost training + MLflow logging
│   ├── api/
│   │   └── app.py                  # FastAPI serving endpoint
│   └── monitoring/
│       └── drift_report.py         # Evidently drift reports
├── pipelines/
│   └── weekly_retrain.py           # Prefect 3 weekly retrain flow
├── models/
│   ├── snow_quality_regressor.json
│   └── good_ski_day_classifier.json
├── data/
│   ├── raw/                        # Parquet files from MeteoSwiss
│   └── processed/                  # Engineered feature sets
├── reports/                        # Drift report outputs (HTML + JSON)
├── app.py                          # Streamlit + PyDeck map frontend
├── hopsworks_utils.py              # Hopsworks Feature Store + Model Registry helpers
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Tech Stack

- **Data**: MeteoSwiss OGD, OpenMeteo ERA5
- **Features**: pandas, numpy
- **Models**: XGBoost
- **Feature store / model registry**: Hopsworks
- **Experiment tracking**: MLflow
- **Orchestration**: Prefect 3
- **Monitoring**: Evidently
- **API**: FastAPI
- **Frontend**: Streamlit, PyDeck
- **Containerisation**: Docker, Docker Compose
- **CI**: GitHub Actions
