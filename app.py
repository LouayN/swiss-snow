"""
Swiss Snow Quality — Streamlit + PyDeck interactive map.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import pydeck as pdk
import streamlit as st
import xgboost as xgb

from src.features.build_features import (
    FEATURE_COLS,
    compute_features,
    compute_targets,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Swiss Snow Quality",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent


# ── helpers ──────────────────────────────────────────────────────────────────
def quality_label(s) -> str:
    if pd.isna(s): return "No data"
    if s >= 65:    return "Excellent"
    if s >= 40:    return "Good"
    if s >= 20:    return "Fair"
    return "Poor"


def score_to_color(s) -> list[int]:
    """RGBA colour per quality label. No-sensor stations are near-transparent."""
    if pd.isna(s):  return [140, 140, 150, 80]   # grey, faded  – no sensor
    if s >= 65:     return [10,  80,  200, 230]   # deep blue    – excellent
    if s >= 40:     return [70, 145,  220, 210]   # mid blue     – good
    if s >= 20:     return [150, 195, 240, 180]   # light blue   – fair
    return          [190, 190, 200, 150]           # light grey   – poor


# ── load data & models ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import os
    if os.environ.get("HOPSWORKS_API_KEY"):
        try:
            from hopsworks_utils import load_model_dir
            reg_dir = load_model_dir("snow_quality_regressor")
            clf_dir = load_model_dir("good_ski_day_classifier")
            reg = xgb.XGBRegressor()
            reg.load_model(reg_dir / "snow_quality_regressor.json")
            clf = xgb.XGBClassifier()
            clf.load_model(clf_dir / "good_ski_day_classifier.json")
            return reg, clf
        except Exception as exc:
            st.warning(f"Hopsworks model load failed, using local files: {exc}")

    reg = xgb.XGBRegressor()
    reg.load_model(BASE / "models" / "snow_quality_regressor.json")
    clf = xgb.XGBClassifier()
    clf.load_model(BASE / "models" / "good_ski_day_classifier.json")
    return reg, clf


def _ensure_recent_data():
    """Fetch recent MeteoSwiss data if the parquet doesn't exist (e.g. fresh HF Space)."""
    recent_path = BASE / "data" / "raw" / "meteoswiss_daily_recent.parquet"
    if recent_path.exists():
        return
    with st.spinner("Fetching latest MeteoSwiss snow data… (first load only, ~60s)"):
        sys.path.insert(0, str(BASE / "src"))
        from data.fetch_meteoswiss import fetch_all_snow_stations, MIN_ALTITUDE
        df_raw = fetch_all_snow_stations(
            min_altitude=MIN_ALTITUDE,
            period="recent",
            cache_dir=BASE / "data" / "raw" / "meteoswiss",
        )
        recent_path.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_parquet(recent_path, index=False)


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame:
    _ensure_recent_data()
    reg, clf = load_models()

    df = pd.read_parquet(BASE / "data" / "raw" / "meteoswiss_daily_recent.parquet")
    df = df.groupby("station_id", group_keys=False).apply(compute_targets)
    df = df.groupby("station_id", group_keys=False).apply(compute_features)
    latest = df.sort_values("date").groupby("station_id").last().reset_index()

    # Flag stations that have a snow depth sensor
    latest["has_sensor"] = latest["htoautd0"].notna()

    # Only predict quality where we have actual snow depth data
    sensor_mask = latest["has_sensor"]
    feats = [c for c in FEATURE_COLS if c in latest.columns]

    latest["snow_quality"] = None
    latest["ski_day_prob"] = None

    if sensor_mask.any():
        X = latest.loc[sensor_mask, feats].fillna(0)
        latest.loc[sensor_mask, "snow_quality"] = reg.predict(X).clip(0, 100).round(1)
        latest.loc[sensor_mask, "ski_day_prob"] = clf.predict_proba(X)[:, 1].round(3)

    latest["snow_quality"] = pd.to_numeric(latest["snow_quality"])
    latest["ski_day_prob"] = pd.to_numeric(latest["ski_day_prob"])
    latest["quality_label"] = latest["snow_quality"].map(quality_label)
    return latest


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("❄️ Swiss Snow Quality")
    st.caption("Predicting tomorrow's ski conditions across Switzerland")
    st.divider()

    min_alt = st.slider("Min altitude (m)", 800, 3000, 1000, step=100)
    show_no_sensor = st.checkbox("Show stations without snow sensor", value=True)
    show_labels = st.checkbox("Show station ID labels", value=False)

    st.divider()
    st.markdown("""
**Score guide**
| Colour | Score | Label |
|---|---|---|
| Deep blue | ≥ 65 | Excellent |
| Mid blue | 40–64 | Good |
| Light blue | 20–39 | Fair |
| Light grey | < 20 | Poor |
| Faded grey | — | No sensor |

**How it works**
Data: MeteoSwiss OGD (2009–2025)
Model: XGBoost, 53K winter station-days
Target: next-day score from depth + temp + freshness − rain
""")


# ── load + filter ─────────────────────────────────────────────────────────────
df_all = load_predictions()

df_map = df_all[
    (df_all["altitude"] >= min_alt) &
    df_all["lat"].notna() &
    df_all["lon"].notna()
].copy()

if not show_no_sensor:
    df_map = df_map[df_map["has_sensor"]]

df_map["color"]  = df_map["snow_quality"].apply(score_to_color)
df_map["radius"] = (df_map["altitude"] / 3600 * 6000 + 3000).clip(lower=3000).astype(int)

df_sensor = df_map[df_map["has_sensor"]]

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations (map)", len(df_map),
          delta=f"{len(df_sensor)} with sensor")
c2.metric("Avg quality (sensor stations)",
          f"{df_sensor['snow_quality'].mean():.1f} / 100" if len(df_sensor) else "–")
c3.metric("Good ski days (prob ≥ 60%)",
          int((df_sensor["ski_day_prob"] >= 0.6).sum()) if len(df_sensor) else 0)
best = (df_sensor.loc[df_sensor["snow_quality"].idxmax(), "station_id"]
        if len(df_sensor) else "–")
c4.metric("Best station", best)

st.divider()

# ── map + detail panel ────────────────────────────────────────────────────────
col_map, col_detail = st.columns([2, 1])

with col_map:
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=5,
        radius_max_pixels=22,
        pickable=True,
        stroked=True,
        get_line_color=[50, 50, 90, 120],
        line_width_min_pixels=1,
    )

    layers = [scatter_layer]

    if show_labels:
        layers.append(pdk.Layer(
            "TextLayer",
            data=df_map,
            get_position=["lon", "lat"],
            get_text="station_id",
            get_size=11,
            get_color=[30, 30, 70],
            get_pixel_offset=[0, -14],
        ))

    tooltip = {
        "html": (
            "<b>{station_id} — {name}</b><br/>"
            "<span style='color:#aac4ff'>Alt:</span> {altitude} m<br/>"
            "<span style='color:#aac4ff'>Snow depth:</span> {htoautd0} cm<br/>"
            "<span style='color:#aac4ff'>Temp:</span> {tre200d0} °C<br/>"
            "<span style='color:#aac4ff'>Quality score:</span> <b>{snow_quality}</b><br/>"
            "<span style='color:#aac4ff'>Ski-day prob:</span> {ski_day_prob}<br/>"
            "<span style='color:#aac4ff'>Rating:</span> <b>{quality_label}</b>"
        ),
        "style": {
            "backgroundColor": "rgba(15,20,50,0.92)",
            "color": "white",
            "fontSize": "13px",
            "padding": "10px 12px",
            "borderRadius": "8px",
            "lineHeight": "1.6",
        },
    }

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=46.8, longitude=8.3, zoom=7.2, pitch=0
        ),
        tooltip=tooltip,
        map_style=pdk.map_styles.CARTO_LIGHT,
    )

    st.pydeck_chart(deck, use_container_width=True)

with col_detail:
    st.subheader("Station detail")

    # Prefer sensor stations in the dropdown; sort by quality desc
    sensor_ids = (df_sensor
                  .sort_values("snow_quality", ascending=False)["station_id"]
                  .tolist())
    no_sensor_ids = (df_map[~df_map["has_sensor"]]
                     .sort_values("altitude", ascending=False)["station_id"]
                     .tolist())
    all_ids = sensor_ids + no_sensor_ids

    selected_id = st.selectbox(
        "Select station",
        all_ids,
        format_func=lambda sid: (
            f"{sid} — "
            f"{df_map.loc[df_map['station_id']==sid, 'name'].values[0]}"
            if len(df_map.loc[df_map['station_id']==sid]) else sid
        ),
    )

    if selected_id and len(df_map[df_map["station_id"] == selected_id]):
        row = df_map[df_map["station_id"] == selected_id].iloc[0]

        if row["has_sensor"]:
            qual = row["snow_quality"]
            prob = row["ski_day_prob"]
            st.metric("Snow quality score",
                      f"{qual:.0f} / 100", delta=row["quality_label"])
            st.metric("Ski-day probability", f"{prob*100:.0f}%")
            st.progress(min(int(qual), 100),
                        text=f"{row['quality_label']} snow conditions")
        else:
            st.info("No snow depth sensor at this station.")

        snow = row.get("htoautd0")
        temp = row.get("tre200d0")
        st.metric("Snow depth",
                  f"{snow:.0f} cm" if pd.notna(snow) else "No sensor")
        st.metric("Temperature",
                  f"{temp:.1f} °C" if pd.notna(temp) else "N/A")
        st.metric("Altitude", f"{row['altitude']:.0f} m")

        dsf = row.get("days_since_snowfall")
        if pd.notna(dsf):
            st.metric("Days since fresh snow", int(dsf))

        st.caption(f"Last observation: {row['date'].date()}")

st.divider()

# ── ranked table (sensor stations only) ───────────────────────────────────────
st.subheader("Ranked stations — snow sensor only")
display_cols = ["station_id", "name", "altitude", "htoautd0",
                "tre200d0", "days_since_snowfall", "snow_quality",
                "ski_day_prob", "quality_label"]
table = (df_sensor[[c for c in display_cols if c in df_sensor.columns]]
         .sort_values("snow_quality", ascending=False)
         .reset_index(drop=True))
table.index += 1
table.columns = [
    c.replace("htoautd0", "snow_cm")
     .replace("tre200d0", "temp_c")
     .replace("ski_day_prob", "ski_prob")
     .replace("snow_quality", "quality")
     .replace("days_since_snowfall", "days_since_snow")
    for c in table.columns
]
st.dataframe(
    table.style.background_gradient(subset=["quality"], cmap="Blues"),
    use_container_width=True,
    height=340,
)
