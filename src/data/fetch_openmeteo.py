"""
Open-Meteo API fetcher.

Free, no API key required.
Docs: https://open-meteo.com/en/docs

Used for:
  - Historical weather reanalysis (ERA5) to fill gaps in station data
  - Current + 7-day forecast for inference inputs
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"


def _make_session(retries: int = 5, backoff: float = 1.5) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "swiss-snow-project/0.1"})
    return session


SESSION = _make_session()

# Daily variables available from Open-Meteo
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "snowfall_sum",           # cm of snowfall water equivalent
    "snow_depth_max",         # m snow depth at end of day
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "precipitation_hours",    # hours with precipitation (proxy for intensity)
    "sunshine_duration",      # seconds
    "weather_code",           # WMO weather code
]


def fetch_forecast(
    lat: float,
    lon: float,
    days: int = 7,
    daily_vars: Optional[list[str]] = None,
    throttle: float = 0.2,
) -> pd.DataFrame:
    """
    Fetch current conditions + N-day forecast from Open-Meteo.

    Parameters
    ----------
    lat, lon : float
        Station WGS84 coordinates.
    days : int
        Forecast horizon (1-16).
    daily_vars : list[str], optional
        Variables to request (defaults to DAILY_VARS).

    Returns
    -------
    pd.DataFrame indexed by date.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily_vars or DAILY_VARS,
        "forecast_days": days,
        "timezone": "Europe/Zurich",
    }
    log.debug("Forecast request lat=%.4f lon=%.4f days=%d", lat, lon, days)
    resp = SESSION.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(throttle)
    return _parse_daily_response(resp.json())


def fetch_historical(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    daily_vars: Optional[list[str]] = None,
    throttle: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch ERA5 reanalysis historical data from Open-Meteo Archive API.

    Parameters
    ----------
    lat, lon : float
        Coordinates.
    start_date, end_date : str
        ISO dates e.g. "2020-01-01".
    daily_vars : list[str], optional
        Variables to request.

    Returns
    -------
    pd.DataFrame indexed by date.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": daily_vars or DAILY_VARS,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Europe/Zurich",
    }
    log.debug(
        "Historical request lat=%.4f lon=%.4f %s→%s", lat, lon, start_date, end_date
    )
    resp = SESSION.get(HISTORICAL_URL, params=params, timeout=60)
    resp.raise_for_status()
    time.sleep(throttle)
    return _parse_daily_response(resp.json())


def _parse_daily_response(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a tidy DataFrame."""
    daily = data.get("daily", {})
    if not daily or "time" not in daily:
        return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("date").reset_index()
    return df


def fetch_historical_for_stations(
    stations: pd.DataFrame,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    throttle: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch historical Open-Meteo data for multiple stations.

    Parameters
    ----------
    stations : pd.DataFrame
        Must have columns: station_id, lat, lon.
    start_date : str
        ISO date string.
    end_date : str, optional
        Defaults to today.

    Returns
    -------
    Long-format DataFrame with station_id column.
    """
    import datetime

    if end_date is None:
        end_date = datetime.date.today().isoformat()

    frames = []
    for _, row in stations.iterrows():
        sid = row["station_id"]
        try:
            df = fetch_historical(
                lat=row["lat"],
                lon=row["lon"],
                start_date=start_date,
                end_date=end_date,
                throttle=throttle,
            )
            if df.empty:
                continue
            df["station_id"] = sid
            frames.append(df)
            log.info("Fetched Open-Meteo historical for %s (%d rows)", sid, len(df))
        except Exception as exc:
            log.warning("Failed for station %s: %s", sid, exc)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def fetch_forecast_for_stations(
    stations: pd.DataFrame,
    days: int = 7,
) -> pd.DataFrame:
    """
    Fetch 7-day forecast for all stations. Used at inference time.
    """
    frames = []
    for _, row in stations.iterrows():
        sid = row["station_id"]
        try:
            df = fetch_forecast(lat=row["lat"], lon=row["lon"], days=days)
            if df.empty:
                continue
            df["station_id"] = sid
            frames.append(df)
        except Exception as exc:
            log.warning("Forecast failed for %s: %s", sid, exc)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    # Quick smoke test — Adelboden station (lat/lon)
    print("Testing forecast for Adelboden (46.49°N, 7.56°E)...")
    df = fetch_forecast(lat=46.49, lon=7.56, days=7)
    print(df.to_string())
