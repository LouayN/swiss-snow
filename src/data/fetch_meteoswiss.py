"""
MeteoSwiss OGD (Open Government Data) fetcher.

Pulls station metadata and daily historical + recent CSV data from:
  https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/

No API key required. Data is CC-BY licensed.
"""

import io
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn"
STAC_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn"

# Daily parameters we care about for snow quality modelling
SNOW_PARAMS = [
    "htoautd0",  # snow depth (cm)
    "tre200d0",  # air temp 2m daily mean (°C)
    "tre200dx",  # air temp 2m daily max (°C)
    "tre200dn",  # air temp 2m daily min (°C)
    "rre150d0",  # precipitation daily total (mm)
    "rka150d0",  # precipitation 0-0 UTC (mm)
    "ure200d0",  # relative humidity daily mean (%)
    "fkl010d0",  # wind speed scalar daily mean (m/s)
    "fkl010d1",  # gust peak daily max (m/s)
    "sre000d0",  # sunshine duration daily total (min)
    "gre000d0",  # global radiation daily mean (W/m²)
]

# Minimum altitude (m) for snow-relevant stations
MIN_ALTITUDE = 800


def _make_session(retries: int = 5, backoff: float = 1.0) -> requests.Session:
    """Session with retry logic and exponential backoff."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "swiss-snow-project/0.1"})
    return session


SESSION = _make_session()


def fetch_stations_meta(cache_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Download the MeteoSwiss station metadata CSV.
    Returns a DataFrame with station_id, name, altitude, lat, lon (WGS84).
    """
    url = f"{BASE_URL}/ogd-smn_meta_stations.csv"
    if cache_path and cache_path.exists():
        log.info("Loading stations from cache: %s", cache_path)
        return pd.read_csv(cache_path, sep=";", encoding="utf-8-sig")

    log.info("Fetching station metadata from %s", url)
    resp = SESSION.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(
        io.BytesIO(resp.content),
        sep=";",
        encoding="latin-1",
    )
    log.info("Loaded %d stations", len(df))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, sep=";", index=False)

    return df



def parse_stations(df_raw: pd.DataFrame, min_altitude: int = MIN_ALTITUDE) -> pd.DataFrame:
    """
    Clean and filter the raw station metadata.
    The OGD CSV already contains WGS84 lat/lon columns.
    Filters by minimum altitude.
    """
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Actual column names from the OGD stations CSV:
    #   station_abbr, station_name, station_height_masl,
    #   station_coordinates_wgs84_lat, station_coordinates_wgs84_lon, ...
    rename = {
        "station_abbr": "station_id",
        "station_name": "name",
        "station_height_masl": "altitude",
        "station_coordinates_wgs84_lat": "lat",
        "station_coordinates_wgs84_lon": "lon",
        "station_canton": "canton",
        "station_type_en": "station_type",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["altitude"] = pd.to_numeric(df.get("altitude"), errors="coerce")
    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")

    df = df[df["altitude"] >= min_altitude].copy()
    log.info("Filtered to %d stations with altitude >= %dm", len(df), min_altitude)

    df["station_id"] = df["station_id"].str.strip().str.upper()
    return df.reset_index(drop=True)


def fetch_station_daily(
    station_id: str,
    period: str = "recent",
    cache_dir: Optional[Path] = None,
    throttle: float = 0.3,
) -> pd.DataFrame:
    """
    Download daily CSV for one station.

    Parameters
    ----------
    station_id : str
        3-letter station code (e.g. "ABO").
    period : str
        "recent" (last ~2 years) or "historical" (all decades).
    cache_dir : Path, optional
        Directory to cache raw CSV files.
    throttle : float
        Seconds to sleep after each request (be polite).

    Returns
    -------
    pd.DataFrame with parsed dates and float columns.
    """
    sid = station_id.lower()
    fname = f"ogd-smn_{sid}_d_{period}.csv"
    url = f"{BASE_URL}/{sid}/{fname}"

    # Check cache
    if cache_dir:
        cache_file = cache_dir / fname
        if cache_file.exists():
            log.debug("Cache hit: %s", cache_file)
            return _parse_daily_csv(cache_file.read_bytes())

    log.info("Fetching %s", url)
    resp = SESSION.get(url, timeout=60)
    if resp.status_code == 404:
        log.warning("Station %s not found for period '%s'", station_id, period)
        return pd.DataFrame()
    resp.raise_for_status()

    time.sleep(throttle)

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / fname).write_bytes(resp.content)

    return _parse_daily_csv(resp.content)


def _parse_daily_csv(raw: bytes) -> pd.DataFrame:
    """Parse the semicolon-delimited MeteoSwiss daily CSV."""
    df = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        encoding="latin-1",
        na_values=["-", ""],
    )
    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Date column is called 'reference_timestamp' or 'time' or 'date'
    date_col = next(
        (c for c in df.columns if "time" in c or "date" in c or "reference" in c),
        None,
    )
    if date_col:
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date").reset_index(drop=True)

    return df


def fetch_all_snow_stations(
    min_altitude: int = MIN_ALTITUDE,
    period: str = "recent",
    cache_dir: Optional[Path] = None,
    max_stations: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch and concatenate daily data for all stations above min_altitude.

    Returns a long-format DataFrame with a 'station_id' column added.
    """
    stations_meta = fetch_stations_meta(
        cache_path=Path(cache_dir / "ogd-smn_meta_stations.csv") if cache_dir else None
    )
    stations = parse_stations(stations_meta, min_altitude=min_altitude)

    if max_stations:
        stations = stations.head(max_stations)

    log.info("Fetching daily data for %d stations (period=%s)", len(stations), period)

    frames = []
    for _, row in stations.iterrows():
        sid = row["station_id"]
        df = fetch_station_daily(sid, period=period, cache_dir=cache_dir)
        if df.empty:
            continue
        df["station_id"] = sid
        # Attach metadata
        for col in ["altitude", "lat", "lon", "name"]:
            if col in row.index:
                df[col] = row[col]
        frames.append(df)

    if not frames:
        log.warning("No data fetched!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Combined dataset: %d rows, %d stations, date range %s – %s",
        len(combined),
        combined["station_id"].nunique(),
        combined["date"].min().date() if "date" in combined.columns else "?",
        combined["date"].max().date() if "date" in combined.columns else "?",
    )
    return combined


if __name__ == "__main__":
    import sys

    data_dir = Path("data/raw/meteoswiss")
    period = sys.argv[1] if len(sys.argv) > 1 else "recent"

    df = fetch_all_snow_stations(
        min_altitude=MIN_ALTITUDE,
        period=period,
        cache_dir=data_dir,
        max_stations=None,
    )

    out = Path("data/raw") / f"meteoswiss_daily_{period}.parquet"
    df.to_parquet(out, index=False)
    log.info("Saved to %s  (%d rows)", out, len(df))
