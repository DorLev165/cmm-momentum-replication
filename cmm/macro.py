"""
Macro feature loader for CMM.

Pulls monthly macroeconomic series from FRED, derives standard transforms
(YoY inflation, term spread, etc.), rank-transforms each series over a
rolling 60-month window to match JKP's [-0.5, 0.5] convention, and lags
every feature by 1 month to respect release timing (CPI for month t is
published mid-t+1, so the lagged value is what would have been public
at month-end t).

Usage
-----
df = fetch_macro_features("1970-01-01", "2026-04-30")
# df is a month-end-indexed DataFrame, columns = macro feature names.

Design
------
- Data source: FRED direct CSV download (no API key, no new dependency).
- Rank transform applied AFTER raw data is loaded; lag applied AFTER
  transform. Both operations are backward-looking, so no look-ahead.
- Missing values before a series begins (e.g. VIX pre-1990) are left as
  NaN and filled with 0 (= rolling-rank median) at merge time.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import urllib.request
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Where scripts/cache_fred.bat puts the downloaded CSVs. Relative to project
# root — macro.py lives in cmm/ so parent() gets us to the repo root.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "fred"


# FRED series we pull. Each value is a human-readable column name.
# All are published monthly (INDPRO, CPI, UNRATE, FEDFUNDS) or daily and
# resampled to month-end (DGS10, DGS3MO, BAA, VIXCLS).
_FRED_SERIES: dict[str, str] = {
    "CPIAUCSL":  "cpi",           # Consumer Price Index, all items, NSA
    "UNRATE":    "unemp",         # Civilian Unemployment Rate, SA
    "FEDFUNDS":  "fedfunds",      # Federal Funds Effective Rate
    "DGS10":     "gs10",          # 10-Year Treasury (constant maturity)
    "DGS3MO":    "gs3m",          # 3-Month Treasury
    "BAA":       "baa",           # Moody's BAA Corporate Bond Yield
    "INDPRO":    "indpro",        # Industrial Production Index
    "VIXCLS":    "vix",           # CBOE VIX (starts 1990-01)
}


def _http_get(url: str, timeout: int = 30) -> str:
    """
    Fetch a URL. Prefers `curl` via subprocess because FRED is behind
    Cloudflare and Python's urllib/requests stacks frequently hang on its
    HTTP/2 responses (observed on Windows 10+ with Python 3.10). Curl is
    pre-installed on Windows 10+ and macOS/Linux, so this works out of
    the box for nearly all users. Falls back to urllib if curl isn't found.
    """
    curl_bin = shutil.which("curl")
    if curl_bin:
        result = subprocess.run(
            [
                curl_bin, "-sSL", "--max-time", str(timeout),
                "-A", "Mozilla/5.0",
                url,
            ],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed (rc={result.returncode}): {result.stderr[:200]}")
        return result.stdout
    # Fallback — may hang on some Windows Python builds
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def _parse_fred_csv(text: str, series_id: str) -> pd.Series:
    """Parse a FRED CSV (from the graph endpoint) into a typed Series."""
    df = pd.read_csv(io.StringIO(text))
    date_col = next(c for c in df.columns if c.lower() in ("date", "observation_date"))
    val_col = next(c for c in df.columns if c != date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col]
    s.name = series_id
    return s


def _parse_fred_json(text: str, series_id: str) -> pd.Series:
    """Parse a FRED API JSON response into a typed Series."""
    data = json.loads(text)
    obs = data.get("observations", [])
    if not obs:
        return pd.Series(dtype=float, name=series_id)
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    s = df.set_index("date")["value"]
    s.name = series_id
    return s


def _fetch_fred_series(
    series_id: str,
    start: str,
    end: str,
    cache_dir: Optional[Path] = None,
) -> pd.Series:
    """
    Load a FRED series, preferring on-disk cache populated by
    `scripts/cache_fred.bat`. Cache supports both JSON (from the API
    endpoint, preferred) and CSV (from the graph endpoint, legacy).

    Fallbacks, in order:
      1. Cached JSON at <cache_dir>/<series_id>.json
      2. Cached CSV  at <cache_dir>/<series_id>.csv
      3. Live HTTP — API endpoint if FRED_API_KEY env var is set
      4. Live HTTP — graph CSV endpoint (known to be flaky)
    """
    cache_dir = cache_dir or _DEFAULT_CACHE_DIR
    json_path = cache_dir / f"{series_id}.json"
    csv_path = cache_dir / f"{series_id}.csv"

    # (1) Cached JSON
    if json_path.exists() and json_path.stat().st_size > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            s = _parse_fred_json(f.read(), series_id)
    # (2) Cached CSV
    elif csv_path.exists() and csv_path.stat().st_size > 0:
        with open(csv_path, "r", encoding="utf-8") as f:
            s = _parse_fred_csv(f.read(), series_id)
    # (3) Live API (needs key)
    elif os.environ.get("FRED_API_KEY"):
        key = os.environ["FRED_API_KEY"]
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={key}&file_type=json"
            f"&observation_start={start}"
        )
        s = _parse_fred_json(_http_get(url, timeout=30), series_id)
    # (4) Live graph CSV (flaky)
    else:
        url = (
            f"https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start}&coed={end}"
        )
        s = _parse_fred_csv(_http_get(url, timeout=30), series_id)

    # Trim to requested window
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return s


def _load_raw_panel(start: str, end: str) -> pd.DataFrame:
    """Fetch all raw FRED series and align to month-end index."""
    raw = {}
    for sid, name in _FRED_SERIES.items():
        try:
            s = _fetch_fred_series(sid, start, end)
            # Resample daily/irregular to month-end (last value of each month).
            # For monthly series this is a no-op.
            s_me = s.resample("ME").last()
            raw[name] = s_me
        except Exception as e:
            warnings.warn(f"FRED fetch failed for {sid}: {e}", stacklevel=2)
            raw[name] = pd.Series(dtype=float)
    df = pd.DataFrame(raw)
    df.index = pd.DatetimeIndex(df.index)
    return df


def _compute_transforms(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the feature set from raw FRED series:
    - cpi_yoy:     12-month log change in CPI (year-over-year inflation)
    - indpro_yoy:  12-month log change in industrial production
    - term_spread: 10y − 3m Treasury
    - credit_spread: BAA − 10y Treasury
    Plus levels: unemp, fedfunds, vix.
    """
    out = pd.DataFrame(index=raw.index)

    if "cpi" in raw:
        out["cpi_yoy"] = np.log(raw["cpi"]) - np.log(raw["cpi"].shift(12))
    if "indpro" in raw:
        out["indpro_yoy"] = np.log(raw["indpro"]) - np.log(raw["indpro"].shift(12))
    if "gs10" in raw and "gs3m" in raw:
        out["term_spread"] = raw["gs10"] - raw["gs3m"]
    if "baa" in raw and "gs10" in raw:
        out["credit_spread"] = raw["baa"] - raw["gs10"]
    if "unemp" in raw:
        out["unemp"] = raw["unemp"]
    if "fedfunds" in raw:
        out["fedfunds"] = raw["fedfunds"]
    if "vix" in raw:
        out["vix"] = raw["vix"]

    return out


def _rolling_rank_transform(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling historical rank → [-0.5, 0.5], matching JKP's rank convention
    but in the time dimension (since macro values are same across stocks
    in a given month). Uses only backward-looking data (no look-ahead).

    At month t, for each feature:
        rank = (rank of value_t within [value_{t-window+1}, ..., value_t]) / n
        out  = rank - 0.5
    Output range: [-0.5, 0.5 - 1/n].
    """
    def _rank_last(x: np.ndarray) -> float:
        # Ignore NaNs in the window; rank only finite values
        finite = x[np.isfinite(x)]
        if len(finite) < 2 or not np.isfinite(x[-1]):
            return np.nan
        # Fraction of values ≤ current value (in [0, 1])
        rank = (finite <= x[-1]).sum() / len(finite)
        return float(rank - 0.5)

    out = df.copy()
    for col in out.columns:
        out[col] = out[col].rolling(window, min_periods=12).apply(_rank_last, raw=True)
    return out


def _lag(df: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """Shift forward by `months` so month-end t holds data known as of t-months."""
    return df.shift(months)


def fetch_macro_features(
    start: str = "1970-01-01",
    end: Optional[str] = None,
    rank_window: int = 60,
    publication_lag_months: int = 1,
) -> pd.DataFrame:
    """
    End-to-end macro feature loader. Returns a month-end-indexed DataFrame
    with columns in [-0.5, 0.5] (NaN where history is too short).

    The output is safe to merge directly onto the CMM monthly panel by
    matching on month-end date.
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    raw = _load_raw_panel(start, end)
    feats = _compute_transforms(raw)
    ranked = _rolling_rank_transform(feats, window=rank_window)
    lagged = _lag(ranked, months=publication_lag_months)
    return lagged


MACRO_FEATURE_NAMES: list[str] = [
    "cpi_yoy", "indpro_yoy", "term_spread", "credit_spread",
    "unemp", "fedfunds", "vix",
]
N_MACRO = len(MACRO_FEATURE_NAMES)
