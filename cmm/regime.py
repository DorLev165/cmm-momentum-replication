"""
Market regime classification for the CMM backtest.

Provides three orthogonal regime dimensions:

1. bull_bear       — trailing 12-month S&P 500 total return sign. "bull" if
                     positive, "bear" if negative. Wikipedia-style, widely
                     used in factor research.
2. recession       — NBER business-cycle contractions vs expansions.
                     Hardcoded from FRED.
3. momentum_crash  — dates literature identifies as classic "momentum
                     crashes" (Daniel & Moskowitz 2016; plus well-known
                     modern episodes). Binary: in-crash vs not.

All three coexist — a month can simultaneously be "bear" and "recession"
and "crash". We report per-regime stats independently.

The paper's headline claim is crash resistance, so the momentum_crash
breakdown is the most diagnostic: CMM should have a *smaller* drawdown
inside crash months than plain momentum.
"""

from __future__ import annotations

import threading
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# On-disk cache for SPX monthly returns. yfinance occasionally hangs for
# 10+ minutes on a single ticker request (Cloudflare rate limit or session
# token expiry — observed in production runs). Caching to CSV makes
# reruns bullet-proof: read from disk, skip the network entirely.
_SPX_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "spx"
_SPX_CACHE_FILE = _SPX_CACHE_DIR / "spx_monthly.csv"


# NBER U.S. recessions (peak month → trough month, both inclusive).
# Source: https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
_NBER_RECESSIONS: list[tuple[str, str]] = [
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

# Known momentum-crash windows (Daniel & Moskowitz 2016 + post-publication
# episodes). These are the months where plain WML (winners-minus-losers)
# recorded the deepest monthly drawdowns in U.S. equities.
_MOMENTUM_CRASHES: list[tuple[str, str]] = [
    ("2001-01", "2001-06"),   # Dot-com unwind; tech momentum reversal
    ("2009-03", "2009-05"),   # Classic post-GFC rebound crash (-60%+ in 3 months)
    ("2019-09", "2019-10"),   # Sept 2019 factor unwind
    ("2020-11", "2021-02"),   # Vaccine-rally value rotation
    ("2023-01", "2023-03"),   # 2023 factor rotation
]


def _yf_download_with_timeout(ticker: str, start: str, end: str, timeout_sec: int = 30):
    """
    yfinance.download in a thread with a hard timeout. yfinance has no
    built-in timeout and has been observed to hang indefinitely. We spawn
    a daemon thread and bail out if it doesn't finish in time.
    """
    import yfinance as yf

    result: dict = {"data": None, "err": None}

    def _run():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result["data"] = yf.download(
                    ticker, start=start, end=end,
                    auto_adjust=True, progress=False,
                )
        except Exception as e:
            result["err"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        raise TimeoutError(f"yfinance hung fetching {ticker} for >{timeout_sec}s")
    if result["err"] is not None:
        raise result["err"]
    return result["data"]


def fetch_market_returns(
    start: str = "1972-01-01",
    end: Optional[str] = None,
    timeout_sec: int = 30,
    use_cache: bool = True,
) -> pd.Series:
    """
    Monthly S&P 500 total return series (log return).

    Load order:
      1. On-disk cache at data/spx/spx_monthly.csv (if `use_cache` and exists)
      2. yfinance with a hard timeout (threaded, since yfinance itself has
         no timeout parameter)

    On successful network fetch, writes to the cache so subsequent runs
    are instantaneous.

    If both cache and network fail, raises — caller should catch and
    degrade gracefully (e.g. fall back to zero regime signal).
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    # (1) On-disk cache
    if use_cache and _SPX_CACHE_FILE.exists() and _SPX_CACHE_FILE.stat().st_size > 0:
        try:
            df = pd.read_csv(_SPX_CACHE_FILE, parse_dates=["date"], index_col="date")
            s = df["sp500_logret"].astype(float)
            s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
            if len(s) > 0:
                return s
        except Exception as e:
            print(f"  [spx-cache] read failed, falling back to network: {e}")

    # (2) yfinance with timeout
    data = _yf_download_with_timeout("^GSPC", start, end, timeout_sec=timeout_sec)
    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned no S&P 500 data")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    close = close.dropna()

    monthly = close.resample("ME").last()
    logret = np.log(monthly / monthly.shift(1)).dropna()
    logret.name = "sp500_logret"

    # Write cache
    try:
        _SPX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logret.to_frame().reset_index().rename(columns={"index": "date", "Date": "date"}).to_csv(
            _SPX_CACHE_FILE, index=False
        )
    except Exception as e:
        print(f"  [spx-cache] write failed: {e}")

    return logret


def trailing_spx_logreturn(
    dates: pd.DatetimeIndex,
    market_returns: pd.Series,
    lookback_months: int = 12,
) -> pd.Series:
    """
    Trailing `lookback_months` cumulative log-return of the S&P 500,
    aligned to each date in `dates`. Value at date d = sum of monthly
    log returns from (d-lookback+1) to d inclusive (month-end dates).

    Returns a Series indexed by `dates`. Missing values (insufficient
    history) are filled with 0 (the cross-sectional median under the
    downstream rolling-rank transform).
    """
    r = market_returns.copy().sort_index()
    trailing = r.rolling(lookback_months, min_periods=lookback_months).sum()
    out = pd.Series(np.nan, index=pd.DatetimeIndex(dates))
    trailing_idx = trailing.dropna().index
    if len(trailing_idx) == 0:
        return out.fillna(0.0)
    for d in out.index:
        pos = trailing_idx.searchsorted(d, side="right") - 1
        if pos >= 0:
            out.loc[d] = float(trailing.loc[trailing_idx[pos]])
    return out


def spx_trend_feature(
    dates: pd.DatetimeIndex,
    lookback_months: int = 12,
    rank_window: int = 60,
) -> pd.Series:
    """
    Cleaned-up SPX trend regime signal, suitable as an FFN input feature
    or interaction multiplier. Prefers a CSV cache; falls back to yfinance
    with a timeout.

    Raises `RuntimeError` if market data cannot be fetched — NO silent
    zero-fallback. Per data-integrity policy, the caller should fix the
    fetch rather than run a degraded pipeline. The error message tells
    the user exactly how to fix it.

    Returns a Series indexed by `dates`.
    """
    req_dates = pd.DatetimeIndex(dates)
    # CRITICAL: compute the rolling-rank transform on UNIQUE month-ends
    # (~600 values), not on the per-row array (~3M values). Running
    # rolling(60).apply(python_func) over 3M rows takes many hours in
    # Python; on 600 rows it's milliseconds. We then broadcast back to
    # per-row via reindex.
    unique_dates = pd.DatetimeIndex(sorted(req_dates.unique()))
    req_min, req_max = unique_dates.min(), unique_dates.max()
    start = (req_min - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
    end = (req_max + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
    try:
        mkt = fetch_market_returns(start=start, end=end, timeout_sec=30)
    except Exception as e:
        raise RuntimeError(
            f"\n\n"
            f"[BLOCKER] cannot fetch market return data for regime signal.\n"
            f"    Underlying error: {type(e).__name__}: {e}\n\n"
            f"    Fix: populate the SPX cache from CRSP (via WRDS):\n"
            f"        python scripts/cache_spx_from_wrds.py\n\n"
            f"    This writes data/spx/spx_monthly.csv which the pipeline\n"
            f"    reads in <1 second, bypassing yfinance entirely.\n"
            f"    Refusing to fall back to zeros (would silently degrade the\n"
            f"    regime-ensemble and D-interaction features to no-ops).\n"
        ) from e

    # Coverage check — refuse rather than silently use stale values.
    if len(mkt) == 0:
        raise RuntimeError("[BLOCKER] market return series is empty.")
    spx_max = pd.Timestamp(mkt.index.max())
    if spx_max + pd.Timedelta(days=10) < req_max:
        raise RuntimeError(
            f"\n\n"
            f"[BLOCKER] SPX coverage ends {spx_max.date()} but JKP panel "
            f"extends to {req_max.date()}.\n"
            f"    Those {((req_max - spx_max).days // 30)} extra months have no "
            f"market-return data.\n\n"
            f"    Options:\n"
            f"    1. Refresh the SPX cache (CRSP msi may have updated):\n"
            f"           python scripts/cache_spx_from_wrds.py\n"
            f"    2. Truncate the JKP panel to <= {spx_max.date()} via\n"
            f"       end_date=... in fetch_jkp_cmm_data (edit main.py).\n"
        )

    # Compute trailing return + rank on unique month-ends (fast).
    raw = trailing_spx_logreturn(unique_dates, mkt, lookback_months=lookback_months)

    def _rank_last(x: np.ndarray) -> float:
        finite = x[np.isfinite(x)]
        if len(finite) < 2 or not np.isfinite(x[-1]):
            return np.nan
        rank = (finite <= x[-1]).sum() / len(finite)
        return float(rank - 0.5)

    ranked_unique = raw.rolling(rank_window, min_periods=12).apply(_rank_last, raw=True)
    ranked_unique = ranked_unique.fillna(0.0)

    # Broadcast back to per-row (req_dates may have duplicates; reindex
    # handles that by looking up each duplicate separately).
    result = ranked_unique.reindex(req_dates).fillna(0.0)
    result.name = None
    return result


def classify_bull_bear(
    dates: pd.DatetimeIndex,
    market_returns: pd.Series,
    lookback_months: int = 12,
) -> pd.Series:
    """
    Bull = trailing 12-month S&P 500 total return > 0, Bear otherwise.

    Computed from `market_returns` (monthly log returns). For dates outside
    the market data range, returns 'unknown'.
    """
    r = market_returns.copy().sort_index()
    trailing = r.rolling(lookback_months, min_periods=lookback_months).sum()
    labels = pd.Series("unknown", index=pd.DatetimeIndex(dates))

    # For each target date, find the most recent trailing value ≤ that date
    trailing_idx = trailing.dropna().index
    if len(trailing_idx) == 0:
        return labels

    for d in labels.index:
        pos = trailing_idx.searchsorted(d, side="right") - 1
        if pos >= 0:
            val = trailing.loc[trailing_idx[pos]]
            labels.loc[d] = "bull" if val > 0 else "bear"

    return labels


def classify_recession(dates: pd.DatetimeIndex) -> pd.Series:
    """NBER recession classification: 'recession' or 'expansion' per month."""
    labels = pd.Series("expansion", index=pd.DatetimeIndex(dates))
    for start, end in _NBER_RECESSIONS:
        s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        in_rec = (labels.index >= s) & (labels.index <= e)
        labels.loc[in_rec] = "recession"
    return labels


def classify_momentum_crash(dates: pd.DatetimeIndex) -> pd.Series:
    """'crash' if the month falls inside a known momentum-crash window, else 'normal'."""
    labels = pd.Series("normal", index=pd.DatetimeIndex(dates))
    for start, end in _MOMENTUM_CRASHES:
        s, e = pd.Timestamp(start), pd.Timestamp(end) + pd.offsets.MonthEnd(0)
        in_crash = (labels.index >= s) & (labels.index <= e)
        labels.loc[in_crash] = "crash"
    return labels


def summarize_by_regime(
    hml_returns: pd.Series,
    regime: pd.Series,
    label: str = "regime",
) -> pd.DataFrame:
    """
    Per-regime summary: n_months, mean, annualized return, annualized vol,
    Sharpe, MDD, hit rate.

    `hml_returns` and `regime` must be indexed by month-end dates.
    Returns a DataFrame with one row per regime value.
    """
    r = hml_returns.dropna()
    reg = regime.reindex(r.index)

    rows = []
    for name in reg.dropna().unique():
        mask = reg == name
        x = r[mask]
        if len(x) == 0:
            continue
        equity = (1 + x).cumprod()
        dd = equity / equity.cummax() - 1
        rows.append({
            label: name,
            "n_months": int(len(x)),
            "mean_pct_mo": float(x.mean() * 100),
            "ann_ret_pct": float(x.mean() * 12 * 100),
            "ann_vol_pct": float(x.std() * np.sqrt(12) * 100),
            "sharpe": float(x.mean() / (x.std() + 1e-12) * np.sqrt(12)),
            "mdd_pct": float(dd.min() * 100),
            "hit_rate_pct": float((x > 0).mean() * 100),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(label).reset_index(drop=True)
    return df


def regime_report(hml_returns: pd.Series) -> dict:
    """
    Full regime breakdown — fetches market returns, classifies all three
    regime dimensions, and returns a dict of summary DataFrames.

    Parameters
    ----------
    hml_returns : pd.Series
        Monthly HML return series, indexed by month-end date.

    Returns
    -------
    dict with keys: 'bull_bear', 'recession', 'momentum_crash' (each is a
    DataFrame with per-regime stats).
    """
    r = hml_returns.dropna()
    if len(r) == 0:
        return {}

    start = (r.index.min() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end = (r.index.max() + pd.DateOffset(months=1)).strftime("%Y-%m-%d")

    try:
        mkt = fetch_market_returns(start=start, end=end)
    except Exception as e:
        print(f"  [regime] market-return fetch failed: {e}")
        mkt = None

    out = {}
    if mkt is not None:
        bb = classify_bull_bear(r.index, mkt, lookback_months=12)
        out["bull_bear"] = summarize_by_regime(r, bb, label="regime")

    rec = classify_recession(r.index)
    out["recession"] = summarize_by_regime(r, rec, label="regime")

    crash = classify_momentum_crash(r.index)
    out["momentum_crash"] = summarize_by_regime(r, crash, label="regime")

    return out
