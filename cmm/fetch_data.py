"""
Fetch real stock data for S&P 500.

- S&P 500 constituents from Wikipedia
- Daily prices from yfinance
- Computes: price-based characteristics only (no fundamentals - avoids look-ahead bias in backtest)
"""

import urllib.request
import warnings
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from cmm.portfolio import get_nyse_tickers

# Data dimensions
N_DAILY_RETURNS = 231  # t-252 to t-22
TRADING_DAYS_PER_YEAR = 252
SKIP_DAYS = 22  # skip most recent month for short-term reversal

# Price-based characteristics only (11). Fundamentals commented out - yfinance provides
# current data only, which would pollute backtests with look-ahead bias.
N_PRICE_CHARS = 11
N_CHAR_DEFAULT = N_PRICE_CHARS

# --- FUNDAMENTALS COMMENTED OUT (look-ahead bias in backtest) ---
# yfinance .info returns current/latest values, not point-in-time. Using them would
# apply today's ROE, D/E etc. to past predictions. For clean backtests, use price-only.
# FUNDAMENTAL_KEYS = ["returnOnEquity", "debtToEquity", ...]


def get_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode()
    tables = pd.read_html(StringIO(html))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def _fetch_shares_panel(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, pd.Series]:
    """
    Fetch historical shares outstanding per ticker using yfinance's
    get_shares_full. Returns dict ticker -> Series indexed by date.

    yfinance's shares data is sparse and limited to a handful of recent years
    for most tickers. Callers must forward-fill to align to price dates and
    fall back gracefully when a ticker has no history. See
    CMM_REPLICATION_ISSUES.md §3 and §7.
    """
    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            s = yf.Ticker(t).get_shares_full(start=start, end=end)
            if s is None or len(s) == 0:
                continue
            s = pd.Series(s).astype(float).sort_index()
            # Drop tz so it can be joined with tz-naive price index
            if getattr(s.index, "tz", None) is not None:
                s.index = s.index.tz_convert(None)
            s.index = pd.to_datetime(s.index).normalize()
            s = s[~s.index.duplicated(keep="last")]
            out[t] = s
        except Exception:
            continue
    return out


def _build_market_cap_frame(
    close: pd.DataFrame,
    shares_panel: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Align per-ticker shares-outstanding series to the daily price index and
    compute market cap = shares * price. Missing shares are forward-filled
    within a ticker; tickers with no shares data produce an NaN column that
    callers must handle (typically by falling back to price as a proxy).
    """
    idx = close.index
    mc = pd.DataFrame(index=idx, columns=close.columns, dtype=np.float64)
    for t in close.columns:
        if t not in shares_panel:
            continue
        s = shares_panel[t].reindex(idx, method="ffill")
        mc[t] = s * close[t]
    return mc


# def _fetch_fundamentals(tickers): ...
# def _build_char_vector(price_chars, fundamentals): ...


def _compute_chars_simple(px: np.ndarray, n_char: int) -> np.ndarray:
    """Compute price-based characteristics from price series (no look-ahead)."""
    if len(px) < 22:
        return np.zeros(n_char)
    ret = np.diff(np.log(px.astype(np.float64) + 1e-10))
    ret = ret[~np.isnan(ret)]
    if len(ret) < 21:
        return np.zeros(n_char)
    chars = [
        np.log(px[-1] + 1e-6),
        np.sum(ret[-21:]) if len(ret) >= 21 else 0,
        np.sum(ret[-63:]) if len(ret) >= 63 else 0,
        np.sum(ret[-126:]) if len(ret) >= 126 else 0,
        np.sum(ret[-min(252, len(ret)):]),
        np.nanstd(ret[-21:]) * np.sqrt(252) if len(ret) >= 21 else 0,
        np.nanstd(ret[-63:]) * np.sqrt(252) if len(ret) >= 63 else 0,
        np.nanstd(ret[-126:]) * np.sqrt(252) if len(ret) >= 126 else 0,
        float(pd.Series(ret[-63:]).skew()) if len(ret) >= 63 else 0,
        np.min(np.cumsum(ret[-126:])) if len(ret) >= 126 else 0,
        np.sum(ret[-5:]) if len(ret) >= 5 else 0,
    ]
    arr = np.zeros(n_char)
    arr[: min(n_char, len(chars))] = chars[:n_char]
    return arr


def _extract_close_prices(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract Close prices DataFrame from yfinance download result."""
    if isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker with group_by='ticker': columns (Ticker, Metric), level 1 = Close/Open/etc
        if "Close" in data.columns.get_level_values(1):
            close = data.xs("Close", axis=1, level=1)
        elif "Adj Close" in data.columns.get_level_values(1):
            close = data.xs("Adj Close", axis=1, level=1)
        else:
            raise ValueError("No Close column in download data")
    else:
        close = pd.DataFrame(data["Close"].values, index=data.index, columns=tickers)
    return close


def fetch_sp500_cmm_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_tickers: Optional[int] = 500,
    n_char: int = N_CHAR_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fetch and prepare CMM data for S&P 500 stocks.
    Uses price-based characteristics only (no fundamentals - avoids look-ahead bias).
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_dt = pd.Timestamp(end_date) - pd.DateOffset(years=6)
        start_date = start_dt.strftime("%Y-%m-%d")

    print("Fetching S&P 500 constituents...")
    tickers = get_sp500_tickers()
    if max_tickers:
        tickers = tickers[:max_tickers]
        print(f"  Using {len(tickers)} tickers (limited)")
    else:
        print(f"  Found {len(tickers)} tickers")

    print("Downloading price data (this may take 1-2 minutes)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=True,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
        )

    # Handle single-ticker download format
    if len(tickers) == 1:
        if "Close" in data.columns:
            close = pd.DataFrame(data["Close"].values, index=data.index, columns=tickers)
        else:
            close = pd.DataFrame(data["Adj Close"].values, index=data.index, columns=tickers)
    else:
        close = _extract_close_prices(data, tickers)
        if close is None or close.empty:
            # Fallback: try common column structures
            if isinstance(data.columns, pd.MultiIndex):
                close = data.xs("Close", axis=1, level=1)
            elif "Close" in data.columns:
                close = data[["Close"]].copy()
                close.columns = [tickers[0]]
            else:
                raise ValueError("Could not extract Close prices from download")

    # Ensure numeric index
    close.index = pd.to_datetime(close.index)
    close = close.dropna(how="all")
    print(f"  Downloaded {len(close)} trading days x {close.shape[1]} tickers")

    # Historical shares outstanding for proper market-cap weighting (§3).
    # yfinance's history is sparse, so we forward-fill per ticker and track
    # missing coverage for a clear warning.
    print("Fetching historical shares outstanding (per ticker)...")
    shares_panel = _fetch_shares_panel(
        list(close.columns),
        start=str(close.index.min().date()),
        end=str(close.index.max().date()),
    )
    market_cap = _build_market_cap_frame(close, shares_panel)
    n_with_shares = sum(1 for c in market_cap.columns if market_cap[c].notna().any())
    print(f"  Shares-outstanding coverage: {n_with_shares}/{close.shape[1]} tickers")
    if n_with_shares < close.shape[1]:
        warnings.warn(
            f"{close.shape[1] - n_with_shares} tickers have no shares data; "
            "falling back to price as market-cap proxy for those tickers "
            "(known yfinance limitation — see CMM_REPLICATION_ISSUES.md §7).",
            stacklevel=2,
        )

    month_ends = close.resample("ME").last().index.dropna()
    min_days = TRADING_DAYS_PER_YEAR + SKIP_DAYS + 30

    rows_char, rows_ret, rows_next, rows_date, rows_ticker, rows_mcap = [], [], [], [], [], []

    for i in range(1, len(month_ends) - 1):
        me = month_ends[i]
        next_me = month_ends[i + 1]
        me_loc = close.index.get_indexer([me], method="ffill")[0]
        next_loc = close.index.get_indexer([next_me], method="ffill")[0]
        if me_loc < min_days:
            continue

        for col in close.columns:
            ticker = col
            px = close[ticker].dropna()
            if len(px) < min_days:
                continue
            try:
                # Daily returns r_{t-252:t-22}: 231 returns
                start_i = me_loc - TRADING_DAYS_PER_YEAR - 1
                end_i = me_loc - SKIP_DAYS
                if start_i < 0 or end_i <= start_i:
                    continue
                window = px.iloc[start_i : end_i + 1].values.astype(np.float64)
                if len(window) < 2:
                    continue
                rets = np.diff(np.log(window + 1e-10))
                if len(rets) < N_DAILY_RETURNS:
                    pad = np.zeros(N_DAILY_RETURNS, dtype=np.float64)
                    pad[-len(rets) :] = rets
                    rets = pad
                else:
                    rets = rets[-N_DAILY_RETURNS:]

                # Next-month return
                p0 = float(px.iloc[me_loc])
                p1_arr = px.iloc[me_loc : next_loc + 1].dropna()
                if len(p1_arr) == 0 or p0 <= 0:
                    continue
                p1 = float(p1_arr.iloc[-1])
                next_ret = np.log(p1 / p0)

                # Characteristics: price-based only (no fundamentals - avoids look-ahead)
                char_data = px.iloc[: me_loc + 1].values
                chars = _compute_chars_simple(char_data, n_char)

                # Market cap at formation date (§3). Fall back to price if
                # shares outstanding were unavailable for this ticker.
                mcap_val = float("nan")
                if ticker in market_cap.columns:
                    mcap_series = market_cap[ticker].iloc[: me_loc + 1].dropna()
                    if len(mcap_series) > 0:
                        mcap_val = float(mcap_series.iloc[-1])
                if not np.isfinite(mcap_val) or mcap_val <= 0:
                    mcap_val = p0  # proxy fallback (documented bias)

                rows_char.append(chars)
                rows_ret.append(rets)
                rows_next.append(next_ret)
                rows_date.append(me)
                rows_ticker.append(ticker)
                rows_mcap.append(mcap_val)
            except Exception:
                continue

    if not rows_char:
        raise ValueError("No valid data. Try expanding date range or reducing tickers.")

    C = np.array(rows_char, dtype=np.float64)
    R = np.array(rows_ret, dtype=np.float64)
    Y = np.array(rows_next, dtype=np.float64)
    dates_series = pd.Series(rows_date)
    uniq = dates_series.unique()
    D = dates_series.map({d: i for i, d in enumerate(uniq)}).values
    timestamps = np.array(rows_date, dtype="datetime64[ns]")
    tickers_arr = np.array(rows_ticker, dtype=object)
    # Previously this was share price ("size"). Now it is market cap at the
    # formation month-end, used by portfolio.py for value weighting (§3).
    market_cap_arr = np.array(rows_mcap, dtype=np.float64)

    # NYSE membership per row (§6). yfinance has no historical exchange
    # listings, so we use a current NYSE ticker set as a proxy — this is a
    # known bias that the JKP loader resolves properly via crsp_exchcd.
    nyse_set = get_nyse_tickers()
    if nyse_set is None:
        is_nyse = np.ones(len(tickers_arr), dtype=bool)  # fallback: all NYSE
    else:
        is_nyse = np.array([t in nyse_set for t in tickers_arr], dtype=bool)

    print(f"  Built {len(rows_char)} stock-month observations")
    # Regime signal is all-zeros for yfinance mode (no SPX trend wired here).
    # Industries are all 99 (single bucket = no adjustment).
    # Both kept for tuple-shape consistency with the JKP loader.
    regime_sig_arr = np.zeros(len(tickers_arr), dtype=np.float64)
    industries = np.full(len(tickers_arr), 99, dtype=np.int32)
    return C, R, Y, D, timestamps, tickers_arr, market_cap_arr, is_nyse, regime_sig_arr, industries
