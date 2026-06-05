"""
JKP-based data loader for CMM (resolves §1, §2, §3, §5, §6, §7 of
CMM_REPLICATION_ISSUES.md in one shot).

Pulls:
1. Monthly firm characteristics (153 already rank-transformed to [-0.5, 0.5])
   and market equity `me` from `contrib.global_factor` on WRDS.
2. Daily returns from `crsp.dsf` to build the 231-day formation window
   r_{t-252:t-22} at each month-end.

Returns the same 7-tuple as `fetch_sp500_cmm_data` for drop-in use in
`run_expanding_window`.

Setup
-----
1. Register for a WRDS account with your university email at
   https://wrds-www.wharton.upenn.edu/register/ (most unis are pre-subscribed).
2. Request access to "Global Factor Data" under Contributed Data.
3. `pip install wrds` and run once: `python -c "import wrds; wrds.Connection()"`
   to cache credentials in ~/.pgpass.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Disk cache for the CRSP daily panel. First run populates it after a
# successful WRDS fetch; subsequent runs load from disk in ~seconds
# instead of re-querying WRDS for 15 minutes.
_CRSP_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "crsp"

from cmm.macro import MACRO_FEATURE_NAMES, fetch_macro_features
from cmm.regime import spx_trend_feature


# JKP character names we interact with the SPX trend regime signal (D).
# If a char name isn't in the panel, the loader silently skips that
# interaction and logs the fallback. Picking chars that are *known* to
# be regime-sensitive: 12-1 momentum, 1-month reversal, size.
_D_INTERACTION_CANDIDATES: list[str] = [
    "ret_12_1",
    "ret_1_0",
    "me",  # market equity rank
]

try:
    import wrds  # type: ignore
    _HAS_WRDS = True
except ImportError:
    _HAS_WRDS = False


# Columns in contrib.global_factor that are NOT characteristics.
# Used to auto-discover the 153 characteristic column names from the
# table schema rather than hardcoding a list that could drift.
_NON_CHAR_COLS = {
    "id", "eom", "excntry", "gvkey", "permno", "iid", "size_grp",
    "source_crsp", "common", "exch_main", "primary_sec", "obs_main",
    "crsp_exchcd", "crsp_shrcd", "adjfct", "shares", "prc",
    "me", "me_company", "ret_exc", "ret_exc_lead1m", "ret_lead1m",
    "ret", "dolvol", "year", "month",
    # Industry-code columns (any of these are used for industry adjustment,
    # not as model features).
    "sic", "siccd", "crsp_siccd", "industry",
}

# Minimum days of history needed to build a 231-day formation window
# ending at t-22.
_TRADING_DAYS_WINDOW = 252  # t-252 to t-1 = 252 days, then skip last 22
_SKIP_DAYS = 22
_N_DAILY_RETURNS = 231


def _connect_wrds() -> "wrds.Connection":
    if not _HAS_WRDS:
        raise ImportError(
            "wrds package is required for the JKP loader. "
            "Install with: pip install wrds"
        )
    return wrds.Connection()


def _detect_industry_column(conn) -> Optional[str]:
    """
    Return the name of an industry-code column in contrib.global_factor
    if one is present, else None. Tries common names in order.
    """
    candidates = ["sic", "siccd", "crsp_siccd", "industry"]
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'contrib'
          AND table_name   = 'global_factor'
    """
    cols = set(conn.raw_sql(q)["column_name"].str.lower().tolist())
    for c in candidates:
        if c in cols:
            return c
    return None


def _discover_char_columns(conn) -> list[str]:
    """
    Query the WRDS information_schema to find every numeric column in
    contrib.global_factor that isn't an identifier or metadata field.
    Should yield the ~153 JKP characteristics.
    """
    q = """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'contrib'
          AND table_name   = 'global_factor'
          AND data_type IN ('double precision', 'numeric', 'real', 'integer')
        ORDER BY ordinal_position
    """
    df = conn.raw_sql(q)
    cols = [c for c in df["column_name"].tolist() if c not in _NON_CHAR_COLS]
    if len(cols) < 100:
        raise RuntimeError(
            f"Expected ~153 characteristic columns, found {len(cols)}. "
            "Schema may have changed — inspect contrib.global_factor."
        )
    return cols


def _load_monthly_panel(
    conn,
    start_date: str,
    end_date: str,
    char_cols: list[str],
) -> pd.DataFrame:
    """
    Pull the US monthly panel with the paper's mandatory screens
    (common stock, main exchange, primary security, CRSP-preferred).
    """
    # crsp_exchcd is pulled for per-month NYSE membership (§6).
    sel_cols = ["permno", "eom", "me", "ret_exc_lead1m", "crsp_exchcd"] + char_cols
    # Probe for an industry-code column. JKP's vintage / contributor may
    # name it `sic`, `siccd`, `crsp_siccd`, or omit it entirely. We add
    # whichever is present so industry-adjusted returns can use it.
    industry_col = _detect_industry_column(conn)
    if industry_col is not None:
        sel_cols.append(industry_col)
    sel_sql = ", ".join(sel_cols)
    q = f"""
        SELECT {sel_sql}
        FROM contrib.global_factor
        WHERE common = 1
          AND exch_main = 1
          AND primary_sec = 1
          AND obs_main = 1
          AND excntry = 'USA'
          AND eom BETWEEN %(start)s AND %(end)s
    """
    df = conn.raw_sql(q, params={"start": start_date, "end": end_date}, date_cols=["eom"])
    # Drop rows missing the target or market equity (can't build portfolios)
    df = df.dropna(subset=["me", "ret_exc_lead1m"])
    return df


def _load_daily_returns(
    conn,
    permnos: list[int],
    start_date: str,
    end_date: str,
    chunk_years: int = 3,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Pull daily returns from CRSP, chunked by date range (not permno list).

    A single query with a 5000-permno IN clause over 55 years of CRSP DSF
    reliably kills the WRDS server connection (~1 billion row intermediate
    result, times out the psql socket). Instead we:

    1. Issue one query per `chunk_years`-year date range with NO permno
       filter (just date + ret != NULL).
    2. Filter to our permno set locally (fast pandas operation).

    Overhead vs. IN-clause approach: we over-pull maybe 30-40% of rows
    (CRSP DSF contains some securities filtered out by our JKP screens).
    But total wall-clock time is lower because each query is small enough
    to complete before the server times it out.

    Retries up to `max_retries` times per chunk on connection errors,
    recreating the WRDS connection as needed.
    """
    from sqlalchemy.exc import OperationalError, DBAPIError  # local import

    try:
        import wrds  # type: ignore
    except ImportError:
        wrds = None  # type: ignore

    permno_set = set(int(p) for p in permnos)
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    results: list[pd.DataFrame] = []
    current = start_ts
    chunk_idx = 0
    total_chunks = int(np.ceil((end_ts.year - start_ts.year + 1) / chunk_years))

    while current <= end_ts:
        chunk_end = min(
            current + pd.DateOffset(years=chunk_years) - pd.Timedelta(days=1),
            end_ts,
        )
        chunk_idx += 1
        label = f"[{chunk_idx}/{total_chunks}] {current.date()} → {chunk_end.date()}"

        q = """
            SELECT permno, date, ret
            FROM crsp.dsf
            WHERE date BETWEEN %(start)s AND %(end)s
              AND ret IS NOT NULL
        """
        params = {"start": current.date(), "end": chunk_end.date()}

        df = None
        for attempt in range(max_retries):
            try:
                t0 = time.time()
                df = conn.raw_sql(q, params=params, date_cols=["date"])
                elapsed = time.time() - t0
                # Filter to our permno set
                df = df[df["permno"].astype(np.int64).isin(permno_set)].copy()
                print(f"  {label}: {len(df):,} rows ({elapsed:.1f}s)", flush=True)
                break
            except (OperationalError, DBAPIError) as e:
                print(
                    f"  {label}: connection error "
                    f"(attempt {attempt + 1}/{max_retries}): {type(e).__name__}",
                    flush=True,
                )
                if attempt == max_retries - 1:
                    raise
                # Try to reconnect
                try:
                    conn.close()
                except Exception:
                    pass
                time.sleep(10 * (attempt + 1))
                if wrds is not None:
                    conn = wrds.Connection()

        if df is not None and len(df) > 0:
            results.append(df)

        current = chunk_end + pd.Timedelta(days=1)

    if not results:
        return pd.DataFrame(columns=["permno", "date", "ret"])
    out = pd.concat(results, ignore_index=True)
    out["permno"] = out["permno"].astype(np.int64)
    out["ret"] = out["ret"].astype(np.float64)
    return out


def _build_daily_return_windows(
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    n_ret: int = _N_DAILY_RETURNS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each (permno, eom) row in `monthly`, build the 231-day log-return
    vector r_{t-252:t-22}. Returns (R, valid_mask) where R has shape
    (n_monthly_rows, n_ret) and valid_mask indicates which rows had enough
    daily history.

    Uses the paper's convention: 231 log returns ending 22 trading days
    before month-end, so the most recent month is excluded (short-term
    reversal control).
    """
    # Precompute log returns and a sorted per-permno index
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    # CRSP `ret` is a simple return, not log — convert for the softmax mechanism
    # to match the paper's "raw daily log return" convention (§4).
    daily["logret"] = np.log1p(daily["ret"].clip(lower=-0.999))

    # Winsorize log returns at ±0.4 (~±49% simple return). A single-day
    # move beyond this almost always reflects splits, mergers, or data
    # errors rather than a real return, and extreme tails destabilize
    # the softmax: one huge |r| dominates scores across all stocks
    # simultaneously.
    daily["logret"] = daily["logret"].clip(-0.4, 0.4)

    # Group daily returns by permno into numpy arrays for fast slicing
    groups = {
        pn: (g["date"].values, g["logret"].values)
        for pn, g in daily.groupby("permno", sort=False)
    }

    n = len(monthly)
    R = np.zeros((n, n_ret), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    # Caller must have already dropped NaN permnos (see fetch_jkp_cmm_data).
    # Assert to catch future mistakes.
    assert monthly["permno"].notna().all(), (
        "monthly['permno'] contains NaN — caller must dropna before calling "
        "_build_daily_return_windows so R/valid are sized correctly."
    )
    monthly_pn = monthly["permno"].to_numpy().astype(np.int64)
    monthly_eom = monthly["eom"].to_numpy().astype("datetime64[D]")

    for i in range(n):
        pn = int(monthly_pn[i])
        eom = monthly_eom[i]
        if pn not in groups:
            continue
        dates, logrets = groups[pn]
        # Find the last date on/before eom
        me_pos = np.searchsorted(dates, eom, side="right") - 1
        if me_pos < 0:
            continue
        # Window: last `_TRADING_DAYS_WINDOW` days ending at me_pos,
        # skipping the most recent `_SKIP_DAYS` → 231 returns.
        end_i = me_pos - _SKIP_DAYS
        start_i = end_i - n_ret + 1
        if start_i < 0 or end_i < start_i:
            continue
        window = logrets[start_i : end_i + 1]
        if len(window) != n_ret:
            continue
        R[i] = window
        valid[i] = True

    return R, valid


def fetch_jkp_cmm_data(
    start_date: str = "1973-01-01",
    end_date: Optional[str] = None,
    n_char: int = 153,
    char_cols_override: Optional[list[str]] = None,
    include_macro: bool = False,
    include_d_interactions: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Drop-in replacement for `fetch_sp500_cmm_data` that sources data from
    the JKP panel + CRSP daily returns on WRDS.

    `include_macro` is **off by default**. Concatenating macro features
    (CPI, VIX, etc.) into the FFN input has known issues: (1) the target
    is cross-sectionally normalized per month, so any feature constant
    across stocks in a given month contributes zero direct gradient; (2)
    effective sample size for macro state is tiny (~50 independent regime
    draws in 50 years). Prefer volatility-managed HML (portfolio.py) over
    macro concat. Kept as opt-in for experimentation — see cmm/macro.py.

    Returns
    -------
    characteristics : (n_obs, n_char)  — JKP chars (+ interactions/macro)
    daily_returns   : (n_obs, 231)     — log returns r_{t-252:t-22}
    next_month_ret  : (n_obs,)         — ret_exc_lead1m (excess return)
    dates           : (n_obs,)         — integer month index for grouping
    timestamps      : (n_obs,) datetime64
    tickers         : (n_obs,) permno as string (used as universe id)
    market_cap      : (n_obs,)         — `me` from JKP (shares * price)
    is_nyse         : (n_obs,) bool    — crsp_exchcd == 1, per month (§6)
    regime_signal   : (n_obs,) float   — trailing SPX-12m rank in [-0.5, 0.5],
                                         broadcast to every stock in month t.
                                         Used by (C) regime-conditional
                                         ensemble. Safe to ignore if not using.
    industries      : (n_obs,) int32   — 2-digit SIC industry code per row,
                                         used by Lewellen-style industry
                                         adjustment in build_hml_portfolio.
                                         99 = unknown / fallback bucket.
    """
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # Daily returns must start ~1.5 years before the first month-end so the
    # t-252 lookback is populated for the earliest training month.
    daily_start = (pd.Timestamp(start_date) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")

    print("Connecting to WRDS...")
    conn = _connect_wrds()

    print("Discovering characteristic columns...")
    char_cols = char_cols_override or _discover_char_columns(conn)
    if n_char < len(char_cols):
        char_cols = char_cols[:n_char]
    print(f"  Using {len(char_cols)} characteristics")

    print(f"Loading monthly panel ({start_date} → {end_date})...")
    monthly = _load_monthly_panel(conn, start_date, end_date, char_cols)
    print(f"  {len(monthly):,} stock-months across {monthly['permno'].nunique():,} permnos")

    # CRSP daily returns — slowest part of the fetch. Cache to parquet on
    # first successful load so reruns are fast.
    _CRSP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"crsp_dsf_{daily_start}_{end_date}.parquet"
    cache_path = _CRSP_CACHE_DIR / cache_key
    permnos = monthly["permno"].dropna().astype(np.int64).unique().tolist()
    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"Loading CRSP daily returns from cache: {cache_path.name}...")
        daily = pd.read_parquet(cache_path)
        # Defensive: filter to our current permno set in case the cache
        # was built from a different universe.
        daily = daily[daily["permno"].isin(permnos)].copy()
        print(f"  {len(daily):,} daily observations (cached)")
    else:
        print(f"Loading CRSP daily returns ({daily_start} → {end_date})...")
        daily = _load_daily_returns(conn, permnos, daily_start, end_date)
        print(f"  {len(daily):,} daily observations")
        try:
            daily.to_parquet(cache_path, index=False)
            print(f"  Cached to {cache_path.name}")
        except Exception as e:
            print(f"  [warn] cache write failed: {e}")

    # Drop rows with NaN permno. Must happen HERE (not inside
    # _build_daily_return_windows) so the caller's `monthly` and the
    # returned `valid` mask stay the same length — otherwise the
    # `monthly.loc[valid]` filter below misaligns.
    if monthly["permno"].isna().any():
        n_before = len(monthly)
        monthly = monthly.dropna(subset=["permno"]).reset_index(drop=True)
        print(f"  [warn] dropped {n_before - len(monthly)} rows with NaN permno")

    print("Building 231-day formation windows...")
    R, valid = _build_daily_return_windows(monthly, daily)
    print(f"  {valid.sum():,}/{len(valid):,} rows have full history")

    monthly = monthly.loc[valid].reset_index(drop=True)
    R = R[valid]

    # Macro features (FRED → rolling rank → lagged). Broadcast to every
    # stock-month row by joining on eom. Missing values (e.g. VIX pre-1990,
    # or the first 12 months where YoY is undefined) are filled with 0 =
    # median under the rank convention. Adds 7 extra "characteristic"
    # columns to the FFN input.
    if include_macro:
        print("Fetching macro features from FRED...")
        macro_start = (pd.Timestamp(start_date) - pd.DateOffset(years=6)).strftime("%Y-%m-%d")
        macro = fetch_macro_features(start=macro_start, end=end_date)
        # Align macro's month-end index to the panel's eom (normalized)
        macro.index = pd.DatetimeIndex(macro.index).normalize()
        eom_norm = pd.DatetimeIndex(monthly["eom"]).normalize()
        macro_rows = macro.reindex(eom_norm)
        macro_arr = macro_rows[MACRO_FEATURE_NAMES].to_numpy(dtype=np.float64)
        macro_arr = np.nan_to_num(macro_arr, nan=0.0)
        coverage = {
            name: int((~macro_rows[name].isna()).sum())
            for name in MACRO_FEATURE_NAMES
        }
        print(f"  Macro coverage (rows with data): {coverage}")
    else:
        macro_arr = np.zeros((len(monthly), 0), dtype=np.float64)

    # Characteristics: fill remaining NaNs with 0 (already rank-transformed,
    # so 0 corresponds to the cross-sectional median per JKP convention).
    C_jkp = monthly[char_cols].to_numpy(dtype=np.float64)
    C_jkp = np.nan_to_num(C_jkp, nan=0.0)

    # (D) Regime-aware interactions: char_i × SPX_trend_12m.
    # These per-stock features encode "the predictive power of characteristic X
    # is amplified/dampened by current market trend." Cross-sectional
    # normalization does NOT kill them because they vary with the stock-specific
    # char value. See CMM_REPLICATION_ISSUES critique Option B.
    print("Fetching SPX regime signal (yfinance ^GSPC)...")
    eom_dates = pd.DatetimeIndex(monthly["eom"]).normalize()
    regime_sig = spx_trend_feature(eom_dates, lookback_months=12, rank_window=60)
    regime_sig_arr = regime_sig.to_numpy(dtype=np.float64)
    # Clip NaNs (insufficient history) to 0 = neutral
    regime_sig_arr = np.nan_to_num(regime_sig_arr, nan=0.0)

    interaction_cols: list[str] = []
    interaction_arrays: list[np.ndarray] = []
    if include_d_interactions:
        for c in _D_INTERACTION_CANDIDATES:
            if c in char_cols:
                idx = char_cols.index(c)
                col_vals = C_jkp[:, idx]
                interaction_arrays.append(col_vals * regime_sig_arr)
                interaction_cols.append(f"{c}_x_spxtrend12m")
            else:
                print(f"  [D] '{c}' not in JKP panel — skipping this interaction")
        if interaction_cols:
            print(f"  [D] Added interaction features: {interaction_cols}")

    if interaction_arrays:
        C_interactions = np.column_stack(interaction_arrays)
    else:
        C_interactions = np.zeros((len(monthly), 0), dtype=np.float64)

    blocks = [C_jkp]
    if macro_arr.shape[1] > 0:
        blocks.append(macro_arr)
    if C_interactions.shape[1] > 0:
        blocks.append(C_interactions)
    C = np.hstack(blocks)

    Y = monthly["ret_exc_lead1m"].to_numpy(dtype=np.float64)
    MCAP = monthly["me"].to_numpy(dtype=np.float64)

    # Per-month NYSE membership from CRSP exchange code (§6). exchcd == 1 = NYSE.
    # Missing exchcd is treated as non-NYSE (conservative — stock is excluded
    # from breakpoint calculation that month).
    is_nyse = (monthly["crsp_exchcd"].fillna(-1).astype(int) == 1).to_numpy()

    # Per-row 2-digit industry code for Lewellen-style industry adjustment.
    # If JKP didn't have an industry column, fill with 99 (single bucket =
    # adjustment becomes a no-op, which is the right default).
    ind_col = next(
        (c for c in ("sic", "siccd", "crsp_siccd", "industry") if c in monthly.columns),
        None,
    )
    if ind_col is not None:
        sic = monthly[ind_col].fillna(9999).astype(int).to_numpy()
        # 2-digit SIC = 80-ish industries, granular enough but not too small.
        industries = (sic // 100).astype(np.int32)
        n_unknown = int((industries == 99).sum())  # 9999 // 100 = 99
        n_known = len(industries) - n_unknown
        print(f"  Industry codes: {n_known:,}/{len(industries):,} rows have known SIC")
    else:
        industries = np.full(len(monthly), 99, dtype=np.int32)
        print("  [warn] no industry column found in JKP panel — industry adjust will be a no-op")

    timestamps = monthly["eom"].to_numpy(dtype="datetime64[ns]")
    uniq_dates = pd.Series(timestamps).unique()
    date_to_idx = {d: i for i, d in enumerate(uniq_dates)}
    D = np.array([date_to_idx[d] for d in timestamps], dtype=np.int64)

    tickers_arr = monthly["permno"].astype(np.int64).astype(str).to_numpy(dtype=object)

    try:
        conn.close()
    except Exception:
        pass

    print(f"Done. {len(C):,} stock-month observations ready for CMM training.")
    print(f"  n_char total: {C.shape[1]} (153 JKP + {C.shape[1] - 153} extras)")
    print(f"  NYSE rows: {is_nyse.sum():,}/{len(is_nyse):,}")
    return C, R, Y, D, timestamps, tickers_arr, MCAP, is_nyse, regime_sig_arr, industries
