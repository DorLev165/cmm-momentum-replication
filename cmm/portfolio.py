"""
Portfolio Construction for CMM.

- Sort stocks into deciles by CMM signal using NYSE breakpoints
- Value-weight within each decile
- HML: explicitly dollar-neutral long-short (long leg weights sum to +1, short to -1)
"""

import urllib.request
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd

from cmm.covariance import ledoit_wolf_constant_corr


def get_nyse_tickers() -> Optional[set[str]]:
    """
    Fetch NYSE-listed tickers for breakpoint calculation.
    Returns None if fetch fails (caller uses all stocks).
    """
    url = "https://raw.githubusercontent.com/datasets/nyse-listings/master/data/nyse-listed.csv"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            text = resp.read().decode()
        df = pd.read_csv(StringIO(text))
        col = "ACT Symbol" if "ACT Symbol" in df.columns else df.columns[0]
        tickers = set()
        for s in df[col].dropna().astype(str):
            t = str(s).split("$")[0].replace(".", "-").strip()
            if 1 <= len(t) <= 6 and t != "nan":
                tickers.add(t)
        return tickers
    except Exception:
        return None


def nyse_breakpoints(
    signal: np.ndarray,
    is_nyse: Optional[np.ndarray],
) -> np.ndarray:
    """
    Decile breakpoints (10th..90th percentile) computed from NYSE stocks
    only (§6). `is_nyse` is a per-row bool array aligned with `signal`;
    if None or has <10 NYSE rows, falls back to all stocks.
    """
    if is_nyse is not None:
        mask = np.asarray(is_nyse, dtype=bool)
        if mask.sum() < 10:
            mask = np.ones(len(signal), dtype=bool)
        signal_use = signal[mask]
    else:
        signal_use = signal

    valid = np.isfinite(signal_use)
    if valid.sum() < 10:
        return np.percentile(np.nan_to_num(signal), [10, 20, 30, 40, 50, 60, 70, 80, 90])

    return np.percentile(signal_use[valid], [10, 20, 30, 40, 50, 60, 70, 80, 90])


def assign_deciles(
    signal: np.ndarray,
    breakpoints: np.ndarray,
) -> np.ndarray:
    """
    Assign each stock to decile 1 (low) .. 10 (high) based on breakpoints.

    Decile 1 = lowest CMM signal, Decile 10 = highest.
    """
    deciles = np.zeros(len(signal), dtype=int)
    bounds = [-np.inf] + list(breakpoints) + [np.inf]
    for d in range(1, 11):
        lo, hi = bounds[d - 1], bounds[d]
        deciles[(signal >= lo) & (signal < hi) & np.isfinite(signal)] = d
    deciles[(signal >= breakpoints[-1]) & np.isfinite(signal)] = 10
    return deciles


def value_weighted_return(
    returns: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Portfolio return = sum(w * r) / sum(w)."""
    w = np.where(np.isfinite(returns) & np.isfinite(weights) & (weights > 0), weights, 0.0)
    if w.sum() <= 0:
        return 0.0
    return np.nansum(w * np.nan_to_num(returns, nan=0)) / w.sum()


def build_hml_portfolio(
    signal: np.ndarray,
    next_returns: np.ndarray,
    market_cap: np.ndarray,
    tickers: np.ndarray,
    timestamps: np.ndarray,
    is_nyse: Optional[np.ndarray] = None,
    industries: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Build decile portfolios and HML return for each month.

    Decile returns (D1–D10) are value-weighted by market cap within each
    decile. Breakpoints use NYSE-only stocks when `is_nyse` is provided
    (§6). HML is built explicitly dollar-neutral at the stock level: long
    leg (D10) weights sum to +1, short leg (D1) weights sum to -1.

    Parameters
    ----------
    signal : (n_obs,) CMM signal (E_CMM) per stock-month
    next_returns : (n_obs,) realized next-month return
    market_cap : (n_obs,) market capitalization (shares * price) at formation
    tickers : (n_obs,) ticker per row
    timestamps : (n_obs,) month-end date per row
    is_nyse : optional (n_obs,) bool, per-row NYSE membership for breakpoints

    Returns
    -------
    DataFrame with columns: date, D1_ret, ..., D10_ret, HML_ret, n_stocks
    """
    ts = pd.to_datetime(timestamps)
    months = ts.normalize()
    results = []

    for month in months.unique():
        mask = months == month
        if mask.sum() < 20:
            continue

        sig_m = signal[mask]
        ret_m = next_returns[mask]
        mcap_m = np.maximum(market_cap[mask], 1e-10)
        nyse_m = is_nyse[mask] if is_nyse is not None else None

        # Lewellen-style industry adjustment: replace ret_m with returns
        # net of same-month, same-industry mean. This neutralizes industry
        # tilts in the decile sort and isolates within-industry alpha.
        if industries is not None:
            ret_m = industry_adjust_returns(ret_m, industries[mask])

        bp = nyse_breakpoints(sig_m, nyse_m)
        dec = assign_deciles(sig_m, bp)

        dec_rets = []
        for d in range(1, 11):
            dm = dec == d
            if dm.sum() == 0:
                dec_rets.append(0.0)
                continue
            r = value_weighted_return(ret_m[dm], mcap_m[dm])
            dec_rets.append(r)

        # Explicit dollar-neutral: stock-level weights long +1, short -1
        w = np.zeros(len(ret_m))
        d1 = dec == 1
        d10 = dec == 10
        s1 = np.where(d1, mcap_m, 0.0).sum()
        s10 = np.where(d10, mcap_m, 0.0).sum()
        if s1 > 0:
            w[d1] = -mcap_m[d1] / s1
        if s10 > 0:
            w[d10] = mcap_m[d10] / s10
        hml = float(np.nansum(w * np.nan_to_num(ret_m, nan=0)))

        results.append({
            "date": month,
            "D1_ret": dec_rets[0],
            "D2_ret": dec_rets[1],
            "D3_ret": dec_rets[2],
            "D4_ret": dec_rets[3],
            "D5_ret": dec_rets[4],
            "D6_ret": dec_rets[5],
            "D7_ret": dec_rets[6],
            "D8_ret": dec_rets[7],
            "D9_ret": dec_rets[8],
            "D10_ret": dec_rets[9],
            "HML_ret": hml,
            "n_stocks": mask.sum(),
        })

    return pd.DataFrame(results)


def industry_adjust_returns(
    returns: np.ndarray,
    industries: np.ndarray,
    min_size: int = 5,
) -> np.ndarray:
    """
    Subtract per-industry mean return from each stock's return — Lewellen
    (2015) industry-relative-return convention. Industries with fewer
    than `min_size` stocks in the cross-section are NOT adjusted (the
    industry mean is too noisy to subtract reliably with so few stocks).

    Parameters
    ----------
    returns : (n_stocks,) array of next-month returns
    industries : (n_stocks,) integer industry codes (e.g., 2-digit SIC)
    min_size : skip adjustment for industries with fewer than this many stocks

    Returns
    -------
    Adjusted returns array, same shape. Non-adjusted stocks (small or
    NaN industry) keep their raw return.
    """
    out = returns.astype(np.float64, copy=True)
    if industries is None:
        return out
    industries = np.asarray(industries)
    finite_ret = np.isfinite(returns)
    for ind in np.unique(industries):
        mask = (industries == ind) & finite_ret
        if mask.sum() >= min_size:
            ind_mean = float(np.nanmean(returns[mask]))
            out[mask] = returns[mask] - ind_mean
    return out


def mean_variance_weights(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_sum: float = 1.0,
    long_only: bool = True,
    min_stocks: int = 3,
) -> np.ndarray:
    """
    Sharpe-maximizing weights proportional to Sigma^{-1} * mu, rescaled to
    sum to `target_sum`. Used for the long leg (target_sum=+1) and the
    short leg (caller passes -mu and target_sum=+1, then negates).

    Parameters
    ----------
    mu : (n,) predicted scores (E_CMM) for the stocks in this leg
    Sigma : (n, n) covariance matrix, already shrunk. Must be PSD.
    target_sum : target sum of weights (e.g., +1 for long leg).
    long_only : if True, clip negative weights to zero before renormalizing.
                Preserves the paper's "long only within the top decile"
                structure. Set False for unconstrained tangency portfolio.
    min_stocks : fallback if fewer than this many valid stocks — use
                 equal-weight across valid stocks.
    """
    n = len(mu)
    if n < min_stocks:
        if n == 0:
            return np.zeros(0)
        return np.full(n, target_sum / n)

    # Subtract cross-sectional mean of mu — this is what makes the weights
    # "active" (zero-investment in a constant signal → zero positions).
    mu_c = mu - mu.mean()

    # Solve Sigma * w = mu_c (more stable than explicit inverse)
    try:
        w_raw = np.linalg.solve(Sigma, mu_c)
    except np.linalg.LinAlgError:
        # Singular — fall back to diagonal (inverse-variance)
        d = np.maximum(np.diag(Sigma), 1e-12)
        w_raw = mu_c / d

    if long_only:
        # Keep only positive signals (stocks we actually want to buy in
        # this leg). Non-positive get zero weight.
        w_raw = np.maximum(w_raw, 0.0)

    s = w_raw.sum()
    if s <= 0:
        # All weights zero or negative after clipping → fall back to equal
        return np.full(n, target_sum / n)

    return (w_raw / s) * target_sum


def build_hml_portfolio_mv(
    signal: np.ndarray,
    next_returns: np.ndarray,
    market_cap: np.ndarray,
    tickers: np.ndarray,
    timestamps: np.ndarray,
    daily_returns: np.ndarray,
    is_nyse: Optional[np.ndarray] = None,
    industries: Optional[np.ndarray] = None,
    cov_lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Mean-variance version of `build_hml_portfolio`.

    Same long/short decile structure as the paper, but within each decile
    the stocks are weighted by w* ~ Sigma^{-1} * mu rather than by market
    cap. Covariance is estimated per-month from the recent `cov_lookback_days`
    daily returns (default 252), using Ledoit-Wolf constant-correlation
    shrinkage.

    Parameters
    ----------
    signal : (n_obs,) E_CMM per stock-month
    next_returns : (n_obs,) realized next-month return
    market_cap : (n_obs,) used only for long-only clipping fallback
    tickers : (n_obs,) per row
    timestamps : (n_obs,) month-end date
    daily_returns : (n_obs, >=cov_lookback_days) per-row daily return vectors
                    (reuses the FFN's 231-day input as the covariance window;
                    trailing portion is used)
    is_nyse : optional (n_obs,) for breakpoints
    cov_lookback_days : number of recent daily obs used for Σ. If the
                         daily_returns matrix has fewer columns, uses all
                         available.
    """
    ts = pd.to_datetime(timestamps)
    months = ts.normalize()
    results = []

    n_daily = daily_returns.shape[1]
    lookback = min(cov_lookback_days, n_daily)

    for month in months.unique():
        mask = months == month
        if mask.sum() < 20:
            continue

        sig_m = signal[mask]
        ret_m = next_returns[mask]
        dr_m = daily_returns[mask][:, -lookback:]  # (n_stocks, lookback)
        nyse_m = is_nyse[mask] if is_nyse is not None else None

        # Lewellen-style industry adjustment (see build_hml_portfolio).
        if industries is not None:
            ret_m = industry_adjust_returns(ret_m, industries[mask])

        bp = nyse_breakpoints(sig_m, nyse_m)
        dec = assign_deciles(sig_m, bp)

        # ------ decile returns for diagnostics (still value-weighted) ------
        mcap_m = np.maximum(market_cap[mask], 1e-10)
        dec_rets = []
        for d in range(1, 11):
            dm = dec == d
            if dm.sum() == 0:
                dec_rets.append(0.0)
                continue
            r = value_weighted_return(ret_m[dm], mcap_m[dm])
            dec_rets.append(r)

        # ------ mean-variance HML ------
        w = np.zeros(len(ret_m))

        long_mask = dec == 10
        short_mask = dec == 1

        # Build LW covariance for each leg's universe.
        # Transpose the (n_stocks, lookback) -> (lookback, n_stocks) that LW expects.
        for leg_mask, leg_target in [(long_mask, +1.0), (short_mask, -1.0)]:
            if leg_mask.sum() < 3:
                continue
            leg_returns = dr_m[leg_mask].T  # (lookback, n_stocks_in_leg)
            # Drop any all-NaN columns (shouldn't happen but defensive)
            finite = np.isfinite(leg_returns).all(axis=0)
            if finite.sum() < 3:
                continue
            valid_rows = np.where(leg_mask)[0][finite]
            leg_returns = leg_returns[:, finite]
            mu_leg = sig_m[valid_rows]

            try:
                Sigma_leg, _ = ledoit_wolf_constant_corr(leg_returns)
            except Exception:
                # Last-resort: use sample diagonal
                Sigma_leg = np.diag(np.maximum(leg_returns.var(axis=0, ddof=0), 1e-12))

            # For short leg: we want large-negative-mu stocks weighted heavily
            # as shorts. Flip sign of mu and produce positive-summing weights,
            # then negate after.
            if leg_target > 0:
                w_leg = mean_variance_weights(mu_leg, Sigma_leg, target_sum=+1.0, long_only=True)
            else:
                w_leg = -mean_variance_weights(-mu_leg, Sigma_leg, target_sum=+1.0, long_only=True)
            w[valid_rows] = w_leg

        hml = float(np.nansum(w * np.nan_to_num(ret_m, nan=0)))

        results.append({
            "date": month,
            "D1_ret": dec_rets[0],
            "D2_ret": dec_rets[1],
            "D3_ret": dec_rets[2],
            "D4_ret": dec_rets[3],
            "D5_ret": dec_rets[4],
            "D6_ret": dec_rets[5],
            "D7_ret": dec_rets[6],
            "D8_ret": dec_rets[7],
            "D9_ret": dec_rets[8],
            "D10_ret": dec_rets[9],
            "HML_ret": hml,
            "n_stocks": int(mask.sum()),
        })

    return pd.DataFrame(results)


def vol_managed_returns(
    hml_returns: pd.Series,
    target_ann_vol: float = 0.12,
    lookback_months: int = 6,
    max_leverage: float = 3.0,
    min_obs: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """
    Volatility-managed version of an HML return series (Barroso &
    Santa-Clara 2015, Moreira & Muir 2017). Scales monthly position size
    inversely to trailing realized vol, targeting a constant annualized
    vol of `target_ann_vol`.

    Leverage at month t is computed from information available *before* t:
        sigma_hat_t = std(r_{t-lookback : t-1})  (annualized)
        leverage_t  = clip(target_ann_vol / sigma_hat_t, 0, max_leverage)
        r_scaled_t  = leverage_t * r_t

    The clip at `max_leverage` prevents degenerate behavior when
    realized vol is near zero (e.g., first few months, quiet periods).
    The first `min_obs` months have leverage = 0 (insufficient history).

    Returns
    -------
    scaled : pd.Series  — vol-managed monthly returns
    leverage : pd.Series — applied leverage each month (for diagnostics)

    Notes
    -----
    This captures most of the regime-conditioning benefit of adding
    macro features to the cross-sectional model, with far less overfit
    risk: it's one hyperparameter, trailing-only, and operates at the
    portfolio level rather than inside the FFN.
    """
    r = hml_returns.copy().sort_index().dropna()
    if len(r) == 0:
        return r, r

    target_mo = target_ann_vol / np.sqrt(12)
    # Trailing std shifted 1 month so month-t scaling uses info through t-1
    rolling_sd = r.rolling(lookback_months, min_periods=min_obs).std().shift(1)
    leverage = (target_mo / rolling_sd).clip(lower=0.0, upper=max_leverage)
    leverage = leverage.fillna(0.0)
    scaled = leverage * r
    return scaled, leverage


def hml_summary(hml_returns: pd.Series) -> dict:
    """
    Summary stats for HML return series. Reports the headline metrics used
    in Beckmeyer & Wiedemann (2025) Table 1: annualized return, vol, Sharpe,
    and max drawdown on the compounded equity curve (§8.1).
    """
    r = hml_returns.dropna()
    if len(r) == 0:
        return {}

    equity = (1.0 + r).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    return {
        "mean_ret": float(r.mean()),
        "ann_ret": float(r.mean() * 12),
        "vol": float(r.std() * np.sqrt(12)),
        "sharpe": float(r.mean() / (r.std() + 1e-10) * np.sqrt(12)),
        "mdd": mdd,
        "n_months": len(r),
    }
