"""
CMM Training with Expanding Window.

- Loss: MSE between E_CMM signal and cross-sectionally normalized target returns
- Optimizer: Adam
- Expanding training window: train 1973-1982, test 1983-1984; shift forward 2 years
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterator, Optional

from cmm import CMMModel, prepare_cmm_data
from cmm.model import CMMRegimeModel
from cmm.portfolio import build_hml_portfolio, build_hml_portfolio_mv, hml_summary


# Default location for per-window checkpointing. Each completed window
# writes one JSONL line + per-window HML returns CSV here, so a kill mid-run
# preserves all results computed so far.
_DEFAULT_PARTIAL_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


def _save_window_result(save_dir: Path, result: dict) -> None:
    """
    Persist one window's result so a kill mid-run doesn't lose work.

    Writes:
      - <save_dir>/results_partial.jsonl  (one line per window, summary)
      - <save_dir>/hml_returns_vw/<key>.csv  (full monthly HML series, VW)
      - <save_dir>/hml_returns_mv/<key>.csv  (if MV portfolio was computed)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    win_key = (
        f"train_{result['train_start']}_{result['train_end']}"
        f"_test_{result['test_start']}_{result['test_end']}"
    )

    # Save the per-month HML returns (DataFrames don't fit in JSON)
    hml_df = result.get("hml_df")
    if hml_df is not None and len(hml_df) > 0:
        vw_dir = save_dir / "hml_returns_vw"
        vw_dir.mkdir(parents=True, exist_ok=True)
        hml_df.to_csv(vw_dir / f"{win_key}.csv", index=False)

    hml_df_mv = result.get("hml_df_mv")
    if hml_df_mv is not None and len(hml_df_mv) > 0:
        mv_dir = save_dir / "hml_returns_mv"
        mv_dir.mkdir(parents=True, exist_ok=True)
        hml_df_mv.to_csv(mv_dir / f"{win_key}.csv", index=False)

    # JSON summary record (everything except the DataFrames)
    summary = {k: v for k, v in result.items() if k not in ("hml_df", "hml_df_mv")}
    jsonl_path = save_dir / "results_partial.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def expanding_window_splits(
    train_start_year: int = 1973,
    train_end_year: int = 1982,
    test_years: int = 2,
    shift_years: int = 2,
    max_end_year: Optional[int] = None,
) -> Iterator[tuple[int, int, int, int]]:
    """
    Generate (train_start, train_end, test_start, test_end) year tuples.

    Iteration 1: train 1973-1982, test 1983-1984
    Iteration 2: train 1973-1984, test 1985-1986
    ...

    The final window is allowed to have a **partial test period**: if
    test_start ≤ max_end_year < test_end, we clip test_end to max_end_year
    so the latest available data is used even when it doesn't span a full
    `test_years` block. This lets a run continue into e.g. 2025-Q1 without
    dropping the window.

    Parameters
    ----------
    max_end_year : int, optional
        Last year of data; defaults to current year.
    """
    if max_end_year is None:
        max_end_year = pd.Timestamp.today().year

    test_start = train_end_year + 1
    test_end = test_start + test_years - 1

    while test_start <= max_end_year:
        effective_test_end = min(test_end, max_end_year)
        yield train_start_year, train_end_year, test_start, effective_test_end
        train_end_year += shift_years
        test_start += shift_years
        test_end += shift_years


def filter_by_year_range(
    timestamps: np.ndarray,
    start_year: int,
    end_year: int,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """
    Filter arrays to rows whose timestamps fall within [start_year, end_year].
    Returns (mask, *filtered_arrays).
    """
    ts = pd.to_datetime(timestamps)
    mask = (ts.year >= start_year) & (ts.year <= end_year)
    if not arrays:
        return mask
    return mask, *(a[mask] for a in arrays)


def _plot_weights_by_date(
    weights: np.ndarray,
    timestamps: np.ndarray,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    n_ret: int = 231,
    max_dates: int = 8,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """
    Plot mean softmax weights over lags for a selection of dates in the test period.
    weights: (n_samples, n_ret), timestamps: (n_samples,) same order as weights.
    Returns the path the figure was saved to, or None.
    """
    if weights.size == 0 or len(timestamps) != len(weights):
        return None
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({"date": ts}, index=range(len(ts)))
    df["period"] = df["date"].dt.to_period("M")
    # Mean weights per (year-month)
    by_period = df.groupby("period").indices
    if not by_period:
        return None
    periods = sorted(by_period.keys())
    # Select up to max_dates spread over the test range
    if len(periods) <= max_dates:
        selected = periods
    else:
        idx = np.linspace(0, len(periods) - 1, max_dates, dtype=int)
        selected = [periods[i] for i in idx]
    lag_idx = np.arange(n_ret)
    fig, ax = plt.subplots(figsize=(10, 4))
    for p in selected:
        rows = by_period[p]
        w = weights[rows].mean(axis=0)
        label = str(p)
        ax.plot(lag_idx, w, label=label, alpha=0.8)
    ax.set_xlabel("Lag (0 = most recent)")
    ax.set_ylabel("Mean weight")
    ax.set_title(f"Train {train_start}-{train_end}, test {test_start}-{test_end}: weights by date")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    plt.close()
    return None


def train_val_split_for_window(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    val_pct: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into train/val with the last val_pct by time.

    If timestamps are provided, splits by unique month-end so the last
    val_pct of months go to validation (no leakage across months).
    Falls back to row-order split if timestamps is None — but this is
    only safe when rows are already sorted by date.
    See CMM_REPLICATION_ISSUES.md §8.3.
    """
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        uniq = np.array(sorted(pd.unique(ts)))
        n_val = max(1, int(len(uniq) * val_pct))
        val_dates = set(uniq[-n_val:])
        val_mask = np.array([d in val_dates for d in ts])
        train_mask = ~val_mask
        return X[train_mask], y[train_mask], X[val_mask], y[val_mask]

    n_val = max(1, int(len(X) * val_pct))
    n_train = len(X) - n_val
    return (
        X[:n_train],
        y[:n_train],
        X[n_train:],
        y[n_train:],
    )


def run_expanding_window(
    characteristics: np.ndarray,
    daily_returns: np.ndarray,
    next_month_returns: np.ndarray,
    dates: np.ndarray,
    timestamps: np.ndarray,
    tickers: np.ndarray,
    market_cap: np.ndarray,
    is_nyse: Optional[np.ndarray] = None,
    regime_signal: Optional[np.ndarray] = None,
    industries: Optional[np.ndarray] = None,
    n_char: int = 11,
    n_ret: int = 231,
    train_start_year: int = 1973,
    initial_train_end_year: int = 1982,
    val_pct: float = 0.2,
    use_nyse_breakpoints: bool = True,
    use_regime_ensemble: bool = False,
    use_mv_portfolio: bool = False,
    use_industry_adjust: bool = False,
    weights_plot_dir: Optional[str] = "plots",
    partial_save_dir: Optional[str] = None,
    **model_kwargs,
) -> list[dict]:
    """
    Run CMM with expanding window. Returns list of results per test period.

    Loss: MSE(E_CMM, y) where y is cross-sectionally normalized target returns.
    Optimizer: Adam.
    Portfolio: deciles by NYSE breakpoints, value-weight, HML explicitly dollar-neutral (long +1, short -1).

    Each result dict: {train_start, train_end, test_start, test_end, ic,
                      hml_df, hml_stats, weights_plot_path, ...}

    weights_plot_dir : str or None
        Directory to save weight-by-date plots per window (default "plots").
        Created if it does not exist. If None, weight plots are not saved.
    """
    X, y, valid = prepare_cmm_data(
        characteristics, daily_returns, next_month_returns, dates,
        n_char=n_char, n_ret=n_ret,
    )
    X, y = X[valid], y[valid]
    timestamps = timestamps[valid]
    tickers = tickers[valid]
    market_cap = market_cap[valid]
    raw_returns = next_month_returns[valid]
    if is_nyse is not None:
        is_nyse = np.asarray(is_nyse, dtype=bool)[valid]
    if regime_signal is not None:
        regime_signal = np.asarray(regime_signal, dtype=np.float64)[valid]
    if industries is not None:
        industries = np.asarray(industries, dtype=np.int32)[valid]
    industries_effective = industries if use_industry_adjust else None
    # Per-row daily return vectors filtered similarly — needed for MV portfolio
    daily_returns_filtered = daily_returns[valid] if use_mv_portfolio else None

    # NYSE breakpoints now come from the per-row is_nyse array (§6).
    # If `use_nyse_breakpoints=False`, pass None to use the full universe.
    is_nyse_effective = is_nyse if use_nyse_breakpoints else None

    # Clear the partial-results file at the start of each fresh run so we
    # don't append to a previous run's results. Per-window CSVs are
    # overwritten in place since their filenames are deterministic.
    if partial_save_dir is not None:
        partial_path = Path(partial_save_dir) / "results_partial.jsonl"
        if partial_path.exists():
            try:
                partial_path.unlink()
                print(f"  [partial-save] cleared previous {partial_path}")
            except Exception as e:
                print(f"  [partial-save] could not clear previous file: {e}")
        print(f"  [partial-save] writing per-window results to {partial_save_dir}")

    ts = pd.to_datetime(timestamps)
    min_year = int(ts.year.min())
    max_year = int(ts.year.max())

    # Adapt to available data: need train years + at least 1 test year.
    # Final test window may be partial (clipped to max_year) — that's OK.
    if min_year > train_start_year:
        train_start_year = min_year
        initial_train_end_year = min_year + 9
    if initial_train_end_year + 1 > max_year:
        initial_train_end_year = max_year - 1  # allow at least 1y of test
    if train_start_year > initial_train_end_year or initial_train_end_year + 1 > max_year:
        return []

    # Count windows for progress (partial final window is included).
    window_list = list(expanding_window_splits(
        train_start_year=train_start_year,
        train_end_year=initial_train_end_year,
        test_years=2,
        shift_years=2,
        max_end_year=max_year,
    ))
    # Filter out any purely in-the-future windows (shouldn't happen with
    # the clipped iterator, but kept as a safety net).
    window_list = [w for w in window_list if w[0] >= min_year and w[2] <= max_year]
    n_windows = len(window_list)
    print(f"  Data year range: {min_year}-{max_year}  |  {n_windows} expanding windows:")
    for (ts_, te_, xs_, xe_) in window_list:
        partial = " (partial)" if (xe_ - xs_ + 1) < 2 else ""
        print(f"    train {ts_}-{te_} / test {xs_}-{xe_}{partial}")

    results = []
    for win_idx, (train_start, train_end, test_start, test_end) in enumerate(window_list):

        train_mask = (pd.to_datetime(timestamps).year >= train_start) & (
            pd.to_datetime(timestamps).year <= train_end
        )
        test_mask = (pd.to_datetime(timestamps).year >= test_start) & (
            pd.to_datetime(timestamps).year <= test_end
        )
        X_train_full = X[train_mask]
        y_train_full = y[train_mask]
        ts_train_full = timestamps[train_mask]
        rs_train_full = regime_signal[train_mask] if regime_signal is not None else None

        X_test = X[test_mask]
        y_test = y[test_mask]
        raw_test = raw_returns[test_mask]
        mcap_test = market_cap[test_mask]
        tickers_test = tickers[test_mask]
        ts_test = timestamps[test_mask]
        nyse_test = is_nyse_effective[test_mask] if is_nyse_effective is not None else None
        rs_test = regime_signal[test_mask] if regime_signal is not None else None
        dr_test = daily_returns_filtered[test_mask] if daily_returns_filtered is not None else None
        ind_test = industries_effective[test_mask] if industries_effective is not None else None

        if len(X_train_full) < 100 or len(X_test) < 10:
            continue

        current = len(results) + 1
        model_kind = "regime-conditional" if use_regime_ensemble else "single"
        print(
            f"  Window {current}/{n_windows}: train {train_start}-{train_end}, "
            f"test {test_start}-{test_end} ({model_kind} ensemble)"
        )

        # Time-based val split across all aligned arrays
        X_train, y_train, X_val, y_val = train_val_split_for_window(
            X_train_full, y_train_full, timestamps=ts_train_full, val_pct=val_pct
        )
        if rs_train_full is not None:
            rs_train, _, rs_val, _ = train_val_split_for_window(
                rs_train_full, y_train_full, timestamps=ts_train_full, val_pct=val_pct
            )
        else:
            rs_train = rs_val = None

        if use_regime_ensemble:
            if rs_train is None:
                raise ValueError(
                    "use_regime_ensemble=True requires regime_signal to be provided."
                )
            model = CMMRegimeModel(n_char=n_char, n_ret=n_ret, **model_kwargs)
            model.fit(
                X_train, y_train, rs_train,
                X_val=X_val, y_val=y_val, regime_signal_val=rs_val,
            )
            pred_test = model.predict(X_test, rs_test)
            weights_test = model.get_weights(X_test, rs_test)
        else:
            model = CMMModel(n_char=n_char, n_ret=n_ret, **model_kwargs)
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            pred_test = model.predict(X_test)
            weights_test = model.get_weights(X_test)

        ic = float(np.corrcoef(pred_test, y_test)[0, 1]) if len(y_test) > 1 else 0.0
        if weights_plot_dir is not None:
            os.makedirs(weights_plot_dir, exist_ok=True)
            save_path = os.path.join(
                weights_plot_dir,
                f"weights_train_{train_start}_{train_end}_test_{test_start}_{test_end}.png",
            )
        else:
            save_path = None
        weights_plot_path = _plot_weights_by_date(
            weights_test,
            ts_test,
            train_start,
            train_end,
            test_start,
            test_end,
            n_ret=n_ret,
            max_dates=8,
            save_path=save_path,
        )
        if weights_plot_path:
            print(f"    Weights plot: {weights_plot_path}")

        # Value-weighted (paper-faithful) portfolio — always computed
        hml_df = build_hml_portfolio(
            signal=pred_test,
            next_returns=raw_test,
            market_cap=mcap_test,
            tickers=tickers_test,
            timestamps=ts_test,
            is_nyse=nyse_test,
            industries=ind_test,
        )
        hml_stats = hml_summary(hml_df["HML_ret"]) if len(hml_df) > 0 else {}

        # Mean-variance (covariance-aware) portfolio — opt-in
        hml_df_mv = None
        hml_stats_mv = {}
        if use_mv_portfolio and dr_test is not None:
            hml_df_mv = build_hml_portfolio_mv(
                signal=pred_test,
                next_returns=raw_test,
                market_cap=mcap_test,
                tickers=tickers_test,
                timestamps=ts_test,
                daily_returns=dr_test,
                is_nyse=nyse_test,
                industries=ind_test,
                cov_lookback_days=231,
            )
            hml_stats_mv = (
                hml_summary(hml_df_mv["HML_ret"]) if len(hml_df_mv) > 0 else {}
            )

        result = {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "ic": ic,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "hml_df": hml_df,
            "hml_stats": hml_stats,
            "hml_df_mv": hml_df_mv,
            "hml_stats_mv": hml_stats_mv,
            "weights_plot_path": weights_plot_path,
        }
        results.append(result)

        # Persist this window so a kill mid-run preserves work.
        if partial_save_dir is not None:
            try:
                _save_window_result(Path(partial_save_dir), result)
            except Exception as e:
                print(f"    [warn] partial-save failed for this window: {e}")

    return results
