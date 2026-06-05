"""
CMM Data Module

Handles:
- Cross-sectional normalization of next-month returns
- Feature construction: 153 firm characteristics + 231 daily log returns
- Returns r_{i,t-252:t-22} (excluding most recent month for short-term reversal)
"""

import numpy as np
import pandas as pd
from typing import Tuple

# Data dimensions
N_CHARACTERISTICS = 153
N_DAILY_RETURNS = 231  # t-252 to t-22 (1 year excluding last month)


def cross_sectional_normalize(
    returns: np.ndarray, group_labels: np.ndarray
) -> np.ndarray:
    """
    Cross-sectionally normalize returns within each month/group.

    For each month t, transforms r_{i,t+1} to have zero mean and unit std
    across stocks i. Facilitates cross-sectional comparability for training.

    Parameters
    ----------
    returns : np.ndarray
        Shape (n_obs,) - next-month realized returns
    group_labels : np.ndarray
        Shape (n_obs,) - month/date identifier for cross-sectional grouping

    Returns
    -------
    np.ndarray
        Cross-sectionally normalized returns
    """
    result = np.full_like(returns, np.nan, dtype=float)
    for g in np.unique(group_labels):
        mask = group_labels == g
        r_g = returns[mask]
        valid = np.isfinite(r_g)
        if valid.sum() > 1:
            mean_g = np.nanmean(r_g)
            std_g = np.nanstd(r_g)
            std_g = std_g if std_g > 1e-10 else 1.0
            result[mask] = (r_g - mean_g) / std_g
        elif valid.sum() == 1:
            result[mask] = 0.0  # single obs: set to 0
    return result


def prepare_cmm_data(
    characteristics: np.ndarray,
    daily_returns: np.ndarray,
    next_month_returns: np.ndarray,
    dates: np.ndarray,
    n_char: int = N_CHARACTERISTICS,
    n_ret: int = N_DAILY_RETURNS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and target for CMM.

    Parameters
    ----------
    characteristics : np.ndarray
        Shape (n_obs, 153) - firm characteristics z_{i,t}
    daily_returns : np.ndarray
        Shape (n_obs, 231) - daily log returns r_{i,t-252:t-22}
    next_month_returns : np.ndarray
        Shape (n_obs,) - realized next-month returns r_{i,t+1}
    dates : np.ndarray
        Shape (n_obs,) - month/date for cross-sectional grouping
    n_char : int
        Number of characteristics (default 153)
    n_ret : int
        Number of daily return lags (default 231)

    Returns
    -------
    X : np.ndarray
        Features [z_{i,t}, r_{i,t-252:t-22}] shape (n_obs, 153+231)
    y : np.ndarray
        Cross-sectionally normalized next-month returns
    valid_mask : np.ndarray
        Boolean mask of valid (non-NaN) rows
    """
    assert characteristics.shape[1] >= n_char
    assert daily_returns.shape[1] >= n_ret

    X_char = characteristics[:, :n_char].astype(np.float64)
    X_ret = daily_returns[:, :n_ret].astype(np.float64)

    X = np.hstack([X_char, X_ret])

    y_raw = np.asarray(next_month_returns, dtype=np.float64)
    y = cross_sectional_normalize(y_raw, dates)

    valid = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
    )
    return X, y, valid


def train_val_split_by_time(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    val_pct: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data by time: train on earlier months, validate on later months.

    Parameters
    ----------
    X, y : feature matrix and target
    dates : month labels
    val_pct : fraction of unique dates for validation (default 0.2)

    Returns
    -------
    X_train, y_train, X_val, y_val, train_dates, val_dates
    """
    uniq_dates = np.unique(dates)
    n_val = max(1, int(len(uniq_dates) * val_pct))
    val_dates = set(uniq_dates[-n_val:])
    train_mask = np.array([d not in val_dates for d in dates])
    val_mask = ~train_mask

    return (
        X[train_mask],
        y[train_mask],
        X[val_mask],
        y[val_mask],
        dates[train_mask],
        dates[val_mask],
    )
