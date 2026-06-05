"""
Covariance estimators for portfolio construction.

Primary estimator: Ledoit-Wolf linear shrinkage toward a constant-correlation
target (Ledoit & Wolf 2004, "Honey, I Shrunk the Sample Covariance Matrix").
This is the benchmark estimator for equity portfolios — stocks share strong
systematic correlation via the market factor, so shrinking toward constant
correlation (rather than identity, as sklearn's default does) is substantially
better.

Also provides:
- `sample_covariance` — plain unshrunk, for comparison/debug
- `factor_model_covariance` — PCA-based statistical factor model, for
  very high-dimensional settings where LW is too slow

References
----------
Ledoit, O., & Wolf, M. (2004). "Honey, I shrunk the sample covariance
matrix." Journal of Portfolio Management, 30(4), 110-119.
"""

from __future__ import annotations

import numpy as np


def _to_returns(returns: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Sanitize input and return centered returns + (T, N)."""
    R = np.asarray(returns, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N); got shape {R.shape}")
    T, N = R.shape
    if T < 2 or N < 1:
        raise ValueError(f"need T>=2, N>=1; got T={T}, N={N}")
    Xc = R - R.mean(axis=0, keepdims=True)
    return Xc, T, N


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    """Plain sample covariance: X'X/T after demeaning. Singular when N > T."""
    Xc, T, _ = _to_returns(returns)
    return (Xc.T @ Xc) / T


def _constant_correlation_target(sample_cov: np.ndarray) -> np.ndarray:
    """
    Build the constant-correlation shrinkage target F:
        F_ii = sigma_ii (sample variance)
        F_ij = r_bar * sqrt(sigma_ii * sigma_jj),  i != j
    where r_bar is the average sample correlation across all off-diagonal
    pairs. Matches Ledoit & Wolf (2004) equation 2.6.
    """
    S = sample_cov
    N = S.shape[0]
    var = np.diag(S).copy()
    vol = np.sqrt(np.maximum(var, 1e-20))
    # Sample correlation matrix
    C = S / np.outer(vol, vol)
    # Off-diagonal mean correlation
    mask = ~np.eye(N, dtype=bool)
    r_bar = C[mask].mean() if N > 1 else 0.0

    F = r_bar * np.outer(vol, vol)
    np.fill_diagonal(F, var)
    return F, float(r_bar)


def ledoit_wolf_constant_corr(
    returns: np.ndarray,
    min_eigvalue: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """
    Ledoit-Wolf linear shrinkage with constant-correlation target.

    Returns
    -------
    Sigma : np.ndarray (N, N)
        Shrunk covariance, positive definite.
    delta : float
        Applied shrinkage intensity in [0, 1]. delta=0 means pure sample
        covariance; delta=1 means pure target F. Computed analytically to
        minimize expected Frobenius distance from the true covariance.

    Notes
    -----
    Implementation follows LW 2004 eq. 2.9 exactly. The core insight is
    that optimal delta = pi_hat - rho_hat / gamma_hat, clipped to [0, 1]:
      pi_hat    — sum of asymptotic variances of sample covariance entries
      rho_hat   — sum of asymptotic covariances between sample cov and target
      gamma_hat — ||S - F||^2 (Frobenius, measures disagreement)
    """
    Xc, T, N = _to_returns(returns)
    S = (Xc.T @ Xc) / T
    F, r_bar = _constant_correlation_target(S)

    # pi_hat: Σ_ij Var(s_ij) using finite-sample approximation from LW 2004
    # (2.3): pi_hat_ij = (1/T) Σ_t (X_ti*X_tj - s_ij)^2
    Xc2 = Xc ** 2
    pi_mat = (Xc2.T @ Xc2) / T - S ** 2
    pi_hat = pi_mat.sum()

    # rho_hat: sum of asymptotic covariances. Has three terms (LW 2004 Section 3.2).
    # Simpler decomposition: rho_hat = Σ_i pi_ii + Σ_{i != j} term_ij
    # where term_ij uses the constant-correlation structure.
    vol = np.sqrt(np.diag(S))
    # Helper: (1/T) Σ_t X_ti^2 * X_ti*X_tj  = E[X_i^3 X_j]
    theta_ii_ij = np.zeros((N, N))  # theta[i,j] = (1/T) Σ_t X_ti^2 * X_ti*X_tj - s_ii*s_ij
    theta_jj_ij = np.zeros((N, N))  # similarly with j^2
    for i in range(N):
        Xi2 = Xc[:, i] ** 2
        theta_ii_ij[i, :] = (Xi2 @ (Xc * Xc[:, [i]])) / T - S[i, i] * S[i, :]
    theta_jj_ij = theta_ii_ij.T  # by symmetry in indexing

    rho_diag = np.diag(pi_mat).sum()
    off = ~np.eye(N, dtype=bool)
    if N > 1:
        # LW 2004 (3.3): off-diagonal contribution
        # rho_off_{ij} = (r_bar/2) * (sqrt(s_jj/s_ii) * theta_ii_ij + sqrt(s_ii/s_jj) * theta_jj_ij)
        ratio = np.outer(vol, 1.0 / np.maximum(vol, 1e-20))  # sqrt(s_jj/s_ii) = vol[j]/vol[i]
        # But we want sqrt(s_jj/s_ii) so ratio_ij = vol_j / vol_i
        ratio = vol[np.newaxis, :] / np.maximum(vol[:, np.newaxis], 1e-20)
        rho_off_term = (r_bar / 2.0) * (ratio * theta_ii_ij + (1.0 / np.maximum(ratio, 1e-20)) * theta_jj_ij)
        rho_off = rho_off_term[off].sum()
    else:
        rho_off = 0.0
    rho_hat = rho_diag + rho_off

    # gamma_hat: ||S - F||^2 (Frobenius, squared)
    diff = S - F
    gamma_hat = float((diff ** 2).sum())

    # Optimal shrinkage intensity
    if gamma_hat <= 0:
        delta = 1.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(np.clip(kappa / T, 0.0, 1.0))

    Sigma = delta * F + (1.0 - delta) * S

    # Enforce PSD via eigenvalue flooring (numerical safety)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (Sigma + Sigma.T))
    eigvals = np.maximum(eigvals, min_eigvalue)
    Sigma = (eigvecs * eigvals) @ eigvecs.T
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma, delta


def factor_model_covariance(
    returns: np.ndarray,
    n_factors: int = 15,
    min_eigvalue: float = 1e-10,
) -> np.ndarray:
    """
    Statistical factor model covariance: Sigma = B·Lambda·B' + diag(D).
    Computed via PCA on the sample covariance matrix. Keeps top `n_factors`
    eigenvalues; remaining variance becomes idiosyncratic diagonal.

    Fast for large N, and equity returns are well-approximated by a small
    number of factors (market + sectors). Sensible fallback when LW is too
    slow (e.g., N > 1000).
    """
    Xc, T, N = _to_returns(returns)
    S = (Xc.T @ Xc) / T
    k = min(n_factors, N - 1, T - 1)
    if k < 1:
        k = 1
    eigvals, eigvecs = np.linalg.eigh(S)
    # eigh returns ascending — take the top k
    top_vals = eigvals[-k:]
    top_vecs = eigvecs[:, -k:]
    # Reconstruct systematic part
    systematic = (top_vecs * top_vals) @ top_vecs.T
    # Idiosyncratic = diagonal of (S - systematic), floored
    idio_diag = np.maximum(np.diag(S) - np.diag(systematic), min_eigvalue)
    Sigma = systematic + np.diag(idio_diag)
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma
