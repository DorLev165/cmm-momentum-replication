"""
Characteristic-Managed Momentum (CMM) Model

Replaces equal-weighting of past returns with a flexible weighting scheme
learned via a feed-forward neural network. Uses firm characteristics z_{i,t}
and daily returns r_{i,t-252:t-22} to predict next-month returns r_{i,t+1}.
"""

import numpy as np
from typing import Optional

from sklearn.preprocessing import StandardScaler

from cmm.data import N_CHARACTERISTICS, N_DAILY_RETURNS
from cmm.ffn import CMMFFNWrapper


class CMMModel:
    """
    Characteristic-Managed Momentum model using a feed-forward neural network.

    FFN outputs scalar z; Softmax weighting: Score_d = z*r_d, w = softmax(Score),
    E_CMM = sum(w*r). Weights are non-negative, sum to 1. Trained so E_CMM
    predicts next-month returns.

    When `n_ensembles` > 1, trains N independent FFNs with different random
    seeds and averages their E_CMM predictions (and softmax weights). This
    reduces per-seed variance substantially — typically the single biggest
    lift in replicating published neural-net trading strategies.
    """

    def __init__(
        self,
        n_char: int = N_CHARACTERISTICS,
        n_ret: int = N_DAILY_RETURNS,
        scale_features: bool = True,
        hidden_sizes: tuple[int, int, int] = (256, 128, 64),
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        n_ensembles: int = 5,
        verbose: bool = True,
        **model_kwargs,
    ):
        """
        Parameters
        ----------
        n_ensembles : int
            Number of independent FFN seeds to train and average (default 5).
            Each seed uses random_state + i.
        verbose : bool
            Print per-seed progress if True.
        **model_kwargs
            Passed through to CMMFFNWrapper (dropout, layer_norm,
            weight_decay, lr_schedule, warmup_epochs, grad_clip_norm,
            early_stopping_patience, min_epochs, loss_fn, ...).
        """
        self.n_char = n_char
        self.n_ret = n_ret
        self.scale_features = scale_features
        self.n_ensembles = max(1, int(n_ensembles))
        self.random_state = random_state
        self.verbose = verbose
        self.model_kwargs = {
            "hidden_sizes": hidden_sizes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            **model_kwargs,
        }

        self._models: list[CMMFFNWrapper] = []
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False

    def _scale_chars_only(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """
        Scale ONLY the first n_char columns (characteristics).
        Returns columns [n_char:] are passed through unchanged so that raw
        daily log returns feed directly into the softmax: Score = z * r_raw.
        See CMM_REPLICATION_ISSUES.md §4.
        """
        X = X.astype(np.float64, copy=True)
        if self.scale_features:
            chars = X[:, : self.n_char]
            if fit:
                self._scaler = StandardScaler()
                X[:, : self.n_char] = self._scaler.fit_transform(chars)
            else:
                X[:, : self.n_char] = self._scaler.transform(chars)
        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CMMModel":
        """
        Fit the CMM ensemble. Trains `n_ensembles` independent FFNs with
        different seeds and stores them. Predictions are averaged.
        """
        if not self.scale_features:
            self._scaler = None

        X_scaled = self._scale_chars_only(X, fit=True)
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scale_chars_only(X_val, fit=False)
        else:
            X_val_scaled = None

        self._models = []
        for i in range(self.n_ensembles):
            seed = self.random_state + i
            if self.verbose and self.n_ensembles > 1:
                print(f"    Seed {i + 1}/{self.n_ensembles} (seed={seed})...")
            m = CMMFFNWrapper(
                n_char=self.n_char,
                n_ret=self.n_ret,
                scale_features=False,
                random_state=seed,
                **self.model_kwargs,
            )
            if X_val_scaled is not None:
                m.fit(X_scaled, y, X_val=X_val_scaled, y_val=y_val)
            else:
                m.fit(X_scaled, y)
            self._models.append(m)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ensemble-averaged E_CMM. Averaging the CMM signal (rather
        than averaging the FFN outputs z or the softmax weights) is what
        matters for the portfolio sort, since the decile assignment uses
        E_CMM directly.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._scale_chars_only(X, fit=False)
        preds = np.stack([m.predict(X) for m in self._models], axis=0)
        return preds.mean(axis=0)

    def get_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble-averaged softmax weights over 231 lags.
        Shape (n_samples, n_ret). Averaging is done in weight space so
        the result is still a valid distribution (sums to 1, non-negative).
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = self._scale_chars_only(X, fit=False)
        ws = np.stack([m.predict_weights(X) for m in self._models], axis=0)
        return ws.mean(axis=0)


class CMMRegimeModel:
    """
    Regime-conditional CMM ensemble (option C).

    Trains two independent `CMMModel` instances on disjoint regime subsets
    of the training data (calm months vs stress months, split at the
    in-sample median of `regime_signal`). At prediction time, blends the
    two models' E_CMM outputs with a sigmoid weight based on the current
    regime signal.

    Rationale: a single global FFN has to fit both bull and crash
    dynamics with one ẑ function. Specialist models within each regime
    can capture regime-specific cross-sectional relationships that a
    global model cannot.

    Parameters
    ----------
    blend_sharpness : float
        Steepness of the sigmoid blend. Larger = harder switch between
        regimes. With regime_signal in [-0.5, 0.5], sharpness=5 gives a
        fairly soft blend.
    **cmm_kwargs
        Passed through to both underlying CMMModel instances.
    """

    def __init__(
        self,
        blend_sharpness: float = 5.0,
        verbose: bool = True,
        **cmm_kwargs,
    ):
        self.blend_sharpness = float(blend_sharpness)
        self.verbose = verbose
        self._cmm_kwargs = cmm_kwargs
        # Two sub-models created lazily in fit() so fresh seeds are used
        self._calm: Optional[CMMModel] = None
        self._stress: Optional[CMMModel] = None
        self._regime_median: Optional[float] = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_signal: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        regime_signal_val: Optional[np.ndarray] = None,
        min_regime_samples: int = 100,
    ) -> "CMMRegimeModel":
        """
        Fit calm & stress sub-models. Splits train (and optionally val)
        by the in-sample median of `regime_signal`.

        If either regime has fewer than `min_regime_samples` rows (degenerate
        signal, e.g. all zeros from the yfinance loader), falls back to a
        single CMMModel trained on all data — verbose warning printed.
        """
        regime_signal = np.asarray(regime_signal, dtype=np.float64)

        # FAIL LOUD on a degenerate regime signal. Silent fallback to a
        # single model would produce a "completed" run that looks normal
        # but is missing the regime-specialization (option C) the user
        # explicitly asked for. Per data-integrity policy, we refuse.
        if np.nanstd(regime_signal) < 1e-9:
            raise ValueError(
                "\n\n"
                "[BLOCKER] CMMRegimeModel received a near-constant regime_signal\n"
                "    (std < 1e-9). This would silently collapse option C\n"
                "    (regime-conditional ensemble) into a single global model.\n\n"
                "    Likely causes + fixes:\n"
                "    1. SPX cache missing & yfinance hung → run:\n"
                "           python scripts/cache_spx_from_wrds.py\n"
                "    2. Using yfinance data source (DATA_SOURCE != 'jkp') →\n"
                "       disable the regime ensemble explicitly:\n"
                "           set CMM_REGIME_ENSEMBLE=0\n"
                "    3. regime_signal pipeline is broken — debug `spx_trend_feature`.\n"
            )

        self._regime_median = float(np.nanmedian(regime_signal))

        calm_mask = regime_signal >= self._regime_median
        stress_mask = ~calm_mask

        # FAIL LOUD on unbalanced split. Same reasoning: silently shrinking
        # to a single model hides that the user's setup is wrong.
        if calm_mask.sum() < min_regime_samples or stress_mask.sum() < min_regime_samples:
            raise ValueError(
                f"\n\n"
                f"[BLOCKER] regime split produced an undersized group: "
                f"calm={calm_mask.sum()}, stress={stress_mask.sum()}, "
                f"min={min_regime_samples}.\n"
                f"    median regime_signal = {self._regime_median:+.4f}\n\n"
                f"    Likely causes + fixes:\n"
                f"    1. Training window is dominated by one regime (e.g. a 10-year\n"
                f"       bull-market-only sample). Extend training window backwards\n"
                f"       so both regimes are represented.\n"
                f"    2. SPX cache contains wrong or truncated data — inspect\n"
                f"       data/spx/spx_monthly.csv and re-run cache_spx_from_wrds.py.\n"
            )

        if X_val is not None and regime_signal_val is not None:
            rv = np.asarray(regime_signal_val, dtype=np.float64)
            calm_val_mask = rv >= self._regime_median
            stress_val_mask = ~calm_val_mask
            Xv_c, yv_c = X_val[calm_val_mask], y_val[calm_val_mask]
            Xv_s, yv_s = X_val[stress_val_mask], y_val[stress_val_mask]
            # Guard against empty val in one regime
            if len(Xv_c) < 50:
                Xv_c, yv_c = None, None
            if len(Xv_s) < 50:
                Xv_s, yv_s = None, None
        else:
            Xv_c = yv_c = Xv_s = yv_s = None

        if self.verbose:
            print(
                f"    Regime split — calm: {calm_mask.sum():,}  "
                f"stress: {stress_mask.sum():,}  "
                f"(median regime_signal = {self._regime_median:+.3f})"
            )

        if self.verbose:
            print("    Training CALM sub-ensemble...")
        self._calm = CMMModel(verbose=self.verbose, **self._cmm_kwargs)
        self._calm.fit(X[calm_mask], y[calm_mask], X_val=Xv_c, y_val=yv_c)

        if self.verbose:
            print("    Training STRESS sub-ensemble...")
        # Shift seed so stress sub-ensemble doesn't duplicate calm's seeds.
        stress_kwargs = dict(self._cmm_kwargs)
        base_seed = stress_kwargs.get("random_state", 42)
        n_ens = stress_kwargs.get("n_ensembles", 1)
        stress_kwargs["random_state"] = base_seed + 10_000 + n_ens
        self._stress = CMMModel(verbose=self.verbose, **stress_kwargs)
        self._stress.fit(X[stress_mask], y[stress_mask], X_val=Xv_s, y_val=yv_s)

        self._fitted = True
        return self

    def _blend(self, regime_signal: np.ndarray) -> np.ndarray:
        """
        Per-row calm weight in [0, 1] based on regime_signal. When the
        signal is above median, weight calm model more; below, weight
        stress model more.
        """
        rs = np.asarray(regime_signal, dtype=np.float64) - (self._regime_median or 0.0)
        # Sigmoid: when rs >> 0 → 1 (all calm); when rs << 0 → 0 (all stress)
        return 1.0 / (1.0 + np.exp(-self.blend_sharpness * rs))

    def predict(self, X: np.ndarray, regime_signal: np.ndarray) -> np.ndarray:
        """Blend E_CMM predictions from calm and stress sub-ensembles."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        calm_pred = self._calm.predict(X)
        stress_pred = self._stress.predict(X)
        w_calm = self._blend(regime_signal)
        return w_calm * calm_pred + (1.0 - w_calm) * stress_pred

    def get_weights(self, X: np.ndarray, regime_signal: np.ndarray) -> np.ndarray:
        """Blend softmax weights. Preserves sum-to-1, non-negative."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        w_c = self._calm.get_weights(X)
        w_s = self._stress.get_weights(X)
        w_calm = self._blend(regime_signal)[:, None]
        return w_calm * w_c + (1.0 - w_calm) * w_s
