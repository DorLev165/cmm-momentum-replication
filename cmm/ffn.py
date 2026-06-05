"""
Feed-Forward Neural Network for CMM with Softmax Weighting Mechanism.

Architecture:
- 3 hidden layers with GELU activation
- Linear output -> scalar z (scaling parameter)
- Softmax weighting: Score_d = z * r_d, w = softmax(Score), E_CMM = sum(w * r)
"""

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from sklearn.preprocessing import StandardScaler


if _HAS_TORCH:

    class CMMFFN(nn.Module):
        """
        FFN producing scalar z. Used by CMMModule for the full softmax pipeline.

        Optional regularization:
        - dropout (default 0.15): applied after each GELU
        - layer_norm (default True): stabilizes training when char inputs
          have small range, e.g. JKP's [-0.5, 0.5] rank transform

        The output layer gets a scaled init (output_init_scale, default 1.0)
        to control the starting magnitude of ẑ. Setting >1 encourages the
        softmax to start slightly peaked rather than uniform.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_sizes: tuple[int, int, int] = (256, 128, 64),
            dropout: float = 0.15,
            layer_norm: bool = True,
            output_init_scale: float = 1.0,
        ):
            super().__init__()
            self.input_dim = input_dim
            h1, h2, h3 = hidden_sizes

            def block(in_d, out_d):
                layers: list[nn.Module] = [nn.Linear(in_d, out_d)]
                if layer_norm:
                    layers.append(nn.LayerNorm(out_d))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                return layers

            modules: list[nn.Module] = []
            modules.extend(block(input_dim, h1))
            modules.extend(block(h1, h2))
            modules.extend(block(h2, h3))
            self.layers = nn.Sequential(*modules)
            self.output_layer = nn.Linear(h3, 1)

            # Kaiming init for GELU is close to 'linear', and PyTorch's default
            # is already Kaiming-uniform with a=sqrt(5). Scale the OUTPUT layer
            # to control the initial magnitude of ẑ.
            with torch.no_grad():
                self.output_layer.weight.mul_(output_init_scale)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns scalar z per sample."""
            h = self.layers(x)
            return self.output_layer(h).squeeze(-1)

    class CMMModule(nn.Module):
        """
        Full CMM pipeline: FFN -> z -> Softmax weighting -> E_CMM.

        - Importance score: Score_{t-d} = z * r_{t-d}
        - Weights: w = softmax(Score)  (non-negative, sum=1)
        - CMM signal: E_CMM = sum(w * r)
        """

        def __init__(
            self,
            input_dim: int,
            n_ret: int,
            hidden_sizes: tuple[int, int, int] = (256, 128, 64),
            dropout: float = 0.15,
            layer_norm: bool = True,
            output_init_scale: float = 1.0,
        ):
            super().__init__()
            self.n_ret = n_ret
            self.n_char = input_dim - n_ret
            self.ffn = CMMFFN(
                input_dim,
                hidden_sizes,
                dropout=dropout,
                layer_norm=layer_norm,
                output_init_scale=output_init_scale,
            )

        def forward(
            self, x: torch.Tensor, return_z: bool = False, return_weights: bool = False
        ):
            """
            Forward pass. Returns E_CMM (CMM momentum signal) per sample.

            x: (batch, n_char + n_ret) - [characteristics, daily_returns]
            """
            z = self.ffn(x)  # (batch,)
            r = x[:, -self.n_ret :]  # (batch, 231) - daily returns

            # Importance score: Score_{t-d} = z_i,t * r_{i,t-d}
            scores = z.unsqueeze(-1) * r  # (batch, 231)

            # Softmax: w = exp(Score) / sum(exp(Score))
            # Boundary conditions: w >= 0, sum(w) == 1 (required for economic interpretability)
            weights = torch.softmax(scores, dim=-1)  # (batch, 231)

            # E_CMM = sum(w * r)
            e_cmm = (weights * r).sum(dim=-1)  # (batch,)

            if return_weights:
                return e_cmm, weights
            if return_z:
                return e_cmm, z
            return e_cmm

else:
    CMMFFN = None  # type: ignore
    CMMModule = None  # type: ignore


class CMMFFNWrapper:
    """
    Wrapper that provides fit/predict interface matching CMMModel.
    """

    def __init__(
        self,
        n_char: int,
        n_ret: int,
        hidden_sizes: tuple[int, int, int] = (256, 128, 64),
        scale_features: bool = True,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
        device: Optional[str] = None,
        # Regularization / architecture
        dropout: float = 0.15,
        layer_norm: bool = True,
        output_init_scale: float = 1.0,
        weight_decay: float = 1e-5,
        # Training dynamics
        grad_clip_norm: float = 1.0,
        lr_schedule: str = "cosine",      # "cosine", "plateau", or "none"
        warmup_epochs: int = 3,
        # Early stopping
        early_stopping_patience: int = 10,
        min_epochs: int = 15,
        # Loss
        loss_fn: str = "mse",              # "mse" or "huber"
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.n_char = n_char
        self.n_ret = n_ret
        self.input_dim = n_char + n_ret
        self.hidden_sizes = hidden_sizes
        self.scale_features = scale_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.dropout = dropout
        self.layer_norm = layer_norm
        self.output_init_scale = output_init_scale
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.early_stopping_patience = early_stopping_patience
        self.min_epochs = min_epochs
        self.loss_fn = loss_fn

        self._model: Optional["CMMModule"] = None
        self._scaler: Optional[StandardScaler] = None
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CMMFFNWrapper":
        """
        Fit the CMM model with regularization, LR scheduling, and early
        stopping. Trains FFN so E_CMM (softmax-weighted momentum) predicts
        next-month returns.

        Training features (all tunable via constructor):
        - Weight decay (AdamW, L2 regularization)
        - Gradient clipping
        - LR schedule: cosine / plateau / none, with optional warmup
        - Huber loss alternative (robust to return outliers)
        - Early stopping with configurable patience + minimum epochs
        """
        if self.scale_features:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
            if X_val is not None:
                X_val = self._scaler.transform(X_val)
        else:
            self._scaler = None

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._model = CMMModule(
            self.input_dim,
            self.n_ret,
            self.hidden_sizes,
            dropout=self.dropout,
            layer_norm=self.layer_norm,
            output_init_scale=self.output_init_scale,
        ).to(self._device)

        X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(y, dtype=torch.float32, device=self._device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv_t = torch.tensor(X_val, dtype=torch.float32, device=self._device)
            yv_t = torch.tensor(y_val, dtype=torch.float32, device=self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        if self.loss_fn == "huber":
            criterion = nn.SmoothL1Loss(beta=0.5)
        else:
            criterion = nn.MSELoss()

        # LR scheduler
        if self.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.epochs - self.warmup_epochs)
            )
        elif self.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3
            )
        else:
            scheduler = None

        best_val = float("inf")
        patience_left = self.early_stopping_patience
        best_state = None

        for epoch in range(self.epochs):
            # LR warmup: linearly ramp from 0 to target LR over `warmup_epochs`
            if epoch < self.warmup_epochs:
                warm_lr = self.learning_rate * (epoch + 1) / max(1, self.warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = warm_lr

            self._model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                e_cmm = self._model(xb)
                loss = criterion(e_cmm, yb)
                loss.backward()
                if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self.grad_clip_norm
                    )
                optimizer.step()

            # Step LR scheduler (cosine steps per-epoch after warmup)
            if scheduler is not None and epoch >= self.warmup_epochs:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    pass  # stepped below with val loss
                else:
                    scheduler.step()

            # Validation + early stopping
            if use_val:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(Xv_t)
                    val_loss = float(criterion(val_pred, yv_t).item())

                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)

                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_left = self.early_stopping_patience
                    best_state = {
                        k: v.detach().clone() for k, v in self._model.state_dict().items()
                    }
                else:
                    patience_left -= 1
                    # Only stop after hitting minimum epoch count, so warmup +
                    # early low-LR phase doesn't trigger premature exit.
                    if patience_left <= 0 and epoch >= self.min_epochs:
                        break

        if use_val and best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict E_CMM (CMM momentum signal) for each firm-month."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._scaler is not None:
            X = self._scaler.transform(X)

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            e_cmm = self._model(X_t).cpu().numpy()
        return e_cmm

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        """Return softmax weights over the 231 lags for each sample. Shape (n_samples, n_ret)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if self._scaler is not None:
            X = self._scaler.transform(X)

        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self._device)
            _, weights = self._model(X_t, return_weights=True)
        return weights.cpu().numpy()
