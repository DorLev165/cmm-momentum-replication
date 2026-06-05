"""
Unit tests for CMM Softmax weighting boundary conditions.

Verifies that weights satisfy:
- sum(weights) == 1.0  (probability distribution)
- all weights >= 0     (non-negative)

These are strict boundary conditions required for economic interpretability.
"""

import numpy as np
import pytest

try:
    import torch
    from cmm.ffn import CMMModule
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


@pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch required")
class TestSoftmaxWeightsBoundaryConditions:
    """Test that softmax weights satisfy sum(w)==1 and w>=0."""

    def test_weights_sum_to_one(self):
        """Weights must sum to exactly 1.0 for each stock-month."""
        n_char, n_ret = 20, 231
        module = CMMModule(input_dim=n_char + n_ret, n_ret=n_ret)

        # Random inputs (various scales)
        for _ in range(5):
            x = torch.randn(32, n_char + n_ret) * 0.1
            _, weights = module(x, return_weights=True)

            sums = weights.sum(dim=-1)
            np.testing.assert_allclose(
                sums.detach().cpu().numpy(),
                np.ones(32),
                rtol=1e-5,
                err_msg="Weights must sum to 1.0",
            )

    def test_weights_non_negative(self):
        """All weights must be >= 0."""
        n_char, n_ret = 20, 231
        module = CMMModule(input_dim=n_char + n_ret, n_ret=n_ret)

        # Include edge cases: large/small scores
        x = torch.randn(16, n_char + n_ret) * 0.5
        _, weights = module(x, return_weights=True)

        assert (weights >= 0).all(), "All weights must be non-negative"
        np.testing.assert_array_less(
            -1e-9,
            weights.detach().cpu().numpy(),
            err_msg="Weights must be >= 0 (no negative values)",
        )

    def test_boundary_conditions_combined(self):
        """Strict boundary: sum(w)==1.0 and w>=0 for economic interpretability."""
        n_char, n_ret = 20, 231
        module = CMMModule(input_dim=n_char + n_ret, n_ret=n_ret)

        x = torch.randn(100, n_char + n_ret) * 0.01
        _, weights = module(x, return_weights=True)
        w = weights.detach().cpu().numpy()

        # Boundary condition 1: non-negative
        assert np.all(w >= -1e-10), "All weights must be >= 0"

        # Boundary condition 2: sum to 1
        sums = w.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, rtol=1e-6, err_msg="sum(weights) must equal 1.0")

    def test_extreme_inputs_still_satisfy_boundaries(self):
        """Boundary conditions hold even with extreme z and return values."""
        n_char, n_ret = 20, 231
        module = CMMModule(input_dim=n_char + n_ret, n_ret=n_ret)

        # Extreme: large daily returns
        x = torch.randn(8, n_char + n_ret) * 0.5
        x[:, -n_ret:] = torch.randn(8, n_ret) * 0.1  # returns
        _, weights = module(x, return_weights=True)
        w = weights.detach().cpu().numpy()

        assert np.all(w >= -1e-10)
        np.testing.assert_allclose(w.sum(axis=-1), 1.0, rtol=1e-5)
