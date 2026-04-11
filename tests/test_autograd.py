"""Tests for PyTorch autograd verification of the gradient identity.

These tests are skipped if PyTorch is not installed.
"""

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestAutograd:
    """Verify that PyTorch autograd produces the same gradient as scipy methods."""

    def test_autograd_matches_analytical(self):
        """Autograd gradient should match analytical gradient (cosine ~ 1.0)."""
        from phasegrad.autograd_verify import verify_autograd

        result = verify_autograd(N=10, seed=42)
        assert result['cos_ag_an'] > 0.999, (
            f"Autograd vs analytical cosine: {result['cos_ag_an']:.6f}")

    def test_autograd_matches_twophase(self):
        """Autograd gradient should match two-phase gradient (cosine ~ 1.0)."""
        from phasegrad.autograd_verify import verify_autograd

        result = verify_autograd(N=10, seed=42)
        assert result['cos_ag_tp'] > 0.999, (
            f"Autograd vs two-phase cosine: {result['cos_ag_tp']:.6f}")

    @pytest.mark.parametrize("N", [6, 10, 15, 20])
    def test_autograd_across_sizes(self, N):
        """Autograd verification at multiple network sizes."""
        from phasegrad.autograd_verify import verify_autograd

        result = verify_autograd(N=N, seed=42)
        assert result['cos_ag_an'] > 0.999, (
            f"N={N}: autograd vs analytical cosine: {result['cos_ag_an']:.6f}")
        assert result['cos_ag_tp'] > 0.999, (
            f"N={N}: autograd vs two-phase cosine: {result['cos_ag_tp']:.6f}")

    def test_newton_solver_converges(self):
        """The PyTorch Newton solver should find equilibrium to machine precision."""
        from phasegrad.autograd_verify import _newton_solve
        from phasegrad.kuramoto import make_random_network

        net = make_random_network(10, K_mean=5.0, seed=42)
        omega_c = torch.tensor(net.omega_centered, dtype=torch.float64)
        K = torch.tensor(net.K, dtype=torch.float64)

        theta, residual = _newton_solve(omega_c, K)
        assert residual < 1e-10, f"Newton residual: {residual:.2e}"

    def test_gradient_zero_at_target(self):
        """When output is already at target, gradient should be zero."""
        from phasegrad.autograd_verify import autograd_gradient
        from phasegrad.kuramoto import make_random_network

        net = make_random_network(8, K_mean=5.0, seed=42)
        theta_star, _ = net.equilibrium()

        # Target = equilibrium → loss = 0 → gradient = 0
        result = autograd_gradient(net.omega, net.K, net.output_ids, theta_star)
        assert np.max(np.abs(result['grad'])) < 1e-10, (
            f"Gradient should be zero at target, max = {np.max(np.abs(result['grad'])):.2e}")
