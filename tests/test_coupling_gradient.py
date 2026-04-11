"""Tests for coupling-strength gradient (grad_K).

The two-phase coupling gradient is:
    ∂L/∂K_ij ≈ [cos(θ_j* - θ_i*) - cos(θ_j^β - θ_i^β)] / β

These tests verify grad_K against finite-difference perturbation of each
coupling weight, closing the validation gap identified in the audit
(omega gradient was well-tested; coupling gradient was not).
"""

import numpy as np
import pytest

from phasegrad.kuramoto import make_random_network
from phasegrad.gradient import two_phase_gradient
from phasegrad.losses import mse_loss, mse_target


def _finite_difference_K_gradient(net, theta_star, target, eps=1e-5):
    """Compute ∂L/∂K by central finite differences on each edge.

    Perturbs K_ij (and K_ji for symmetry) by ±eps, re-solves equilibrium,
    and measures the change in loss.
    """
    grad_K = np.zeros_like(net.K)
    for (i, j) in net.edges:
        # +eps
        net.K[i, j] += eps
        net.K[j, i] += eps
        theta_plus, _ = net.equilibrium(theta_init=theta_star.copy())
        L_plus = mse_loss(theta_plus, target, net.output_ids)

        # -2eps (back to -eps from original)
        net.K[i, j] -= 2 * eps
        net.K[j, i] -= 2 * eps
        theta_minus, _ = net.equilibrium(theta_init=theta_star.copy())
        L_minus = mse_loss(theta_minus, target, net.output_ids)

        # Restore
        net.K[i, j] += eps
        net.K[j, i] += eps

        grad_K[i, j] = (L_plus - L_minus) / (2 * eps)
        grad_K[j, i] = grad_K[i, j]

    return grad_K


class TestCouplingGradient:
    """Verify that the two-phase coupling gradient matches finite differences."""

    def test_coupling_gradient_direction(self):
        """Coupling gradient should agree with FD in direction (cosine ~ 1)."""
        net = make_random_network(N=8, K_mean=5.0, omega_spread=0.3,
                                  connectivity=0.6, seed=42)
        theta_star, res = net.equilibrium()
        assert res < 1e-6

        target = mse_target(net.N, net.output_ids, 0, margin=0.2)

        _, grad_K_tp, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)
        grad_K_fd = _finite_difference_K_gradient(net, theta_star, target)

        # Extract edge values only
        edges = net.edges
        tp_vals = np.array([grad_K_tp[i, j] for (i, j) in edges])
        fd_vals = np.array([grad_K_fd[i, j] for (i, j) in edges])

        ntp = np.linalg.norm(tp_vals)
        nfd = np.linalg.norm(fd_vals)
        assert ntp > 1e-10, "Two-phase K gradient is zero"
        assert nfd > 1e-10, "Finite-diff K gradient is zero"

        cosine = np.dot(tp_vals, fd_vals) / (ntp * nfd)
        assert cosine > 0.99, (
            f"Coupling gradient cosine: {cosine:.6f} (expected > 0.99)")

    @pytest.mark.parametrize("seed", [42, 77, 123])
    def test_coupling_gradient_across_seeds(self, seed):
        """Coupling gradient agreement should hold across different networks."""
        net = make_random_network(N=10, K_mean=5.0, omega_spread=0.3,
                                  connectivity=0.5, seed=seed)
        theta_star, res = net.equilibrium()
        if res > 1e-4:
            pytest.skip(f"Equilibrium did not converge (res={res:.2e})")

        target = mse_target(net.N, net.output_ids, 0, margin=0.2)

        _, grad_K_tp, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)
        grad_K_fd = _finite_difference_K_gradient(net, theta_star, target)

        edges = net.edges
        tp_vals = np.array([grad_K_tp[i, j] for (i, j) in edges])
        fd_vals = np.array([grad_K_fd[i, j] for (i, j) in edges])

        ntp, nfd = np.linalg.norm(tp_vals), np.linalg.norm(fd_vals)
        if ntp < 1e-10 or nfd < 1e-10:
            pytest.skip("Gradient too small to compare")

        cosine = np.dot(tp_vals, fd_vals) / (ntp * nfd)
        assert cosine > 0.99, (
            f"seed={seed}: coupling gradient cosine: {cosine:.6f}")

    def test_coupling_gradient_zero_at_target(self):
        """When output is at target, coupling gradient should be near zero."""
        net = make_random_network(N=8, K_mean=5.0, seed=42)
        theta_star, _ = net.equilibrium()

        # Target = equilibrium → loss = 0 → gradient = 0
        _, grad_K, _ = two_phase_gradient(net, theta_star, theta_star, beta=1e-3)

        edges = net.edges
        max_grad = max(abs(grad_K[i, j]) for (i, j) in edges)
        assert max_grad < 1e-4, (
            f"Coupling gradient should be ~0 at target, max = {max_grad:.2e}")

    def test_coupling_gradient_symmetry(self):
        """grad_K should be symmetric: ∂L/∂K_ij = ∂L/∂K_ji."""
        net = make_random_network(N=10, K_mean=5.0, seed=42)
        theta_star, _ = net.equilibrium()
        target = mse_target(net.N, net.output_ids, 0, margin=0.2)

        _, grad_K, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)

        np.testing.assert_allclose(
            grad_K, grad_K.T, atol=1e-12,
            err_msg="Coupling gradient should be symmetric")

    def test_coupling_gradient_per_edge(self):
        """Per-edge agreement between two-phase and FD gradients."""
        net = make_random_network(N=8, K_mean=5.0, omega_spread=0.3,
                                  connectivity=0.6, seed=42)
        theta_star, res = net.equilibrium()
        assert res < 1e-6

        target = mse_target(net.N, net.output_ids, 0, margin=0.2)

        _, grad_K_tp, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)
        grad_K_fd = _finite_difference_K_gradient(net, theta_star, target)

        for (i, j) in net.edges:
            fd_val = grad_K_fd[i, j]
            tp_val = grad_K_tp[i, j]
            if abs(fd_val) > 1e-6:
                rel_err = abs(tp_val - fd_val) / abs(fd_val)
                assert rel_err < 0.1, (
                    f"Edge ({i},{j}): tp={tp_val:.6f} fd={fd_val:.6f} "
                    f"rel_err={rel_err:.2e}")
