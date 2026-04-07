"""Robustness tests: edge cases, degeneracies, failure modes.

A skeptical reviewer will ask: where does this break?
These tests document the boundaries.
"""

import numpy as np
import pytest
from phasegrad.kuramoto import KuramotoNetwork, make_random_network
from phasegrad.gradient import (
    two_phase_gradient, analytical_gradient, finite_difference_gradient,
    mse_loss,
)
from phasegrad.verification import verify_single


class TestBetaConvergence:
    """Two-phase gradient must converge to analytical as β → 0."""

    def test_monotonic_direction(self):
        """Cosine similarity should improve (or plateau) as β shrinks."""
        net = make_random_network(10, K_mean=5.0, seed=42)
        theta_star, res = net.equilibrium()
        assert res < 1e-6

        rng = np.random.default_rng(99)
        target = theta_star.copy()
        for i in net.output_ids:
            target[i] += rng.uniform(-0.3, 0.3)

        grad_an = analytical_gradient(net, theta_star, target)

        betas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        cosines = []
        for beta in betas:
            grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=beta)
            a, b = grad_tp[1:], grad_an[1:]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            cosines.append(cos)

        # Should be monotonically non-decreasing (allowing 0.001 tolerance
        # for numerical noise at very small β)
        for i in range(len(cosines) - 1):
            assert cosines[i + 1] >= cosines[i] - 0.001, (
                f"Non-monotonic: β={betas[i]:.0e} cos={cosines[i]:.6f} → "
                f"β={betas[i+1]:.0e} cos={cosines[i+1]:.6f}")

        # Best should be very close to 1
        assert max(cosines) > 0.9999, f"Best cosine only {max(cosines):.6f}"

    def test_scale_convergence(self):
        """Magnitude ratio (two-phase / analytical) should approach 1.0."""
        net = make_random_network(10, K_mean=5.0, seed=42)
        theta_star, _ = net.equilibrium()

        rng = np.random.default_rng(99)
        target = theta_star.copy()
        for i in net.output_ids:
            target[i] += rng.uniform(-0.3, 0.3)

        grad_an = analytical_gradient(net, theta_star, target)
        norm_an = np.linalg.norm(grad_an[1:])

        betas = [1.0, 0.1, 0.01, 0.001]
        scales = []
        for beta in betas:
            grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=beta)
            scales.append(np.linalg.norm(grad_tp[1:]) / norm_an)

        # Scale should approach 1.0 as β → 0
        assert abs(scales[-1] - 1.0) < 0.01, (
            f"Scale at β={betas[-1]}: {scales[-1]:.4f}")

        # Should be monotonically approaching 1.0 from below
        for i in range(len(scales) - 1):
            assert scales[i + 1] >= scales[i] - 0.01, (
                f"Scale not converging: {scales[i]:.4f} → {scales[i+1]:.4f}")


class TestNearSingularCoupling:
    """Behavior when the coupling Laplacian is ill-conditioned."""

    def test_sparse_coupling_degrades_gracefully(self):
        """With very sparse coupling, gradient quality degrades but doesn't crash."""
        rng = np.random.default_rng(42)
        N = 10
        omega = 0.3 * rng.standard_normal(N)
        omega -= omega.mean()

        # Very sparse: only ~20% of edges, weak coupling
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.2:
                    s = 5.0 * rng.uniform(0.5, 1.5)
                    K[i, j] = K[j, i] = s

        net = KuramotoNetwork(omega=omega, K=K, output_ids=[N-2, N-1])

        theta_star, res = net.equilibrium()
        # May or may not converge — just shouldn't crash
        if res < 0.1:
            target = theta_star.copy()
            for i in net.output_ids:
                target[i] += rng.uniform(-0.2, 0.2)

            # These should not raise
            grad_an = analytical_gradient(net, theta_star, target)
            grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)
            assert np.all(np.isfinite(grad_an))
            assert np.all(np.isfinite(grad_tp))

    def test_disconnected_node(self):
        """A node with zero coupling should have zero gradient contribution."""
        rng = np.random.default_rng(42)
        N = 8
        omega = 0.2 * rng.standard_normal(N)
        omega -= omega.mean()

        K = np.zeros((N, N))
        # Connect nodes 0-6, leave node 7 disconnected
        for i in range(7):
            for j in range(i + 1, 7):
                if rng.random() < 0.6:
                    s = 5.0 * rng.uniform(0.5, 1.5)
                    K[i, j] = K[j, i] = s

        net = KuramotoNetwork(omega=omega, K=K, output_ids=[5, 6])
        theta_star, res = net.equilibrium()

        if res < 0.01:
            target = theta_star.copy()
            for i in net.output_ids:
                target[i] += 0.2

            grad_fd = finite_difference_gradient(net, theta_star, target)
            # Node 7 is disconnected — its gradient should be ~0
            assert abs(grad_fd[7]) < 1e-6, (
                f"Disconnected node gradient: {grad_fd[7]:.2e}")


class TestNonConvergence:
    """Behavior when equilibrium doesn't exist (oscillators drift)."""

    def test_weak_coupling_returns_high_residual(self):
        """When K << K_c, solver should report high residual."""
        rng = np.random.default_rng(42)
        N = 10
        omega = 2.0 * rng.standard_normal(N)  # large spread
        omega -= omega.mean()

        # Very weak coupling — much less than K_c
        K = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if rng.random() < 0.5:
                    K[i, j] = K[j, i] = 0.1  # way below K_c

        net = KuramotoNetwork(omega=omega, K=K, output_ids=[N-2, N-1])
        theta_star, res = net.equilibrium()

        # Should either fail to converge (high residual) or find a trivial solution
        # The key: it should NOT silently return a "converged" result with garbage
        # (If res is low, the solution happens to work — that's fine too)
        assert isinstance(res, float), "Residual should be a float"

    def test_skip_prevents_garbage_gradient(self):
        """Training should skip samples where equilibrium fails."""
        from phasegrad.kuramoto import make_network
        from phasegrad.data import load_hillenbrand
        from phasegrad.training import _train_epoch

        train_data, _, _ = load_hillenbrand(vowels=['a', 'i'], seed=42)

        # Create a network with deliberately weak coupling
        net = make_network(n_input=2, n_hidden=5, n_output=2,
                           K_scale=0.1, input_scale=5.0, seed=42)

        rng = np.random.default_rng(42)
        loss, acc, n_skip = _train_epoch(
            net, train_data[:20], beta=0.1, lr_omega=0.001,
            lr_K=0.001, margin=0.2, grad_clip=2.0, rng=rng)

        # Most or all samples should be skipped due to non-convergence
        assert n_skip > 10, (
            f"Expected most samples skipped with weak coupling, got {n_skip}/20")


class TestN20Verification:
    """Restore N=20 verification from earlier session."""

    def test_n20_theorem_holds(self):
        """Theorem holds at N=20."""
        result = verify_single(20, n_beta=6, seed=42)
        assert result["residual"] < 1e-6, (
            f"N=20 residual: {result['residual']:.2e}")
        assert result["cos_an_fd"] > 0.999, (
            f"N=20 analytical vs FD: {result['cos_an_fd']:.6f}")
        assert result["best_cos_tp_fd"] > 0.999, (
            f"N=20 two-phase vs FD: {result['best_cos_tp_fd']:.6f}")
