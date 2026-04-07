"""Test suite: analytical gradient matches finite-difference."""

import numpy as np
import pytest
from phasegrad.kuramoto import make_random_network
from phasegrad.gradient import analytical_gradient, finite_difference_gradient, mse_loss


def test_analytical_matches_fd():
    """Analytical gradient equals finite-difference to high precision."""
    net = make_random_network(10, K_mean=5.0, seed=42)
    theta_star, res = net.equilibrium()
    assert res < 1e-6

    rng = np.random.default_rng(99)
    target = theta_star.copy()
    for i in net.output_ids:
        target[i] += rng.uniform(-0.3, 0.3)

    grad_an = analytical_gradient(net, theta_star, target)
    grad_fd = finite_difference_gradient(net, theta_star, target, eps=1e-5)

    # Skip node 0 (pinned)
    for k in range(1, net.N):
        if abs(grad_fd[k]) > 1e-8:
            rel_err = abs(grad_an[k] - grad_fd[k]) / abs(grad_fd[k])
            assert rel_err < 1e-3, (
                f"Node {k}: analytical={grad_an[k]:.8f} fd={grad_fd[k]:.8f}")


def test_gradient_zero_at_target():
    """Gradient should be ~zero when equilibrium matches target."""
    net = make_random_network(8, K_mean=5.0, seed=42)
    theta_star, res = net.equilibrium()
    assert res < 1e-6

    # Target = current equilibrium → loss = 0, gradient = 0
    target = theta_star.copy()
    grad = analytical_gradient(net, theta_star, target)

    assert np.max(np.abs(grad)) < 1e-10, (
        f"Gradient not zero at target: max={np.max(np.abs(grad)):.2e}")
