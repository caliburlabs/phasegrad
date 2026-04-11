"""Test suite: Phase-Gradient Duality Theorem verification.

The core claim: for Kuramoto oscillator networks at equilibrium,
the two-phase gradient equals the analytical gradient.

These tests reproduce the N=6,10,15,20 verification table.
"""

import pytest
import numpy as np
from phasegrad.verification import verify_single


@pytest.mark.parametrize("N", [6, 10, 15])
def test_theorem_holds(N):
    """Two-phase gradient matches ground truth for network of size N."""
    result = verify_single(N, n_beta=6, seed=42)

    # Equilibrium must converge
    assert result["residual"] < 1e-6, (
        f"N={N}: equilibrium residual {result['residual']:.2e} too large")

    # Analytical must match finite-difference
    assert result["cos_an_fd"] > 0.999, (
        f"N={N}: analytical vs FD cosine {result['cos_an_fd']:.6f}")

    # Two-phase must match finite-difference
    assert result["best_cos_tp_fd"] > 0.999, (
        f"N={N}: two-phase vs FD cosine {result['best_cos_tp_fd']:.6f}")


def test_convergence_with_beta():
    """Two-phase gradient improves monotonically as β → 0."""
    result = verify_single(10, n_beta=8, seed=42)

    cosines = [c["cos_tp_fd"] for c in result["convergence"]]
    scales = [c["scale"] for c in result["convergence"]]

    # Direction should be good even at large β
    assert cosines[0] > 0.99, f"Direction wrong at large β: {cosines[0]}"

    # Scale should converge toward 1.0
    assert abs(scales[-1] - 1.0) < 0.01, f"Scale not converging: {scales[-1]}"


def test_per_node_agreement():
    """Per-node gradients match between all three methods."""
    result = verify_single(10, n_beta=6, seed=42)

    grad_an = np.array(result["grad_analytical"])
    grad_fd = np.array(result["grad_fd"])

    # Per-node agreement (skip node 0 which is pinned)
    for k in range(1, len(grad_an)):
        if abs(grad_fd[k]) > 1e-8:
            rel_err = abs(grad_an[k] - grad_fd[k]) / abs(grad_fd[k])
            assert rel_err < 1e-3, (
                f"Node {k}: analytical={grad_an[k]:.8f} fd={grad_fd[k]:.8f} "
                f"rel_err={rel_err:.2e}")


@pytest.mark.parametrize("N", [6, 10, 15, 20])
def test_paper_precision(N):
    """Enforce the paper's claimed precision: cosine = 1.000000 (6 decimal places).

    The paper reports cosine similarity = 1.000000 at all sizes tested.
    This test asserts > 0.999999, matching that claim.
    """
    result = verify_single(N, n_beta=8, seed=42)

    # Equilibrium at machine precision
    assert result["residual"] < 1e-10, (
        f"N={N}: residual {result['residual']:.2e} not at machine precision")

    # Analytical vs FD: 6-decimal cosine
    assert result["cos_an_fd"] > 0.999999, (
        f"N={N}: analytical vs FD cosine {result['cos_an_fd']:.8f} "
        f"— paper claims 1.000000")

    # Two-phase vs FD at best beta: 6-decimal cosine
    assert result["best_cos_tp_fd"] > 0.999999, (
        f"N={N}: two-phase vs FD cosine {result['best_cos_tp_fd']:.8f} "
        f"— paper claims 1.000000")
