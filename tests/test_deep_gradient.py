"""Gradient identity verification for deep (multi-hidden-layer) networks.

Verifies that the two-phase gradient matches the analytical gradient
(and both match finite-difference) for networks built with
make_deep_network at various depths and widths.

This extends the paper 1 verification table to every architecture
tested in paper 2.
"""

import pytest
import numpy as np
from phasegrad.kuramoto import make_deep_network
from phasegrad.gradient import (
    analytical_gradient,
    two_phase_gradient,
    finite_difference_gradient,
    mse_loss,
)


def _cosine(a, b):
    """Cosine similarity, skipping pinned node 0."""
    a, b = a[1:], b[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _verify_gradient(net, seed=42):
    """Run the three-way gradient check on a network.

    Returns dict with cosine similarities and residual.
    """
    rng = np.random.default_rng(seed)

    # Find free equilibrium
    theta_star, residual = net.equilibrium()
    assert residual < 1e-4, f"Equilibrium failed: residual={residual:.2e}"

    # Random target: perturb output phases
    target = theta_star.copy()
    for i in net.output_ids:
        target[i] += rng.uniform(-0.3, 0.3)

    # Three gradient methods
    grad_an = analytical_gradient(net, theta_star, target)
    grad_fd = finite_difference_gradient(net, theta_star, target, eps=1e-5)
    grad_tp, _, _ = two_phase_gradient(net, theta_free=theta_star,
                                       target=target, beta=1e-4)

    cos_an_fd = _cosine(grad_an, grad_fd)
    cos_tp_fd = _cosine(grad_tp, grad_fd)
    cos_tp_an = _cosine(grad_tp, grad_an)

    return {
        'residual': residual,
        'cos_an_fd': cos_an_fd,
        'cos_tp_fd': cos_tp_fd,
        'cos_tp_an': cos_tp_an,
        'grad_an': grad_an,
        'grad_fd': grad_fd,
    }


# ── Depth tests ──────────────────────────────────────────────────────

DEPTH_CONFIGS = [
    ([5, 5], "2+5+5+2"),
    ([5, 5, 5], "2+5+5+5+2"),
]


@pytest.mark.parametrize("hidden_layers,label", DEPTH_CONFIGS,
                         ids=[c[1] for c in DEPTH_CONFIGS])
def test_gradient_identity_depth(hidden_layers, label):
    """Phase-gradient identity holds for deep layered networks."""
    net = make_deep_network(n_input=2, hidden_layers=hidden_layers,
                            n_output=2, K_scale=2.0, seed=42)
    r = _verify_gradient(net)

    assert r['cos_an_fd'] > 0.9999, (
        f"{label}: analytical vs FD cosine {r['cos_an_fd']:.6f}")
    assert r['cos_tp_fd'] > 0.999, (
        f"{label}: two-phase vs FD cosine {r['cos_tp_fd']:.6f}")


# ── Width tests ──────────────────────────────────────────────────────

WIDTH_CONFIGS = [
    ([3], "2+3+2"),
    ([5], "2+5+2"),
    ([12], "2+12+2"),
    ([20], "2+20+2"),
]


@pytest.mark.parametrize("hidden_layers,label", WIDTH_CONFIGS,
                         ids=[c[1] for c in WIDTH_CONFIGS])
def test_gradient_identity_width(hidden_layers, label):
    """Phase-gradient identity holds across hidden layer widths."""
    net = make_deep_network(n_input=2, hidden_layers=hidden_layers,
                            n_output=2, K_scale=2.0, seed=42)
    r = _verify_gradient(net)

    assert r['cos_an_fd'] > 0.9999, (
        f"{label}: analytical vs FD cosine {r['cos_an_fd']:.6f}")
    assert r['cos_tp_fd'] > 0.999, (
        f"{label}: two-phase vs FD cosine {r['cos_tp_fd']:.6f}")


# ── Robustness: multiple random equilibria per architecture ──────────

@pytest.mark.parametrize("seed", range(5))
def test_gradient_depth2_multi_seed(seed):
    """Gradient identity at depth=2 across 5 random equilibria."""
    net = make_deep_network(hidden_layers=[5, 5], K_scale=2.0, seed=42 + seed)
    r = _verify_gradient(net, seed=seed)
    assert r['cos_an_fd'] > 0.9999
    assert r['cos_tp_fd'] > 0.999


# ── Per-node agreement for deep networks ─────────────────────────────

def test_per_node_agreement_deep():
    """Per-node gradients match between analytical and FD for 2+5+5+2."""
    net = make_deep_network(hidden_layers=[5, 5], K_scale=2.0, seed=42)
    r = _verify_gradient(net)

    grad_an = r['grad_an']
    grad_fd = r['grad_fd']

    for k in range(1, len(grad_an)):
        if abs(grad_fd[k]) > 1e-8:
            rel_err = abs(grad_an[k] - grad_fd[k]) / abs(grad_fd[k])
            assert rel_err < 1e-3, (
                f"Node {k}: analytical={grad_an[k]:.8f} "
                f"fd={grad_fd[k]:.8f} rel_err={rel_err:.2e}")
