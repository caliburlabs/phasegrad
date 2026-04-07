"""Gradient verification utilities for the Phase-Gradient Duality Theorem.

Verifies that the two-phase gradient converges to the analytical gradient
as clamping strength β → 0, across multiple network sizes.
"""

from __future__ import annotations

import time

import numpy as np

from phasegrad.kuramoto import KuramotoNetwork, make_random_network
from phasegrad.gradient import (
    analytical_gradient, two_phase_gradient, finite_difference_gradient,
    mse_loss,
)


def run_verification(sizes: list[int] = [6, 10, 15, 20],
                     n_beta: int = 8,
                     seed: int = 42,
                     verbose: bool = True,
                     ) -> list[dict]:
    """Verify the Phase-Gradient Duality Theorem across network sizes.

    For each size N:
      1. Create a random synchronized network.
      2. Find free equilibrium.
      3. Compute analytical, finite-difference, and two-phase gradients.
      4. Report cosine similarities.

    Args:
        sizes: list of network sizes to test.
        n_beta: number of β values for convergence sweep.
        seed: random seed.
        verbose: print progress.

    Returns:
        List of result dicts, one per size.
    """
    results = []

    for N in sizes:
        if verbose:
            print(f"\n  N={N}:", end=" ", flush=True)

        r = verify_single(N, n_beta=n_beta, seed=seed, verbose=verbose)
        results.append(r)

        if verbose:
            print(f"an-fd={r['cos_an_fd']:+.6f} "
                  f"tp-fd={r['best_cos_tp_fd']:+.6f} "
                  f"res={r['residual']:.1e}")

    if verbose:
        print(f"\n  {'N':>5s}  {'residual':>10s}  {'an vs fd':>10s}  {'tp vs fd':>10s}")
        for r in results:
            print(f"  {r['N']:5d}  {r['residual']:10.1e}  "
                  f"{r['cos_an_fd']:+10.6f}  {r['best_cos_tp_fd']:+10.6f}")

    return results


def verify_single(N: int, n_beta: int = 8, seed: int = 42,
                  verbose: bool = False) -> dict:
    """Verify the theorem for a single network size."""
    rng = np.random.default_rng(seed)

    n_output = max(2, N // 4)
    net = make_random_network(N, K_mean=5.0, omega_spread=0.3,
                              connectivity=0.6, n_output=n_output, seed=seed)

    # Find free equilibrium
    theta_star, residual = net.equilibrium()

    if residual > 1e-4:
        # Increase coupling if needed
        net.K *= 2.0
        theta_star, residual = net.equilibrium()

    # Random target (small perturbation from equilibrium)
    target = theta_star.copy()
    for i in net.output_ids:
        target[i] += rng.uniform(-0.3, 0.3)

    loss = mse_loss(theta_star, target, net.output_ids)

    # Analytical gradient
    grad_an = analytical_gradient(net, theta_star, target)

    # Finite-difference gradient
    t0 = time.time()
    grad_fd = finite_difference_gradient(net, theta_star, target, eps=1e-5)
    fd_time = time.time() - t0

    # Cosine: analytical vs FD
    cos_an_fd = _cosine_skip0(grad_an, grad_fd)

    # Two-phase at decreasing β
    betas = np.logspace(1, -3, n_beta)
    convergence = []
    for beta in betas:
        grad_tp, _, tp_res = two_phase_gradient(net, theta_star, target, beta=beta)
        cos_tp_fd = _cosine_skip0(grad_tp, grad_fd)
        cos_tp_an = _cosine_skip0(grad_tp, grad_an)
        scale = np.linalg.norm(grad_tp[1:]) / (np.linalg.norm(grad_an[1:]) + 1e-15)
        convergence.append({
            "beta": float(beta),
            "cos_tp_fd": cos_tp_fd,
            "cos_tp_an": cos_tp_an,
            "scale": float(scale),
        })

    best_tp_fd = max(convergence, key=lambda c: abs(c["cos_tp_fd"]))

    return {
        "N": N,
        "n_output": n_output,
        "residual": residual,
        "loss": loss,
        "cos_an_fd": cos_an_fd,
        "best_cos_tp_fd": best_tp_fd["cos_tp_fd"],
        "best_beta": best_tp_fd["beta"],
        "convergence": convergence,
        "grad_analytical": grad_an.tolist(),
        "grad_fd": grad_fd.tolist(),
        "fd_time_s": fd_time,
    }


def _cosine_skip0(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, skipping index 0 (pinned node)."""
    a, b = a[1:], b[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
