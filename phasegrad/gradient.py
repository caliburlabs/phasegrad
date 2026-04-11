"""Gradient computation for Kuramoto oscillator networks.

Three methods to compute ∂L/∂ω:

1. **Two-phase (equilibrium propagation)**: The physics-based method.
   Run free equilibrium → clamp outputs → measure phase displacement.
   No backpropagation. (Eq. 8)

2. **Analytical (implicit differentiation)**: Exact gradient via
   the inverse of the reduced Jacobian. (Eq. 6-7)

3. **Finite-difference**: Numerical reference. Perturb each ω_k
   by ±ε and re-solve. Slow but unambiguous. (Eq. 9)

The Phase-Gradient Duality Theorem states that (1) converges to (2)
as the clamping strength β → 0:

    lim_{β→0} (θ^β - θ*) / β = -∂L/∂ω     (Theorem 1)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from phasegrad.kuramoto import KuramotoNetwork, kuramoto_jacobian
from phasegrad.losses import mse_loss  # noqa: F401 — re-exported


LossFn = Callable[[np.ndarray, np.ndarray, list[int]], float]


def two_phase_gradient(net: KuramotoNetwork,
                       theta_free: np.ndarray,
                       target: np.ndarray,
                       beta: float = 1e-4,
                       ) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute gradient via equilibrium propagation (Eq. 8).

    The two-phase gradient for natural frequencies:
        ∂L/∂ω_k ≈ -(θ_k^β - θ_k*) / β

    And for coupling strengths:
        ∂L/∂K_ij ≈ [cos(θ_j* - θ_i*) - cos(θ_j^β - θ_i^β)] / β

    Args:
        net: the oscillator network.
        theta_free: (N,) free equilibrium phases θ*.
        target: (N,) target phases for clamping.
        beta: clamping strength (smaller → more accurate, harder to solve).

    Returns:
        (grad_omega, grad_K, residual):
            grad_omega: (N,) gradient w.r.t. natural frequencies.
            grad_K: (N, N) gradient w.r.t. coupling matrix.
            residual: max |RHS| at the clamped equilibrium.
    """
    omega_c = net.omega_centered
    theta_clamped, residual = net.clamped_equilibrium(
        beta, target, theta_init=theta_free.copy(), omega_c=omega_c)

    grad_omega = -(theta_clamped - theta_free) / beta

    # Coupling gradient from equilibrium propagation
    diff_free = theta_free[np.newaxis, :] - theta_free[:, np.newaxis]
    diff_clamp = theta_clamped[np.newaxis, :] - theta_clamped[:, np.newaxis]
    grad_K = (np.cos(diff_free) - np.cos(diff_clamp)) / beta

    return grad_omega, grad_K, residual


def analytical_gradient(net: KuramotoNetwork,
                        theta_star: np.ndarray,
                        target: np.ndarray,
                        output_ids: list[int] | None = None,
                        ) -> np.ndarray:
    """Compute ∂L/∂ω analytically via implicit differentiation (Eq. 6-7).

    At equilibrium, F(θ*, ω) = 0. By the implicit function theorem:
        ∂θ*/∂ω = -J⁻¹

    where J is the Jacobian of the equilibrium equations. Then:
        ∂L/∂ω = -J⁻ᵀ · (∂L/∂θ)

    For MSE loss, ∂L/∂θ_i = θ_i - target_i at output nodes.

    The Jacobian has a zero eigenvalue (global phase rotation), so we
    work with the reduced system (θ_0 pinned).

    Args:
        net: the oscillator network.
        theta_star: (N,) free equilibrium phases.
        target: (N,) target phases.
        output_ids: which nodes have loss terms. Defaults to net.output_ids.

    Returns:
        (N,) gradient vector. grad[0] = 0 (pinned node).
    """
    if output_ids is None:
        output_ids = net.output_ids

    N = net.N
    J_full = kuramoto_jacobian(theta_star, net.K)
    J_red = J_full[1:, 1:]  # remove pinned node

    # Error vector (only at output nodes)
    error = np.zeros(N)
    for i in output_ids:
        error[i] = theta_star[i] - target[i]
    error_red = error[1:]

    # Solve J_red^T · x = error_red
    try:
        x = np.linalg.solve(J_red.T, error_red)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(J_red.T, error_red, rcond=None)[0]

    grad = np.zeros(N)
    grad[1:] = -x
    return grad


def finite_difference_gradient(net: KuramotoNetwork,
                               theta_star: np.ndarray,
                               target: np.ndarray,
                               output_ids: list[int] | None = None,
                               loss_fn: LossFn = mse_loss,
                               eps: float = 1e-5,
                               ) -> np.ndarray:
    """Compute ∂L/∂ω by finite differences (Eq. 9).

    Perturbs the mean-centered frequencies directly to avoid the
    mean-shift artifact: changing raw ω_k shifts mean(ω) by eps/N,
    corrupting all other centered frequencies.

    Args:
        net: the oscillator network.
        theta_star: (N,) free equilibrium (used as warm start).
        target: (N,) target phases.
        output_ids: which nodes have loss terms.
        loss_fn: callable(theta, target, output_ids) → float.
        eps: perturbation size.

    Returns:
        (N,) gradient vector.
    """
    if output_ids is None:
        output_ids = net.output_ids

    N = net.N
    omega_c = net.omega_centered.copy()
    grad = np.zeros(N)

    for k in range(N):
        omega_plus = omega_c.copy()
        omega_plus[k] += eps
        theta_plus, _ = net.equilibrium(theta_init=theta_star.copy(),
                                         omega_c=omega_plus)
        L_plus = loss_fn(theta_plus, target, output_ids)

        omega_minus = omega_c.copy()
        omega_minus[k] -= eps
        theta_minus, _ = net.equilibrium(theta_init=theta_star.copy(),
                                          omega_c=omega_minus)
        L_minus = loss_fn(theta_minus, target, output_ids)

        grad[k] = (L_plus - L_minus) / (2 * eps)

    return grad


def verify_gradients(net: KuramotoNetwork,
                     theta_star: np.ndarray,
                     target: np.ndarray,
                     beta: float = 1e-4,
                     eps: float = 1e-5,
                     ) -> dict:
    """Compare all three gradient methods and return cosine similarities.

    Args:
        net: the oscillator network.
        theta_star: (N,) free equilibrium.
        target: (N,) target phases.
        beta: clamping strength for two-phase method.
        eps: perturbation for finite differences.

    Returns:
        dict with keys: grad_analytical, grad_twophase, grad_fd,
        cos_an_fd, cos_tp_fd, cos_tp_an.
    """
    grad_an = analytical_gradient(net, theta_star, target)
    grad_tp, _, _ = two_phase_gradient(net, theta_free=theta_star, target=target, beta=beta)
    grad_fd = finite_difference_gradient(net, theta_star, target, eps=eps)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        # Skip pinned node (index 0)
        a, b = a[1:], b[1:]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    return {
        "grad_analytical": grad_an,
        "grad_twophase": grad_tp,
        "grad_fd": grad_fd,
        "cos_an_fd": _cosine(grad_an, grad_fd),
        "cos_tp_fd": _cosine(grad_tp, grad_fd),
        "cos_tp_an": _cosine(grad_tp, grad_an),
    }
