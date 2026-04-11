"""PyTorch autograd verification of the Phase-Gradient Duality Theorem.

Provides a computationally independent check of the gradient identity by
reimplementing the Kuramoto equilibrium solver in PyTorch and differentiating
through it via the implicit function theorem using a custom
torch.autograd.Function.

This implementation shares NO code with the scipy-based methods in
gradient.py / verification.py. It uses:
  - PyTorch tensors throughout (no numpy in the forward/backward path)
  - Newton iteration for the equilibrium solve (not scipy.optimize.fsolve)
  - Implicit differentiation via torch.linalg.solve for the backward pass

Usage:
    from phasegrad.autograd_verify import autograd_gradient, verify_autograd

    # Single verification
    result = verify_autograd(N=15, seed=42)
    print(f"Cosine(AG vs TP): {result['cos_ag_tp']:.6f}")

    # Full table
    results = verify_autograd_table(sizes=[6, 10, 15, 20, 30, 50, 100, 200])
"""

from __future__ import annotations

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np


def _require_torch():
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for autograd verification. "
            "Install it with: pip install torch"
        )


# ---------------------------------------------------------------------------
# Kuramoto equilibrium as a custom autograd function
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    def _kuramoto_rhs_torch(theta, omega_c, K):
        """Kuramoto RHS in PyTorch: ω_i^c + Σ_j K_ij sin(θ_j - θ_i)."""
        diff = theta.unsqueeze(0) - theta.unsqueeze(1)  # diff[i,j] = θ_j - θ_i
        return omega_c + (K * torch.sin(diff)).sum(dim=1)

    def _kuramoto_jacobian_torch(theta, K):
        """Jacobian of Kuramoto equilibrium equations in PyTorch."""
        diff = theta.unsqueeze(0) - theta.unsqueeze(1)
        cos_diff = torch.cos(diff)
        J = K * cos_diff
        J = J - torch.diag(J.sum(dim=1))  # zero diagonal first, then set
        # More precisely: off-diag stays, diag = -sum of off-diag in each row
        J_off = K * cos_diff
        J_off = J_off - torch.diag(torch.diag(J_off))  # zero the diagonal
        J = J_off - torch.diag(J_off.sum(dim=1))
        return J

    def _newton_solve(omega_c, K, max_iter=200, tol=1e-12):
        """Solve Kuramoto equilibrium via Newton iteration in PyTorch.

        Pins θ_0 = 0, solves the reduced (N-1) system.
        Returns (theta, residual) as plain tensors (no grad tracking).
        """
        N = omega_c.shape[0]
        phi = torch.zeros(N - 1, dtype=omega_c.dtype)

        for _ in range(max_iter):
            theta = torch.zeros(N, dtype=omega_c.dtype)
            theta[1:] = phi

            F = _kuramoto_rhs_torch(theta, omega_c, K)
            F_red = F[1:]

            res = F_red.abs().max().item()
            if res < tol:
                break

            J = _kuramoto_jacobian_torch(theta, K)
            J_red = J[1:, 1:]

            dphi = torch.linalg.solve(J_red, -F_red)
            phi = phi + dphi

        theta = torch.zeros(N, dtype=omega_c.dtype)
        theta[1:] = phi
        residual = _kuramoto_rhs_torch(theta, omega_c, K)[1:].abs().max().item()
        return theta, residual

    class KuramotoEquilibrium(torch.autograd.Function):
        """Custom autograd function for Kuramoto equilibrium.

        Forward: solve F(θ*, ω^c) = 0 for θ* given ω^c and K.
        Backward: use implicit function theorem.
            ∂θ*/∂ω^c = -J⁻¹  (since ∂F/∂ω^c = I)
            So ∂L/∂ω^c = -(J⁻ᵀ) · ∂L/∂θ*
        """

        @staticmethod
        def forward(ctx, omega_c, K):
            # Solve equilibrium (no grad tracking needed in the solve)
            with torch.no_grad():
                theta_star, residual = _newton_solve(omega_c.detach(), K.detach())

            # Save for backward
            ctx.save_for_backward(theta_star, K)
            ctx.residual = residual
            return theta_star

        @staticmethod
        def backward(ctx, grad_output):
            theta_star, K = ctx.saved_tensors

            # Jacobian at equilibrium
            J = _kuramoto_jacobian_torch(theta_star, K)
            J_red = J[1:, 1:]

            # grad_output is ∂L/∂θ*
            # We need ∂L/∂ω^c = -(J_red⁻ᵀ) · (∂L/∂θ*)[1:]
            grad_out_red = grad_output[1:]

            # Solve J_red^T · x = grad_out_red
            x = torch.linalg.solve(J_red.T, grad_out_red)

            grad_omega = torch.zeros_like(grad_output)
            grad_omega[1:] = -x

            # No gradient for K in this path (we only verify ω gradients)
            return grad_omega, None

    def autograd_gradient(omega, K, output_ids, target):
        """Compute ∂L/∂ω^c via PyTorch autograd.

        Args:
            omega: (N,) numpy array of natural frequencies.
            K: (N, N) numpy array of coupling matrix.
            output_ids: list of output node indices.
            target: (N,) numpy array of target phases.

        Returns:
            dict with 'grad': (N,) numpy gradient, 'theta_star': (N,) equilibrium,
            'residual': float, 'loss': float.
        """
        _require_torch()

        N = len(omega)
        omega_c_np = omega - omega.mean()

        omega_c = torch.tensor(omega_c_np, dtype=torch.float64, requires_grad=True)
        K_t = torch.tensor(K, dtype=torch.float64)
        target_t = torch.tensor(target, dtype=torch.float64)

        # Get equilibrium and residual via a standalone solve first
        with torch.no_grad():
            _, residual = _newton_solve(omega_c.detach(), K_t)

        # Forward: find equilibrium (with autograd tracking)
        theta_star = KuramotoEquilibrium.apply(omega_c, K_t)

        # MSE loss on output nodes
        loss = 0.5 * sum((theta_star[i] - target_t[i]) ** 2 for i in output_ids)

        # Backward
        loss.backward()

        return {
            'grad': omega_c.grad.detach().numpy().copy(),
            'theta_star': theta_star.detach().numpy().copy(),
            'residual': residual,
            'loss': loss.item(),
        }


def verify_autograd(N=15, seed=42, beta=1e-4):
    """Verify the gradient identity using PyTorch autograd as independent check.

    Creates a random network, computes gradients via:
    1. Two-phase (EP) — from phasegrad.gradient
    2. Analytical (IFT) — from phasegrad.gradient
    3. PyTorch autograd — this module (computationally independent)

    Returns dict with cosine similarities between all pairs.
    """
    _require_torch()

    from phasegrad.kuramoto import make_random_network
    from phasegrad.gradient import two_phase_gradient, analytical_gradient

    n_output = max(2, N // 4)
    net = make_random_network(N, K_mean=5.0, omega_spread=0.3,
                              connectivity=0.6, n_output=n_output, seed=seed)

    # Find equilibrium (scipy path)
    theta_star, residual = net.equilibrium()
    if residual > 1e-4:
        net.K *= 2.0
        theta_star, residual = net.equilibrium()

    # Random target
    rng = np.random.default_rng(seed)
    target = theta_star.copy()
    for i in net.output_ids:
        target[i] += rng.uniform(-0.3, 0.3)

    # Method 1: Two-phase (scipy)
    grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=beta)

    # Method 2: Analytical (scipy)
    grad_an = analytical_gradient(net, theta_star, target)

    # Method 3: PyTorch autograd (independent)
    ag_result = autograd_gradient(net.omega, net.K, net.output_ids, target)
    grad_ag = ag_result['grad']

    def _cosine(a, b):
        a, b = a[1:], b[1:]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    return {
        'N': N,
        'residual': residual,
        'cos_ag_tp': _cosine(grad_ag, grad_tp),
        'cos_ag_an': _cosine(grad_ag, grad_an),
        'cos_tp_an': _cosine(grad_tp, grad_an),
        'grad_ag': grad_ag,
        'grad_tp': grad_tp,
        'grad_an': grad_an,
    }


def verify_autograd_table(sizes=None, seed=42, beta=1e-4, verbose=True):
    """Reproduce the verification table with autograd cross-check.

    Args:
        sizes: list of network sizes. Default: [6, 10, 15, 20, 30, 50, 100, 200].
        seed: random seed.
        beta: clamping strength for two-phase method.
        verbose: print results.

    Returns:
        List of result dicts, one per size.
    """
    _require_torch()

    if sizes is None:
        sizes = [6, 10, 15, 20, 30, 50, 100, 200]

    results = []
    if verbose:
        print(f"{'N':>5s}  {'Freq params':>11s}  {'Cos(TP vs AN)':>13s}  "
              f"{'Cos(AG vs TP)':>13s}  {'Eq. residual':>12s}")

    for N in sizes:
        r = verify_autograd(N=N, seed=seed, beta=beta)

        entry = {
            'N': N,
            'freq_params': N - 1,
            'cos_tp_an': r['cos_tp_an'],
            'cos_ag_tp': r['cos_ag_tp'],
            'cos_ag_an': r['cos_ag_an'],
            'residual': r['residual'],
        }
        results.append(entry)

        if verbose:
            print(f"{N:5d}  {N-1:11d}  {r['cos_tp_an']:13.6f}  "
                  f"{r['cos_ag_tp']:13.6f}  {r['residual']:12.1e}")

    return results
