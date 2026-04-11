#!/usr/bin/env python3.12
"""Verify the ω-gradient identity on the forced Kuramoto system.

The forced system (Eq. F1):
    dθ_i/dt = ω_i + Σ K_ij sin(θ_j - θ_i) + F_i sin(Ψ - θ_i)

The Jacobian of the forced system:
    J_ij = K_ij cos(θ_j - θ_i)                     (i ≠ j, same as unforced)
    J_ii = -Σ_j K_ij cos(θ_j - θ_i) - F_i cos(Ψ - θ_i)   (extra forcing term)

The forcing term adds to the diagonal only → J is still symmetric
(diagonal additions preserve symmetry). Therefore the EP identity
should hold: (θ^β - θ*)/β = -∂L/∂ω as β → 0.

This script verifies this numerically.
"""
import json, numpy as np
from phasegrad.forced import OscillatorBank, forced_kuramoto_rhs


def forced_jacobian(theta, K, F, input_phase):
    """Jacobian of the forced Kuramoto system.

    J_ij = K_ij cos(θ_j - θ_i)                                   for i ≠ j
    J_ii = -Σ_j K_ij cos(θ_j - θ_i) - F_i cos(input_phase - θ_i)  diagonal
    """
    N = len(theta)
    diff = theta[np.newaxis, :] - theta[:, np.newaxis]
    cos_diff = np.cos(diff)
    J = K * cos_diff
    np.fill_diagonal(J, 0.0)
    diag = -np.sum(J, axis=1)
    # Add forcing term to diagonal
    diag -= F * np.cos(input_phase - theta)
    np.fill_diagonal(J, diag)
    return J


def analytical_gradient_forced(bank, theta_star, target, input_freq):
    """Analytical gradient via IFT on the forced system."""
    N = bank.N
    omega_rot = bank._omega_in_input_frame(input_freq)
    input_phase = 0.0

    J_full = forced_jacobian(theta_star, bank.K, bank.F, input_phase)
    J_red = J_full[1:, 1:]  # remove pinned oscillator

    # Check symmetry
    asym = np.max(np.abs(J_red - J_red.T))

    error = np.zeros(N)
    for o in bank.output_ids:
        error[o] = theta_star[o] - target[o]
    error_red = error[1:]

    try:
        x = np.linalg.solve(J_red.T, error_red)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(J_red.T, error_red, rcond=None)[0]

    grad = np.zeros(N)
    grad[1:] = -x
    return grad, asym


def fd_gradient_forced(bank, theta_star, target, input_freq, eps=1e-5):
    """Finite-difference gradient on the forced system."""
    N = bank.N
    grad = np.zeros(N)

    def loss(theta):
        return 0.5 * sum((theta[o] - target[o])**2 for o in bank.output_ids)

    for k in range(N):
        orig = bank.omega[k]

        bank.omega[k] = orig + eps
        th_p, _ = bank.forced_equilibrium(input_freq, theta_star.copy())
        L_p = loss(th_p)

        bank.omega[k] = orig - eps
        th_m, _ = bank.forced_equilibrium(input_freq, theta_star.copy())
        L_m = loss(th_m)

        grad[k] = (L_p - L_m) / (2 * eps)
        bank.omega[k] = orig

    return grad


def twophase_gradient_forced(bank, theta_star, target, input_freq, beta=1e-3):
    """Two-phase (EP) gradient on the forced system."""
    theta_clamp, res = bank.forced_clamped_equilibrium(
        input_freq, beta, target, theta_star.copy())
    grad = -(theta_clamp - theta_star) / beta
    return grad, res


def cosine(a, b):
    """Cosine similarity, skipping index 0 (pinned)."""
    a, b = a[1:], b[1:]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


if __name__ == '__main__':
    print("Forced Kuramoto Gradient Verification")
    print("=" * 60)

    results = []

    # Try multiple configurations to find convergent equilibria
    configs = [
        # (n_sensors, freq_range, K_scale, F_strength, test_freqs)
        (6, (1.0, 6.0), 2.0, 5.0, [1.5, 3.0, 5.0]),
        (8, (2.0, 8.0), 3.0, 8.0, [3.0, 5.0, 7.0]),
        (4, (1.0, 4.0), 3.0, 10.0, [1.5, 2.5, 3.5]),
        (6, (0.5, 3.0), 5.0, 15.0, [0.8, 1.5, 2.5]),
    ]

    for cfg_idx, (ns, fr, Ks, Fs, test_freqs) in enumerate(configs):
        print(f"\n  Config {cfg_idx}: {ns} sensors, freq={fr}, K={Ks}, F={Fs}")

        bank = OscillatorBank(n_sensors=ns, n_hidden=3, n_output=2,
                               freq_range=fr, K_scale=Ks,
                               F_strength=Fs, seed=42)
        print(f"    N={bank.N}, sensor ω={bank.omega[bank.sensor_ids].round(2)}")

        for input_freq in test_freqs:
            theta_star, res_free = bank.forced_equilibrium(input_freq)

            if res_free > 0.01:
                print(f"    f={input_freq:.1f}: FREE FAILED (res={res_free:.2e})")
                continue

            # Target: small perturbation from equilibrium
            rng = np.random.default_rng(42)
            target = theta_star.copy()
            for o in bank.output_ids:
                target[o] += rng.uniform(-0.3, 0.3)

            # Three gradient methods
            grad_an, J_asym = analytical_gradient_forced(
                bank, theta_star, target, input_freq)

            grad_fd = fd_gradient_forced(
                bank, theta_star, target, input_freq, eps=1e-5)

            grad_tp, res_clamp = twophase_gradient_forced(
                bank, theta_star, target, input_freq, beta=1e-3)

            if res_clamp > 0.01:
                print(f"    f={input_freq:.1f}: CLAMP FAILED (res={res_clamp:.2e})")
                continue

            cos_an_fd = cosine(grad_an, grad_fd)
            cos_tp_fd = cosine(grad_tp, grad_fd)
            cos_tp_an = cosine(grad_tp, grad_an)

            print(f"    f={input_freq:.1f}: an-fd={cos_an_fd:+.6f} "
                  f"tp-fd={cos_tp_fd:+.6f} tp-an={cos_tp_an:+.6f} "
                  f"J_asym={J_asym:.1e} res_free={res_free:.1e}")

            results.append({
                'config': cfg_idx, 'input_freq': input_freq,
                'cos_an_fd': cos_an_fd, 'cos_tp_fd': cos_tp_fd,
                'cos_tp_an': cos_tp_an, 'J_asymmetry': float(J_asym),
                'residual_free': float(res_free),
                'residual_clamp': float(res_clamp),
                'n_sensors': ns, 'N': bank.N,
            })

    print(f"\n{'='*60}")
    if results:
        print(f"Summary ({len(results)} converged test points):")
        cos_an_fds = [r['cos_an_fd'] for r in results]
        cos_tp_fds = [r['cos_tp_fd'] for r in results]
        J_asyms = [r['J_asymmetry'] for r in results]
        print(f"  an-fd: mean={np.mean(cos_an_fds):+.6f} min={min(cos_an_fds):+.6f}")
        print(f"  tp-fd: mean={np.mean(cos_tp_fds):+.6f} min={min(cos_tp_fds):+.6f}")
        print(f"  J asymmetry: max={max(J_asyms):.1e}")

        if min(cos_tp_fds) > 0.999:
            print(f"\n  *** IDENTITY HOLDS for forced Kuramoto system ***")
        elif min(cos_tp_fds) > 0.99:
            print(f"\n  ** Strong agreement — identity approximately holds **")
        else:
            print(f"\n  Partial agreement — identity may not extend cleanly")
    else:
        print("  No converged test points found.")

    with open('experiments/forced_gradient_verification.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to experiments/forced_gradient_verification.json")
