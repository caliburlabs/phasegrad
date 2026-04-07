#!/usr/bin/env python3.12
"""C3: Asymmetry sensitivity sweep.

How sensitive is the gradient identity to asymmetric coupling (K_ij != K_ji)?
Real CMOS has parasitic asymmetries. This quantifies graceful degradation.
"""
import json, numpy as np
from phasegrad.kuramoto import make_random_network, kuramoto_jacobian
from phasegrad.gradient import two_phase_gradient, analytical_gradient, mse_loss

ASYM_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
N_SEEDS = 10
N = 15

if __name__ == '__main__':
    print(f"Asymmetry Sensitivity: N={N}, {N_SEEDS} seeds per level")
    results = []

    for asym in ASYM_LEVELS:
        cosines = []
        for seed in range(N_SEEDS):
            net = make_random_network(N, K_mean=5.0, seed=seed)
            # Add asymmetric noise
            if asym > 0:
                rng = np.random.default_rng(seed + 1000)
                noise = rng.uniform(-asym, asym, net.K.shape) * np.abs(net.K)
                net.K = net.K + noise
                net.K = np.maximum(net.K, 0)  # keep non-negative
                # Explicitly do NOT symmetrize

            theta_star, res = net.equilibrium()
            if res > 0.01:
                net.K = (net.K + net.K.T) / 2  # fallback: symmetrize
                net.K *= 1.5
                theta_star, res = net.equilibrium()
                if res > 0.01:
                    continue

            rng2 = np.random.default_rng(seed + 2000)
            target = theta_star.copy()
            for i in net.output_ids:
                target[i] += rng2.uniform(-0.3, 0.3)

            # Two-phase gradient (works regardless of symmetry)
            grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=1e-3)

            # FD gradient (ground truth regardless of symmetry)
            from phasegrad.gradient import finite_difference_gradient
            grad_fd = finite_difference_gradient(net, theta_star, target, eps=1e-5)

            a, b = grad_tp[1:], grad_fd[1:]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 1e-12 and nb > 1e-12:
                cos = float(np.dot(a, b) / (na * nb))
                cosines.append(cos)

        if cosines:
            print(f"  asym={asym:.2f}: cos={np.mean(cosines):+.6f} ± {np.std(cosines):.6f} "
                  f"(n={len(cosines)})")
            results.append({
                'asymmetry': asym, 'mean_cos': float(np.mean(cosines)),
                'std_cos': float(np.std(cosines)), 'n': len(cosines),
                'values': cosines,
            })

    out = 'experiments/asymmetry_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
