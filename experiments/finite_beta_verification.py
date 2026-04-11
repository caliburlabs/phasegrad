#!/usr/bin/env python3
"""Finite-beta verification: gradient quality at training-scale beta.

Backs the claim in Section 4.3: at beta=0.1 (training value), the gradient
direction is correct (cosine > 0.999) with a mean scale error of ~0.3%
across 20 random networks.
"""
import json
import time

import numpy as np

from phasegrad.kuramoto import make_random_network
from phasegrad.gradient import two_phase_gradient, analytical_gradient
from phasegrad.losses import mse_target


def verify_finite_beta(N=15, beta=0.1, n_networks=20, seed_base=0):
    """Measure gradient quality at finite beta across random networks."""
    results = []

    for i in range(n_networks):
        seed = seed_base + i
        net = make_random_network(N=N, K_mean=5.0, omega_spread=0.3,
                                  connectivity=0.6, seed=seed)
        theta_star, res = net.equilibrium()
        if res > 1e-4:
            continue

        target = mse_target(net.N, net.output_ids, 0, margin=0.2)

        # Two-phase at training beta
        grad_tp, _, _ = two_phase_gradient(net, theta_star, target, beta=beta)

        # Analytical (exact)
        grad_an = analytical_gradient(net, theta_star, target)

        # Compare on non-pinned nodes
        idx = list(range(1, net.N))
        tp, an = grad_tp[idx], grad_an[idx]

        ntp, nan_ = np.linalg.norm(tp), np.linalg.norm(an)
        if nan_ < 1e-12 or ntp < 1e-12:
            continue

        cosine = float(np.dot(tp, an) / (ntp * nan_))
        scale = float(ntp / nan_)
        scale_error = abs(scale - 1.0)

        results.append({
            'seed': seed,
            'N': N,
            'beta': beta,
            'cosine': cosine,
            'scale': scale,
            'scale_error_pct': scale_error * 100,
            'residual': float(res),
        })

    return results


if __name__ == '__main__':
    print("Finite-beta verification: beta=0.1, N=15, 20 networks")
    t0 = time.time()

    results = verify_finite_beta(N=15, beta=0.1, n_networks=20, seed_base=0)

    cosines = [r['cosine'] for r in results]
    scale_errors = [r['scale_error_pct'] for r in results]

    print(f"  Networks tested: {len(results)}")
    print(f"  Cosine:      min={min(cosines):.6f}  mean={np.mean(cosines):.6f}")
    print(f"  Scale error:  mean={np.mean(scale_errors):.1f}%  "
          f"std={np.std(scale_errors):.1f}%")
    print(f"  All cosines > 0.999: {all(c > 0.999 for c in cosines)}")
    print(f"  Time: {time.time() - t0:.1f}s")

    summary = {
        'beta': 0.1,
        'N': 15,
        'n_networks': len(results),
        'mean_cosine': float(np.mean(cosines)),
        'min_cosine': float(min(cosines)),
        'mean_scale_error_pct': float(np.mean(scale_errors)),
        'std_scale_error_pct': float(np.std(scale_errors)),
        'all_above_0999': all(c > 0.999 for c in cosines),
        'per_network': results,
    }

    out = 'experiments/finite_beta_results.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")
