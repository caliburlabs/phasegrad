#!/usr/bin/env python3.12
"""C4: Phase measurement noise (TDC resolution).

Sweeps both TDC noise level and beta to find the practical operating point.
Key insight: larger beta → larger phase displacement → better SNR,
but larger O(β) approximation error. There's a sweet spot.
"""
import json, numpy as np
from phasegrad.kuramoto import make_random_network
from phasegrad.gradient import analytical_gradient

NOISE_PS = [0, 1, 5, 10, 20, 50, 100, 200, 500]
FREQ_GHZ = 3.0
N_SEEDS = 20
N = 10
BETAS = [0.1, 0.01, 0.001]

if __name__ == '__main__':
    print(f"TDC Noise Sensitivity: N={N}, {N_SEEDS} seeds, freq={FREQ_GHZ}GHz")
    results = []

    for beta in BETAS:
        print(f"\n  β = {beta}")
        for noise_ps in NOISE_PS:
            noise_rad = 2 * np.pi * FREQ_GHZ * 1e9 * noise_ps * 1e-12
            cosines = []

            for seed in range(N_SEEDS):
                net = make_random_network(N, K_mean=5.0, seed=seed)
                theta_star, res = net.equilibrium()
                if res > 0.01:
                    continue

                rng = np.random.default_rng(seed + 100)
                target = theta_star.copy()
                for i in net.output_ids:
                    target[i] += rng.uniform(-0.3, 0.3)

                grad_clean = analytical_gradient(net, theta_star, target)

                theta_clamp, _ = net.clamped_equilibrium(
                    beta, target, theta_star.copy())

                # Clean two-phase gradient (for reference)
                grad_tp_clean = -(theta_clamp - theta_star) / beta

                # Noisy measurements
                rng_f = np.random.default_rng(seed * 10000 + noise_ps * 100 + 1)
                rng_c = np.random.default_rng(seed * 10000 + noise_ps * 100 + 2)
                ts_noisy = theta_star + rng_f.normal(0, noise_rad, theta_star.shape)
                tc_noisy = theta_clamp + rng_c.normal(0, noise_rad, theta_clamp.shape)
                grad_noisy = -(tc_noisy - ts_noisy) / beta

                a, b = grad_noisy[1:], grad_clean[1:]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-12 and nb > 1e-12:
                    cosines.append(float(np.dot(a, b) / (na * nb)))

            if cosines:
                # Also compute typical displacement for context
                disp = float(np.linalg.norm(theta_clamp - theta_star))
                print(f"    noise={noise_ps:4d}ps ({noise_rad:.3f}rad): "
                      f"cos={np.mean(cosines):+.4f} ± {np.std(cosines):.4f}")
                results.append({
                    'beta': beta, 'noise_ps': noise_ps,
                    'noise_rad': float(noise_rad),
                    'mean_cos': float(np.mean(cosines)),
                    'std_cos': float(np.std(cosines)),
                    'n': len(cosines),
                })

    out = 'experiments/tdc_noise_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
