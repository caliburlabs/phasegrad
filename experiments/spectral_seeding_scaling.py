#!/usr/bin/env python3.12
"""Spectral seeding scaling test.

Sweep network size N to test whether multi_eigen keeps working as
the reduced Laplacian grows. Log success rate, converged accuracy,
seeding time, and total training time separately.

Tests:
  - N = 9 (2+5+2), 12 (2+8+2), 17 (2+13+2), 22 (2+18+2), 32 (2+28+2)
  - 30 seeds per size
  - random vs multi_eigen
  - ω-only training, 200 epochs
"""
import json
import time

import numpy as np

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

from experiments.spectral_seeding import seed_random, seed_multi_eigen

SIZES = [
    (2, 5, 2),    # N=9
    (2, 8, 2),    # N=12
    (2, 13, 2),   # N=17
    (2, 18, 2),   # N=22
    (2, 28, 2),   # N=32
]

N_SEEDS = 30
EPOCHS = 200


def run_scaling():
    results = {}

    for n_in, n_hid, n_out in SIZES:
        N = n_in + n_hid + n_out
        key = f"{n_in}+{n_hid}+{n_out}"
        print(f"\n=== N={N} ({key}) ===", flush=True)
        results[key] = {}

        for strat_name, strat_fn in [('random', seed_random),
                                      ('multi_eigen', seed_multi_eigen)]:
            accs = []
            seed_times = []
            train_times = []

            for seed in range(N_SEEDS):
                net = make_network(n_input=n_in, n_hidden=n_hid, n_output=n_out,
                                   seed=seed, K_scale=2.0, input_scale=1.5)
                tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

                rng = np.random.default_rng(seed + 10000)

                t0 = time.perf_counter()
                strat_fn(net, rng)
                seed_time = time.perf_counter() - t0
                seed_times.append(seed_time)

                t0 = time.perf_counter()
                hist = train(net, tr, te, lr_omega=0.001, lr_K=0.0, beta=0.1,
                             epochs=EPOCHS, verbose=False, eval_every=EPOCHS,
                             seed=seed)
                train_time = time.perf_counter() - t0
                train_times.append(train_time)

                accs.append(hist[-1]['acc'])

            conv = sum(1 for a in accs if a > 0.60)
            conv_accs = [a for a in accs if a > 0.60]
            conv_mean = f"{np.mean(conv_accs):.1%}" if conv_accs else "N/A"
            mean_seed_ms = np.mean(seed_times) * 1000
            mean_train_s = np.mean(train_times)

            print(f"  {strat_name:15s}: {conv}/{N_SEEDS} conv, "
                  f"conv_acc={conv_mean}, "
                  f"seed={mean_seed_ms:.2f}ms, "
                  f"train={mean_train_s:.1f}s", flush=True)

            results[key][strat_name] = {
                'accs': accs,
                'conv': conv,
                'conv_accs': conv_accs,
                'seed_times_ms': [t * 1000 for t in seed_times],
                'train_times_s': train_times,
            }

    return results


if __name__ == '__main__':
    results = run_scaling()

    print("\n" + "=" * 75)
    print(f"{'Size':>10s}  {'N':>3s}  {'Strategy':>12s}  {'Conv':>7s}  "
          f"{'Acc':>7s}  {'Seed(ms)':>9s}  {'Train(s)':>9s}")
    print("-" * 75)
    for key, strats in results.items():
        parts = key.split('+')
        N = sum(int(p) for p in parts)
        for sname, sdata in strats.items():
            conv = sdata['conv']
            ca = sdata['conv_accs']
            ca_str = f"{np.mean(ca):.1%}" if ca else "N/A"
            seed_ms = np.mean(sdata['seed_times_ms'])
            train_s = np.mean(sdata['train_times_s'])
            print(f"{key:>10s}  {N:3d}  {sname:>12s}  "
                  f"{conv:2d}/{N_SEEDS:2d}   {ca_str:>7s}  "
                  f"{seed_ms:9.3f}  {train_s:9.1f}")

    out_path = 'experiments/spectral_seeding_scaling_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
