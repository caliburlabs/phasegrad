#!/usr/bin/env python3.12
"""C6: Scale verification N=100, N=200.

Extend Table 1 to larger networks. Address "does it scale?" question.
"""
import json, time
from phasegrad.verification import verify_single

SIZES = [6, 10, 15, 20, 30, 50, 100, 200]

if __name__ == '__main__':
    print(f"Scale Verification: N = {SIZES}")
    results = []

    for N in SIZES:
        t0 = time.time()
        r = verify_single(N, n_beta=8, seed=42)
        elapsed = time.time() - t0

        print(f"  N={N:4d}: an-fd={r['cos_an_fd']:+.6f} "
              f"tp-fd={r['best_cos_tp_fd']:+.6f} "
              f"res={r['residual']:.1e} "
              f"time={elapsed:.1f}s", flush=True)

        results.append({
            'N': N,
            'cos_an_fd': r['cos_an_fd'],
            'cos_tp_fd': r['best_cos_tp_fd'],
            'best_beta': r['best_beta'],
            'residual': r['residual'],
            'time_s': elapsed,
        })

    out = 'experiments/scale_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")
