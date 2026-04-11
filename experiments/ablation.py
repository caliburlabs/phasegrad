#!/usr/bin/env python3.12
"""C5: Omega-only vs K-only vs both ablation.

This is the critical experiment for differentiating from Wang et al. 2024.
They learn coupling (K) only. We learn frequency (omega). Does omega-only work?
"""
import json, time, numpy as np
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

N_SEEDS = 20
EPOCHS = 200

def run_ablation(mode, seed):
    """Train with only omega, only K, or both."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)

    if mode == 'omega_only':
        lr_w, lr_K = 0.001, 0.0
    elif mode == 'K_only':
        lr_w, lr_K = 0.0, 0.001
    else:  # both
        lr_w, lr_K = 0.001, 0.001

    hist = train(net, tr, te, lr_omega=lr_w, lr_K=lr_K, beta=0.1,
                 epochs=EPOCHS, verbose=False, eval_every=20, seed=seed)
    return hist[-1]['acc']  # final-epoch accuracy, not best-during-training

if __name__ == '__main__':
    print(f"Ablation: omega-only vs K-only vs both ({N_SEEDS} seeds, {EPOCHS} epochs)")
    modes = ['omega_only', 'K_only', 'both']
    results = {m: [] for m in modes}
    t0 = time.time()

    for seed in range(N_SEEDS):
        row = {}
        for mode in modes:
            acc = run_ablation(mode, seed)
            results[mode].append(acc)
            row[mode] = acc
        print(f"  seed {seed:2d}: omega={row['omega_only']:.1%} "
              f"K={row['K_only']:.1%} both={row['both']:.1%} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*60}")
    print(f"{'mode':15s} {'mean':>8s} {'std':>8s} {'conv':>6s} {'conv_mean':>10s}")
    for mode in modes:
        accs = results[mode]
        conv = [a for a in accs if a > 0.60]
        print(f"{mode:15s} {np.mean(accs):8.1%} {np.std(accs):8.1%} "
              f"{len(conv):3d}/{N_SEEDS:2d} "
              f"{np.mean(conv):10.1%}" if conv else f"{0:10.1%}")

    out = 'experiments/ablation_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out}")
