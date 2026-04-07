#!/usr/bin/env python3.12
"""Omega-only learning rate sweep — mirror of k_only_lr_sweep.py."""
import json, time, numpy as np
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

N_SEEDS = 20
EPOCHS = 200
LRS = [0.0001, 0.0005, 0.001, 0.005, 0.01]

if __name__ == '__main__':
    print(f"Omega-only LR sweep ({N_SEEDS} seeds, {EPOCHS} epochs)")
    results = {str(lr): [] for lr in LRS}
    t0 = time.time()

    for seed in range(N_SEEDS):
        row = {}
        for lr in LRS:
            tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
            net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
            hist = train(net, tr, te, lr_omega=lr, lr_K=0.0, beta=0.1,
                         epochs=EPOCHS, verbose=False, eval_every=40, seed=seed)
            acc = hist[-1]['acc']  # final-epoch, not best-during-training
            results[str(lr)].append(acc)
            row[lr] = acc
        print(f"  seed {seed:2d}: " +
              " ".join(f"lr={lr}:{row[lr]:.0%}" for lr in LRS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'='*70}")
    print(f"{'mode':20s} {'mean':>7s} {'conv':>6s} {'conv_mean':>10s}")
    for lr in LRS:
        accs = results[str(lr)]
        conv = [a for a in accs if a > 0.6]
        cm = f"{np.mean(conv):.1%}" if conv else "N/A"
        print(f"omega lr={lr:<12} {np.mean(accs):7.1%} {len(conv):3d}/{N_SEEDS:2d} {cm:>10s}")

    with open('experiments/omega_only_lr_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to experiments/omega_only_lr_sweep_results.json")
