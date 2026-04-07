#!/usr/bin/env python3.12
"""E3: K-only learning rate sweep. Does K-only improve with tuned lr?"""
import json, time, numpy as np
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

N_SEEDS = 20
EPOCHS = 200
LRS = [0.0001, 0.0005, 0.001, 0.005, 0.01]

if __name__ == '__main__':
    print(f"E3: K-only LR sweep ({N_SEEDS} seeds, {EPOCHS} epochs)")
    results = {str(lr): [] for lr in LRS}
    t0 = time.time()

    for seed in range(N_SEEDS):
        row = {}
        for lr in LRS:
            tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
            net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
            hist = train(net, tr, te, lr_omega=0.0, lr_K=lr, beta=0.1,
                         epochs=EPOCHS, verbose=False, eval_every=40, seed=seed)
            acc = hist[-1]['acc']  # final-epoch, not best-during-training
            results[str(lr)].append(acc)
            row[lr] = acc
        print(f"  seed {seed:2d}: " +
              " ".join(f"lr={lr}:{row[lr]:.0%}" for lr in LRS) +
              f" ({time.time()-t0:.0f}s)", flush=True)

    # Also run omega-only at lr=0.001 for reference
    omega_ref = []
    for seed in range(N_SEEDS):
        tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
        net = make_network(seed=seed, K_scale=2.0, input_scale=1.5)
        hist = train(net, tr, te, lr_omega=0.001, lr_K=0.0, beta=0.1,
                     epochs=EPOCHS, verbose=False, eval_every=40, seed=seed)
        omega_ref.append(hist[-1]['acc']  # final-epoch, not best-during-training)

    print(f"\n{'='*70}")
    print(f"{'mode':20s} {'mean':>7s} {'conv':>6s} {'conv_mean':>10s}")
    for lr in LRS:
        accs = results[str(lr)]
        conv = [a for a in accs if a > 0.6]
        cm = f"{np.mean(conv):.1%}" if conv else "N/A"
        print(f"K lr={lr:<12} {np.mean(accs):7.1%} {len(conv):3d}/{N_SEEDS:2d} {cm:>10s}")
    conv_o = [a for a in omega_ref if a > 0.6]
    cm_o = f"{np.mean(conv_o):.1%}" if conv_o else "N/A"
    print(f"omega lr=0.001      {np.mean(omega_ref):7.1%} {len(conv_o):3d}/{N_SEEDS:2d} {cm_o:>10s}")

    results['omega_ref_0.001'] = omega_ref
    with open('experiments/k_only_lr_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
