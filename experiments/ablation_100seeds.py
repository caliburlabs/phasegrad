#!/usr/bin/env python3.12
"""D2: Ablation with 100 seeds for statistical power.

Methodology:
  - Convergence defined by final-epoch TRAIN accuracy > 0.6
    (never uses test accuracy for selection decisions)
  - Reports final-epoch TEST accuracy for converged seeds
  - Normalization uses train-set statistics only (fixed in data.py)
"""
import json, time, numpy as np
from scipy.stats import fisher_exact, ttest_ind, mannwhitneyu
from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train

N_SEEDS = 100
EPOCHS = 150

def run(mode, seed):
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)
    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)
    if mode == 'omega_only':
        lr_w, lr_K = 0.001, 0.0
    elif mode == 'K_only':
        lr_w, lr_K = 0.0, 0.001
    else:
        lr_w, lr_K = 0.001, 0.001
    hist = train(net, tr, te, lr_omega=lr_w, lr_K=lr_K, beta=0.1,
                 epochs=EPOCHS, verbose=False, eval_every=30, seed=seed)
    final = hist[-1]
    return {
        'test_acc': final['acc'],
        'train_acc': final['train_acc'],
    }

if __name__ == '__main__':
    print(f"D2: Ablation 100 seeds x 3 modes x {EPOCHS} epochs")
    print("Convergence defined by final-epoch TRAIN acc > 0.6")
    print("=" * 70)
    modes = ['omega_only', 'K_only', 'both']
    results = {m: [] for m in modes}
    t0 = time.time()

    for seed in range(N_SEEDS):
        for mode in modes:
            r = run(mode, seed)
            results[mode].append(r)
        if (seed + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {seed+1}/{N_SEEDS} done ({elapsed:.0f}s)", flush=True)

    # --- Analysis ---
    print(f"\n{'='*70}")
    print(f"{'mode':15s}  {'all-seed test':>14s}  {'conv rate':>10s}  "
          f"{'conv test':>14s}  {'conv train':>14s}")
    for mode in modes:
        test_accs = [r['test_acc'] for r in results[mode]]
        train_accs = [r['train_acc'] for r in results[mode]]
        # Convergence from TRAIN accuracy only
        conv_mask = [r['train_acc'] > 0.60 for r in results[mode]]
        conv_test = [r['test_acc'] for r, c in zip(results[mode], conv_mask) if c]
        conv_train = [r['train_acc'] for r, c in zip(results[mode], conv_mask) if c]
        n_conv = sum(conv_mask)
        ct = f"{np.mean(conv_test):.1%}+/-{np.std(conv_test):.1%}" if conv_test else "N/A"
        ctr = f"{np.mean(conv_train):.1%}+/-{np.std(conv_train):.1%}" if conv_train else "N/A"
        print(f"{mode:15s}  {np.mean(test_accs):14.1%}  "
              f"{n_conv:3d}/{N_SEEDS:3d}     {ct:>14s}  {ctr:>14s}")

    # Fisher's exact test: omega vs K convergence rates (train-defined)
    o_conv = sum(1 for r in results['omega_only'] if r['train_acc'] > 0.60)
    k_conv = sum(1 for r in results['K_only'] if r['train_acc'] > 0.60)
    table = [[o_conv, N_SEEDS - o_conv], [k_conv, N_SEEDS - k_conv]]
    odds, p_fisher = fisher_exact(table)
    print(f"\nFisher's exact (omega vs K convergence, train-defined):")
    print(f"  omega: {o_conv}/{N_SEEDS}, K: {k_conv}/{N_SEEDS}")
    print(f"  odds ratio: {odds:.2f}, p = {p_fisher:.4f}")

    # Converged TEST accuracy comparison
    o_conv_test = [r['test_acc'] for r in results['omega_only'] if r['train_acc'] > 0.60]
    k_conv_test = [r['test_acc'] for r in results['K_only'] if r['train_acc'] > 0.60]
    if len(o_conv_test) >= 5 and len(k_conv_test) >= 5:
        t, p_welch = ttest_ind(o_conv_test, k_conv_test, equal_var=False)
        u, p_mann = mannwhitneyu(o_conv_test, k_conv_test, alternative='two-sided')
        print(f"\nConverged test accuracy (convergence defined by train acc > 0.6):")
        print(f"  omega: {np.mean(o_conv_test):.1%}+/-{np.std(o_conv_test):.1%} (n={len(o_conv_test)})")
        print(f"  K:     {np.mean(k_conv_test):.1%}+/-{np.std(k_conv_test):.1%} (n={len(k_conv_test)})")
        print(f"  Welch's t={t:.2f}, p={p_welch:.2e}")
        print(f"  Mann-Whitney U={u:.0f}, p={p_mann:.2e}")

    # All-seed comparison (no selection)
    all_o = [r['test_acc'] for r in results['omega_only']]
    all_k = [r['test_acc'] for r in results['K_only']]
    t_all, p_all = ttest_ind(all_o, all_k, equal_var=False)
    u_all, p_all_mw = mannwhitneyu(all_o, all_k, alternative='two-sided')
    print(f"\nAll-seed test accuracy (no filtering):")
    print(f"  omega: {np.mean(all_o):.1%}+/-{np.std(all_o):.1%}")
    print(f"  K:     {np.mean(all_k):.1%}+/-{np.std(all_k):.1%}")
    print(f"  Welch's t={t_all:.2f}, p={p_all:.2e}")
    print(f"  Mann-Whitney U={u_all:.0f}, p={p_all_mw:.2e}")

    # Save everything
    serializable = {}
    for mode in modes:
        serializable[mode] = results[mode]
    with open('experiments/ablation_100seeds_results.json', 'w') as f:
        json.dump({
            'results': serializable,
            'convergence_criterion': 'final_epoch_train_acc > 0.6',
            'metric': 'final_epoch_test_acc',
        }, f, indent=2, default=str)
    print(f"\nSaved to experiments/ablation_100seeds_results.json")
