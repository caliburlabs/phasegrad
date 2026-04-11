"""Exp A: Joint learning rate ratio sweep on 2+5+2 with spectral seeding.

Paper 1 showed omega-only > K-only at equal lr=0.001, but "both" mode
(equal lr) was slightly worse than omega-only. This was tested at exactly
one ratio. This experiment sweeps lr_omega/lr_K to find whether joint
learning ever beats either alone.

Setup:
  - Architecture: 2+5+2 (paper 1 baseline)
  - Data: Hillenbrand /a/ vs /i/ (paper 1 task)
  - Spectral seeding: multi_eigen (100% convergence for omega-only)
  - 100 seeds per ratio, 200 epochs, eval every 10 epochs
  - 7 lr ratios: anchor lr_K=0.001, vary lr_omega

Baseline check: omega-only and K-only endpoints must reproduce
paper 1 numbers (~94% and ~83% converged test acc).
"""

import json
import os
import sys
import time

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from phasegrad.kuramoto import make_network
from phasegrad.data import load_hillenbrand
from phasegrad.training import train


# ── Spectral seeding (from functional_composition.py) ────────────────

def seed_multi_eigen(net, scale=0.3):
    """Linear combination of eigenvectors, weighted by output separation / lambda."""
    N = net.N
    K = net.K
    D = np.diag(K.sum(axis=1))
    L = D - K
    L_red = L[1:, 1:]
    evals, evecs = np.linalg.eigh(L_red)
    out_red = [o - 1 for o in net.output_ids]
    combo = np.zeros(N - 1)
    for i in range(len(evals)):
        sep = evecs[out_red[0], i] - evecs[out_red[1], i]
        combo += (sep / evals[i]) * evecs[:, i]
    new_omega = np.zeros(N)
    new_omega[1:] = combo
    for inp in net.input_ids:
        new_omega[inp] = 0.0
    mx = np.max(np.abs(new_omega))
    if mx > 1e-10:
        new_omega = new_omega / mx * scale
    net.omega = new_omega


# ── Constants ─────────────────────────────────────────────────────────

N_SEEDS = 100
EPOCHS = 200
EVAL_EVERY = 10
BETA = 0.1
CONV_THRESHOLD = 0.60  # train acc threshold for convergence

# 7 lr ratio points: anchor lr_K = 0.001, vary lr_omega
# Ratio = lr_omega / lr_K (0 = K-only, inf = omega-only)
LR_CONFIGS = [
    {'label': 'K_only',    'lr_omega': 0.0,    'lr_K': 0.001, 'ratio': 0.0},
    {'label': 'ratio_0.1', 'lr_omega': 0.0001, 'lr_K': 0.001, 'ratio': 0.1},
    {'label': 'ratio_0.3', 'lr_omega': 0.0003, 'lr_K': 0.001, 'ratio': 0.3},
    {'label': 'ratio_1',   'lr_omega': 0.001,  'lr_K': 0.001, 'ratio': 1.0},
    {'label': 'ratio_3',   'lr_omega': 0.003,  'lr_K': 0.001, 'ratio': 3.0},
    {'label': 'ratio_10',  'lr_omega': 0.01,   'lr_K': 0.001, 'ratio': 10.0},
    {'label': 'omega_only', 'lr_omega': 0.001, 'lr_K': 0.0,   'ratio': float('inf')},
]


# ── Single run ────────────────────────────────────────────────────────

def run_one(seed, lr_omega, lr_K):
    """Train one seed, return per-epoch history and final metrics."""
    tr, te, _ = load_hillenbrand(vowels=['a', 'i'], seed=seed)

    net = make_network(n_input=2, n_hidden=5, n_output=2,
                       K_scale=2.0, input_scale=1.5, seed=seed)
    seed_multi_eigen(net)

    hist = train(net, tr, te,
                 lr_omega=lr_omega, lr_K=lr_K,
                 beta=BETA, epochs=EPOCHS,
                 verbose=False, eval_every=EVAL_EVERY, seed=seed)

    final = hist[-1]
    converged = final['train_acc'] > CONV_THRESHOLD

    # Epochs to 80% test accuracy
    epochs_to_80 = None
    for h in hist:
        if h['acc'] >= 0.80:
            epochs_to_80 = h.get('epoch', None)
            # train() history entries may not have 'epoch' key —
            # infer from position: entry i corresponds to epoch (i+1)*eval_every
            if epochs_to_80 is None:
                idx = hist.index(h)
                epochs_to_80 = (idx + 1) * EVAL_EVERY
            break

    return {
        'seed': seed,
        'test_acc': final['acc'],
        'train_acc': final['train_acc'],
        'converged': converged,
        'epochs_to_80': epochs_to_80,
        'history': [{'acc': h['acc'], 'train_acc': h['train_acc']}
                    for h in hist],
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 78)
    print("EXP A: LR RATIO SWEEP — 2+5+2 + spectral seeding + Hillenbrand a/i")
    print("=" * 78)
    print(f"  {N_SEEDS} seeds × {len(LR_CONFIGS)} ratios = "
          f"{N_SEEDS * len(LR_CONFIGS)} training runs")
    print(f"  {EPOCHS} epochs each, eval every {EVAL_EVERY}")
    print()

    all_results = {}
    t0 = time.time()

    for cfg in LR_CONFIGS:
        label = cfg['label']
        print(f"--- {label} (lr_omega={cfg['lr_omega']}, "
              f"lr_K={cfg['lr_K']}) ---")
        results = []

        for seed in range(N_SEEDS):
            r = run_one(seed, cfg['lr_omega'], cfg['lr_K'])
            results.append(r)

            if (seed + 1) % 25 == 0:
                elapsed = time.time() - t0
                conv_so_far = sum(1 for r in results if r['converged'])
                mean_acc = np.mean([r['test_acc'] for r in results])
                print(f"  {seed+1}/{N_SEEDS}  "
                      f"conv={conv_so_far}/{seed+1}  "
                      f"mean_test={mean_acc:.1%}  "
                      f"({elapsed:.0f}s)", flush=True)

        all_results[label] = results

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s "
          f"({total_time/60:.1f}min)")

    # ── Summary table ─────────────────────────────────────────────────

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(f"\n{'config':>12s} {'ratio':>7s} {'conv':>8s} "
          f"{'conv_test':>12s} {'conv_train':>12s} "
          f"{'all_test':>12s} {'med_e80':>8s}")
    print("-" * 78)

    summary = []
    for cfg in LR_CONFIGS:
        label = cfg['label']
        results = all_results[label]

        test_accs = [r['test_acc'] for r in results]
        conv_mask = [r['converged'] for r in results]
        n_conv = sum(conv_mask)
        conv_test = [r['test_acc'] for r in results if r['converged']]
        conv_train = [r['train_acc'] for r in results if r['converged']]
        epochs_80 = [r['epochs_to_80'] for r in results
                     if r['epochs_to_80'] is not None]

        ct = f"{np.mean(conv_test):.1%}±{np.std(conv_test):.1%}" \
            if conv_test else "N/A"
        ctr = f"{np.mean(conv_train):.1%}±{np.std(conv_train):.1%}" \
            if conv_train else "N/A"
        at = f"{np.mean(test_accs):.1%}±{np.std(test_accs):.1%}"
        med_e80 = f"{int(np.median(epochs_80))}" if epochs_80 else "N/A"
        ratio_str = f"{cfg['ratio']:.1f}" if cfg['ratio'] != float('inf') \
            else "inf"

        print(f"{label:>12s} {ratio_str:>7s} "
              f"{n_conv:>3d}/{N_SEEDS:<3d}  "
              f"{ct:>12s} {ctr:>12s} {at:>12s} {med_e80:>8s}")

        summary.append({
            'label': label,
            'ratio': cfg['ratio'] if cfg['ratio'] != float('inf') else 'inf',
            'lr_omega': cfg['lr_omega'],
            'lr_K': cfg['lr_K'],
            'n_converged': n_conv,
            'conv_test_mean': float(np.mean(conv_test)) if conv_test else None,
            'conv_test_std': float(np.std(conv_test)) if conv_test else None,
            'conv_train_mean': float(np.mean(conv_train)) if conv_train else None,
            'all_test_mean': float(np.mean(test_accs)),
            'all_test_std': float(np.std(test_accs)),
            'median_epochs_to_80': int(np.median(epochs_80)) if epochs_80 else None,
        })

    # ── Statistical tests ─────────────────────────────────────────────

    print("\n" + "=" * 78)
    print("PAIRWISE TESTS: each ratio vs omega-only (converged test acc)")
    print("=" * 78)

    omega_conv_test = [r['test_acc'] for r in all_results['omega_only']
                       if r['converged']]

    for cfg in LR_CONFIGS:
        label = cfg['label']
        if label == 'omega_only':
            continue
        other_conv_test = [r['test_acc'] for r in all_results[label]
                           if r['converged']]
        if len(other_conv_test) < 5:
            print(f"  {label}: too few converged ({len(other_conv_test)})")
            continue

        t, p_t = ttest_ind(omega_conv_test, other_conv_test, equal_var=False)
        u, p_u = mannwhitneyu(omega_conv_test, other_conv_test,
                              alternative='two-sided')
        diff = np.mean(omega_conv_test) - np.mean(other_conv_test)
        print(f"  omega_only vs {label:>12s}: "
              f"diff={diff:+.1%}  "
              f"Welch p={p_t:.2e}  "
              f"MW p={p_u:.2e}")

    # ── Same-seed tracking ────────────────────────────────────────────

    print("\n" + "=" * 78)
    print("SAME-SEED TRACKING: how many seeds converge at ALL ratios?")
    print("=" * 78)

    labels = [cfg['label'] for cfg in LR_CONFIGS]
    for seed in range(N_SEEDS):
        conv_at = [label for label in labels
                   if all_results[label][seed]['converged']]
        fail_at = [label for label in labels
                   if not all_results[label][seed]['converged']]
        if len(fail_at) > 0 and len(conv_at) > 0:
            # Mixed — interesting seeds
            pass  # will summarize below

    # Count seeds by convergence pattern
    always_conv = sum(1 for seed in range(N_SEEDS)
                      if all(all_results[l][seed]['converged'] for l in labels))
    never_conv = sum(1 for seed in range(N_SEEDS)
                     if not any(all_results[l][seed]['converged'] for l in labels))
    mixed = N_SEEDS - always_conv - never_conv

    print(f"  Always converge (all 7 ratios): {always_conv}/{N_SEEDS}")
    print(f"  Never converge (no ratio):      {never_conv}/{N_SEEDS}")
    print(f"  Mixed (some ratios only):        {mixed}/{N_SEEDS}")

    # ── Save ──────────────────────────────────────────────────────────

    outpath = os.path.join(os.path.dirname(__file__),
                           'lr_ratio_sweep_results.json')

    # Strip history for compact save (keep per-seed final metrics)
    save_results = {}
    for label, results in all_results.items():
        save_results[label] = []
        for r in results:
            save_results[label].append({
                'seed': r['seed'],
                'test_acc': r['test_acc'],
                'train_acc': r['train_acc'],
                'converged': r['converged'],
                'epochs_to_80': r['epochs_to_80'],
            })

    with open(outpath, 'w') as f:
        json.dump({
            'summary': summary,
            'results': save_results,
            'config': {
                'n_seeds': N_SEEDS,
                'epochs': EPOCHS,
                'eval_every': EVAL_EVERY,
                'beta': BETA,
                'convergence_threshold': CONV_THRESHOLD,
                'lr_configs': [{k: v for k, v in c.items()
                                if k != 'ratio' or v != float('inf')}
                               for c in LR_CONFIGS],
                'architecture': '2+5+2',
                'data': 'hillenbrand_a_i',
                'seeding': 'multi_eigen',
            },
        }, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
